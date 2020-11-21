#
# 
# This above is a jupyter command to save this cell in python format! (.py)
#
"""
Script to parse regular human-readable output of projwfc.x.

Basically scopiazzato da:
https://aiida-quantumespresso.readthedocs.io/en/latest/_modules/aiida_quantumespresso/parsers/projwfc.html#ProjwfcParser
"""

import numpy as np #Can use numpy arrays
import re #Can use regular expressions
import sys
import argparse #Parser for command line inputs

# Functions
def get_line_numbers(lines):
    """
    Store line numbers where (i) kpts, (ii) energies, (iii) psi^2, (iv) states appear
    """
    k_lines = []
    e_lines = []
    psi_lines = []
    wfc_lines = []

    for i, line in enumerate(lines):
        if 'k =' in line:
            k_lines.append(i)
        if '==== e(' in line or line.strip().startswith('e ='):
            # The energy format in output was changed in QE6.3
            # this check supports old and new format
            e_lines.append(i)
        if '|psi|^2' in line:
            psi_lines.append(i)
        if 'state #' in line:
            wfc_lines.append(i)
    return k_lines,e_lines,psi_lines,wfc_lines

def get_state_info(lines,wfc_lines):
    """
    Build python dictionaries for state info like this:
    - One dictionary per atomic orbital (N_states in total)
    - Keys are 'atomnum', 'kind_name', 'angular_momentum', 'magnetic_number'
    """
    #Regular expressions 
    atomnum_re = re.compile(r'atom\s*([0-9]+?)[^0-9]')
    element_re = re.compile(r'atom\s*[0-9]+\s*\(\s*([A-Za-z]+?)\s*\)')
    lnum_re = re.compile(r'l=\s*([0-9]+?)[^0-9]')
    mnum_re = re.compile(r'm=\s*([-0-9]+?)[^-0-9]')

    state_lines = [lines[wfc_line] for wfc_line in wfc_lines]
    state_dicts = []
    for state_line in state_lines:
        state_dict = {}
        state_dict['atomnum'] = int(atomnum_re.findall(state_line)[0])
        state_dict['atomnum'] -= 1  # to keep with orbital indexing
        state_dict['kind_name'] = element_re.findall(state_line)[0].strip()
        state_dict['angular_momentum'] = int(lnum_re.findall(state_line)[0])
        state_dict['magnetic_number'] = int(mnum_re.findall(state_line)[0])
        state_dict['magnetic_number'] -= 1  # to keep with orbital indexing
        state_dicts.append(state_dict)
    return state_dicts

def get_elements(state_dicts):
    """
    Get list of elements present in the system
    """
    elements = []
    for state_dict in state_dicts:
        el = state_dict['kind_name']
        if el not in elements: elements.append(el)
    return elements

def get_orbitals(state_dicts):
    """
    Get list of orbitals present in the system
    """
    orbitals = []
    orbitals_labels = []
    for state_dict in state_dicts:
        orb = state_dict['angular_momentum']
        if orb==0: orb_label = 's'
        if orb==1: orb_label = 'p'
        if orb==2: orb_label = 'd'
        if orb==3: orb_label = 'f'
        if orb not in orbitals: orbitals.append(orb)
        if orb_label not in orbitals_labels: orbitals_labels.append(orb_label)
    return orbitals,orbitals_labels

def get_species_orbitals(elements_list,state_dicts):
    """
    Get list of species-orbitals (each element with the orbitals it appears with)
    """
    sp_orb = []
    for el in elements_list:
        l=-1
        for state_dict in state_dicts:
            el_tmp = state_dict['kind_name']
            l_tmp  = state_dict['angular_momentum']
            if el==el_tmp and l_tmp>l: l=l_tmp
        list_tmp = [ '%s-s'%el, '%s-p'%el, '%s-d'%el, '%s-s'%el ]
        for il in range(l+1): sp_orb.append(list_tmp[il])
                
    return sp_orb
           
def get_linear_combinations(k_lines,e_lines,psi_lines,Nb):
    """
    Read orbital composition for each (nk) band state.
    """
    #Regular expressions
    WaveFraction1_re = re.compile(r'\=(.*?)\*')  
    WaveFractionremain_re = re.compile(r'\+(.*?)\*')
    FunctionId_re = re.compile(r'\#(.*?)\]')
    
    which_k = [lines[k_line ] for k_line in k_lines ]
    proj_wfc_weight = [] #np.zeros([len(k_lines),len(e_lines)])
    proj_wfc_index  = [] #np.zeros([len(k_lines),len(e_lines)],dtype=np.int)
    for k in range(len(k_lines)): #kpoints
        for n in range(k * Nb, (k + 1) * Nb): #band states
            #subloop grabs pdos
            wave_fraction = []
            wave_id = []
            for nk in range(e_lines[n] + 1, psi_lines[n]): #Loop over orbital for each nk state
                line = lines[nk]
                wave_fraction += WaveFraction1_re.findall(line)
                wave_fraction += WaveFractionremain_re.findall(line)
                wave_id += FunctionId_re.findall(line)
            if len(wave_id) != len(wave_fraction):
                raise IndexError
            for l in range(len(wave_id)):
                wave_id[l] = int(wave_id[l]) 
                wave_id[l] -= 1 # to keep with orbital indexing
                wave_fraction[l] = float(wave_fraction[l])
            proj_wfc_weight.append(wave_fraction)
            proj_wfc_index.append(wave_id)  
    proj_wfc_weight = np.array(proj_wfc_weight,dtype=object).reshape(len(k_lines),Nb)
    proj_wfc_index = np.array(proj_wfc_index,dtype=object).reshape(len(k_lines),Nb)      
        
    return proj_wfc_weight,proj_wfc_index

def get_plot_elements(bloch_state_indices, bloch_state_weights, state_dicts, elements_list):
    """
    Get (x,y) histogram data for plot in the 'elements' case.
    """
    x = np.array([i for i in range(len(elements_list))])
    y = np.zeros(len(elements_list))

    for s,state in enumerate(bloch_state_indices): #Each orbital in the linear combination
        el_tmp = state_dicts[state]['kind_name'] #Element corresponding to that orbital
        for i, el in enumerate(elements_list): #Check position in elements list
            if el == el_tmp: #Accumulate element-orbital weight in correct position and go back to states loop 
                y[i]+=bloch_state_weights[s]
                break
    return x,y

def get_plot_atoms(bloch_state_indices, bloch_state_weights, state_dicts, Nat, elements_list):
    """
    Get (x,y) histogram data for plot in the 'atoms' case.
    """
    x = np.array([i for i in range(Nat)])
    y = np.zeros(Nat)
    at_list = []
    for s,state in enumerate(bloch_state_indices): #Each orbital in the linear combination
        at_ind = state_dicts[state]['atomnum'] #atomic index corresponding to that orbital
        y[at_ind]+=bloch_state_weights[s] #Accumulate atomic weight

    for at in range(Nat): #Get species-index list for plot labels
        for state_dict in state_dicts:
            at_label = state_dict['kind_name']
            at_nmbr  = state_dict['atomnum']
            at_str = '%d\n%s'%(at_nmbr,at_label)
            if at_nmbr==at and at_str not in at_list: at_list.append(at_str)
    return x,y,at_list

def get_plot_orbitals(bloch_state_indices, bloch_state_weights, state_dicts, orbitals_list):
    """
    Get (x,y) histogram data for plot in the 'orbitals' case.
    """
    x = np.array([i for i in range(len(orbitals_list))])
    y = np.zeros(len(orbitals_list))

    for s,state in enumerate(bloch_state_indices): #Each orbital in the linear combination
        orb_tmp = state_dicts[state]['angular_momentum'] #l value corresponding to that orbital
        for i, orb in enumerate(orbitals_list): #Check position in elements list
            if orb == orb_tmp: #Accumulate l-orbital weight in correct position and go back to states loop
                y[i]+=bloch_state_weights[s]
                break
                
    return x,y

def plot_histogram(x,y,plt_type,nk,label_list,file_nm,save=False):
    """
    Generate histogram
    """
    import matplotlib.pyplot as plt
    if plt_type=='atoms':
        from matplotlib.ticker import FixedLocator, FormatStrFormatter
        f = plt.figure(figsize=(16,4))
        plt.rc('xtick',labelsize=10)
        plt.ylim(0.,np.max(y)*1.4)
    else:
        plt.rc('xtick',labelsize=15)
        plt.ylim(0.,1.1)
    width=1.
    plt.xlim(x[0]-width/2.,x[-1]+width/2.)
    plt.xticks(x, label_list)
    plt.axhline(1.,ls='--',color='black')
    plt.bar(x,y,width=1,label='Bloch state (n=%d,k=%d)'%(nk[1],nk[0]))
    plt.legend()
    if save: #Plot name needs to be more indicative of the plot type
        plt_nm = file_nm + '_histogram_n%d_k%d_%s'%(nk[1],nk[0],plt_type)
        plt.savefig('%s.pdf'%plt_nm,format='pdf')
    plt.show()

def store_data(x,y,comments,plt_type,nk,file_nm):
    """
    Store values in .dat file for later plotting
    """
    y_str = [ '{:10.5}'.format(i_y) for i_y in y ]
    x_str = [ '{0:d}'.format(i_x) for i_x in x ]
    comments = np.array([ '#%s'%comment.replace('\n', ' ')  for comment in comments ])
        
    out_nm = file_nm + '_n%s_k%s_%s.dat'%(nk[1],nk[0],plt_type)
    np.savetxt(out_nm, np.c_[x_str,y_str,comments], fmt='%s')
    print("--- output: %s ---"%out_nm)

def print_str(file_nm,Nk,Nb,Ns,Na,els,orbs):
    """
    Print general info on system
    """
    species_string =''
    for el in els: species_string += '  %s'%el
    orb_string =''
    for orb in orbs: orb_string += '  %s'%orb
    print("====== Reading output of projwfc.x ======")
    print("--- File name: %s ---"%file_nm)
    print("   ")
    print("- Number of kpoints: %d"%Nk)
    print("- Number of bands: %d"%Nb)
    print("- Number of atoms: %d"%Na)
    print("     with species present:%s"%species_string)
    print("- Number of states: %d"%Ns)
    print("     Angular momenta present:%s"%orb_string)
    print("   ")
    print("=========================================")

if __name__ == "__main__":

    #####################
    # Input parameters  #
    #####################
    file_nm='data/projwfc.out' #Can be relative or absolute path to file
    N_atom    = None #Get info about specific atom [Integer]
                   #21,2
    plot_nk   = [0,21] #Plot PDOS histogram relative to Bloch state kn [List of two integers]
                        # [0,21],[0,8]
    plot_type = 'orbitals' #Data type in histogram [String] 
                         #(i)   elements, 
                         #(ii)  orbitals, 
                         #(iv)  atoms
    plot_save = False #Set to True to save .pdf plot
    out_file  = False #Set to True to save .dat file with data

    ########################
    # End input parameters #
    ########################

    # Read all lines of output file
    fil = open(file_nm, 'r')
    lines = fil.readlines()

    # Store line numbers where (i) kpts, (ii) energies, (iii) psi^2, (iv) states appear
    k_lines, e_lines, psi_lines, wfc_lines = get_line_numbers(lines)

    # Number of kpoints
    N_kpoints = len(k_lines)
    # Number of bands
    N_bands = len(psi_lines) // len(k_lines)
    # Number of atomic orbitals
    N_states = len(wfc_lines)

    # Build python dictionaries (see function docstring)
    state_dicts = get_state_info(lines,wfc_lines)  # Slight change in this function! Can you guess why?

    # Number of atoms
    N_atoms = state_dicts[N_states-1]['atomnum']+1

    # List of elements present
    elements_list = get_elements(state_dicts)

    # List of orbitals present
    orbitals_list, orbitals_labels = get_orbitals(state_dicts)

    # List of species-orbitals present
    sp_orb_list = get_species_orbitals(elements_list, state_dicts)

    # Read pdos as linear combination of atomic orbitals for each (nk) band state
    orbital_weights, orbital_indices = get_linear_combinations(k_lines,e_lines,psi_lines,N_bands)
    
    # Print general info
    print_str(file_nm,N_kpoints,N_bands,N_states,N_atoms,elements_list,orbitals_labels)

    # Get info relative to a specific atomic number
    if N_atom is not None:
        
        assert isinstance(N_atom, int) # Check that this is correctly an integer
        assert N_atom>0               # Check that this is an allowed value
        
        n_atom = N_atom -1
        for i,state_dict in enumerate(state_dicts):
            if state_dict['atomnum']==n_atom: 
                print("State: %d"%i)
                print(state_dict)

    # Plot histogram relative to specific (nk) Bloch state
    if plot_nk is not None:
        
        i_k,i_n = plot_nk
        assert isinstance(i_k, int)
        assert isinstance(i_n, int)
        assert i_k>=0
        assert i_n>=0
        assert plot_type is not None
        
        # Histogram relative to atomic elements
        if plot_type=='elements':
            x,y = get_plot_elements(orbital_indices[i_k][i_n], orbital_weights[i_k][i_n], state_dicts, elements_list)
            comment_list = elements_list
        # Histogram relative to atomic indices
        if plot_type=='atoms':
            x,y,at_list = get_plot_atoms(orbital_indices[i_k][i_n], orbital_weights[i_k][i_n], state_dicts, N_atoms,elements_list)
            comment_list = at_list
        # Histogram relative to orbital types
        if plot_type=='orbitals':
            x,y = get_plot_orbitals(orbital_indices[i_k][i_n], orbital_weights[i_k][i_n], state_dicts, orbitals_list)
            comment_list = orbitals_labels

        # Plot
        plot_histogram(x,y,plot_type,plot_nk,comment_list,file_nm,save=plot_save)

        # Print output data
        if out_file: store_data(x,y,comment_list,plot_type,plot_nk,file_nm)
