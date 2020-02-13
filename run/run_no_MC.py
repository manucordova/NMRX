##
#Last modified 12.02.2020
#Inclusion of Ritonavir, first iteration
#All possible changes are included. First the reference structure is always loaded and then randomized.
#Set up for cocaine and azd8329
# One can also load a different conformation, by rewriting the conformation in the reference structure. 1H, 13C and
#DFTB can be included or excluded. Temperature gradient can be turned on or off.
#If new 1H chi < H_cutoff, then it is set to old 1H chi
#If new 13H chi < 4.3, then it is set to old 1H chi
#Adding simplex optimisation at the end
#Included rotations for azd


##
#Initialization, numerical analysis and system tools
import numpy as np
import math
import sys,os
import random
import copy
import time
import scipy.optimize as op
from shutil import copyfile

#Atomic simulation environment
import ase
from ase.io import read,write
from ase.visualize import view
from ase.build import minimize_rotation_and_translation
from ase.spacegroup import crystal
from ase.data import chemical_symbols
from ase.data import atomic_numbers, vdw_radii


#Paths used for different system modules
sys.path.insert(0, "../src")
import mc_functions as mc
import ml_functions as ml
import generate_crystals as cr
import dftb as dftb

##
random.seed()

#The choice of the space group
choice = 1
space_groups = [14,19,2,4,61,115,33,9,29,5]
nr_molecules = [4,4,2,2,8,4,4,2,4,2]
space_group_sym = ["M","O","Triclinic","M","O","Tetragonal","O","M","O","M"]
sg = space_groups[choice]

# def select_space_group(choice) -> return sg, sym, n_mol


# Primary parameters to change
molecule = "cocaine"
nloop = 0
structures = 500
simplex = True
use_energy = True
C13 = False #Not yet implemented for azd #think about including for bigger molecules like ritonavir
H1 = True
H_cutoff= 0.1
rotation_cycles = 10
factor = 0.005
experiment = "1"
comment = '_test'
#parameter_set = ['']
#parameter_set = ['rot']
#parameter_set = ['c']
#parameter_set = ['a','b','c','beta']
#parameter_set = ['a','b','c']
#parameter_set = ['a','b','c','alpha','beta','gamma']
#parameter_set = ['a','b','c','beta','trans','rot','conf']
#parameter_set = ['trans','rot']
parameter_set = ['a','b','c','trans','rot']
#parameter_set = ['conf']
directory = '../data/test_cocaine_2/'
vol_high = 3.0


# Secondary parameters to change
low_len = 2.0*nr_molecules[choice] #this needs to be changed later depending on the projections of the molecule
high_len = 7.5*nr_molecules[choice] #this needs to be changed later depending on the projections of the molecule
low_angle = 45
high_angle = 135
low_trans = 0 #this needs to be changed later depend on the selected cell lenghts and angles
high_trans = 7.5*nr_molecules[choice] #this needs to be changed later depend on the selected cell lenghts and angles
rotate_high = 360
rotate_low = 0
angle_conf = 180 #np.linspace(180, 1, nloop)
#step_list = [2.0, 2.0, 2.0, 20.0, 20.0, 20.0, 2.0]
step_list = [4.0, 4.0, 4.0, 20.0, 20.0, 20.0, 2.0]
naccept = 0
use_RT = True
RT_start = 0.1
RT_end = 0.01
cut = 1.6


energy_constant = 1233197.53153

initial_structure_original = "../data/input_structures/{}/{}.pdb".format(molecule, molecule)

#if molecule == "cocaine":
#    initial_structure_original = root+'/balodis/work/cocaine/structure_files/COCAIN10_H_relaxed_out_cell_vc-relax_DFTBplus.pdb'
#    atoms = 43
    #energy_constant = 276900
#if molecule == "azd":
#    initial_structure_original = root+'/balodis/work/azd8329/structure_files/azd8329_csp.vc-relax.pdb'
#    atoms = 62
    #energy_constant = 376543
#if molecule == "ritonavir": #space group 19
#    initial_structure_original = root+'/balodis/work/ritonavir/structure_files/Ritonavir_reordered/Ritonavir_polymorph_2_DFTB_vc-relax.pdb'
    #initial_structure_original = root + '/balodis/work/ritonavir/structure_files/Ritonavir_polymorph_2_DFTB_vc-relax.pdb'
#    atoms = 98

atoms = 43 # We should get it from gas phase structure

if use_RT:
    RT_list = np.linspace(RT_start,RT_end,nloop) #Should depend on nloops or accepted structures?
else:
    RT_list = np.linspace(0.001, 0.001, nloop)

try:
    os.mkdir(directory)
except:
    pass

if use_energy:
    name = str(nloop) + "_loops_" + str(factor)+ '_factor_' + "_H1_" + str(H1) + \
       "_C13_" + str(C13) +str(comment)+"_"+experiment
else:
    name = str(nloop) + "_loops_" + "_H1_" + str(H1) + \
       "_C13_" + str(C13) +str(comment)+"_"+experiment

try:
    os.mkdir(directory + name)
except:
    pass

#Loading of the kernel
krr, representation, trainsoaps, model_numbers, zeta = ml.load_kernel(1)
if C13:
    krr_13C, representation_13C, trainsoaps_13C, model_numbers_13C, zeta_13C = ml.load_kernel(6)

##

#Copy the initial structure
number = random.random()
copyfile(initial_structure_original, directory + name + '/'+str(number)+'.pdb')
initial_structure = directory + name + '/'+str(number)+'.pdb'




def calculate_1H(trial_crystal,krr,representation,trainsoaps,model_numbers,zeta,molecule):
    y_pred = ml.predict_shifts([trial_crystal], krr, representation, trainsoaps, model_numbers, zeta, sp=1)
    y_pred, y_exp = ml.exp_rmsd(y_pred,molecule=molecule)
    chi = ml.rmsd(y_exp,y_pred)
    return chi

# def calculate_C13(trial_crystal, krr_13C, representation_13C, trainsoaps_13C, model_numbers_13C, zeta_13C, molecule):
#     y_pred = ml.predict_shifts([trial_crystal], krr_13C, representation_13C, trainsoaps_13C, model_numbers_13C, zeta_13C, sp=6)
#     y_exp = ml.exp_rmsd_13C()
#     chi_13C = ml.rmsd(y_pred[:17],y_exp)
#     return chi_13C


#Read in the structure with the right conformation and coordinates
starting = read(initial_structure)
# Get the lattice parameters and sites from the reference structure
lat = starting.get_cell_lengths_and_angles()
sites = starting.get_chemical_symbols()
# Creation of a big unit cell to extract intraatomic distances that are later used to confirm that no other short distances have been created
big_crystal = cr.create_crystal(starting, molecule, sg, atoms, sites, [400., 400., 400., 90., 90., 90.], [100, 100, 100],
                             [0, 0, 0], [0, 0, 0, 0, 0, 0], nr_molecules[choice])
close_atoms = ase.geometry.get_duplicate_atoms(big_crystal, cutoff=cut, delete=False)

##
n_success = 0
tot_trials = 0
for k in range(structures):
    trial_crystal, n_failed = cr.generate_crystal(initial_structure, parameter_set, high_len, low_len, high_angle, low_angle, high_trans, low_trans, rotate_high, rotate_low, sg, atoms, nr_molecules[choice], molecule, cut, close_atoms, vol_high)
    tot_trials += n_failed + 1
    n_success += 1

    write(directory + name + '/' + experiment + "_" + str(k) + '_init_structure.cif', trial_crystal)
    print("\t\t\t\t{:.2f}".format(trial_crystal.get_volume()))

print("Crystal generation done! success rate: {:.2f}%, {} trials, {} trial structures generated".format(float(n_success)/float(tot_trials)*100., tot_trials, n_success))

## END OF SCRIPT







