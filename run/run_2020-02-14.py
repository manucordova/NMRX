##
#Last modified 13.02.2020
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
choice = 3
space_groups = [14,19,2,4,61,115,33,9,29,5]
nr_molecules = [4,4,2,2,8,4,4,2,4,2]
space_group_sym = ["M","O","Triclinic","M","O","Tetragonal","O","M","O","M"]
sg = space_groups[choice]

# def select_space_group(choice) -> return sg, sym, n_mol
def which(pgm):
    path=os.getenv('PATH')
    for p in path.split(os.path.pathsep):
        p=os.path.join(p,pgm)
        if os.path.exists(p) and os.access(p,os.X_OK):
            return p

dftb_path = which("dftb+")

# Primary parameters to change
molecule = "cocaine"
nloop = 0
structures = 999
simplex = False
use_energy = True
C13 = False #Not yet implemented for azd #think about including for bigger molecules like ritonavir
H1 = True
H_cutoff = 0.1
#rotation_cycles = 10
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
parameter_set = ['a','b','c', 'beta', 'trans','rot', 'conf']
#parameter_set = ['conf']
directory = os.path.abspath('../data/test_cocaine/') + "/"
vol_high = 3.0


# Secondary parameters to change
low_angle = 45
high_angle = 135
low_trans = 0 #this needs to be changed later depend on the selected cell lenghts and angles
high_trans = 7.5*nr_molecules[choice] #this needs to be changed later depend on the selected cell lenghts and angles
rotate_high = 360
rotate_low = 0
rot_amplitude = 0.1
angle_conf = 10 #np.linspace(180, 1, nloop)
#step_list = [2.0, 2.0, 2.0, 20.0, 20.0, 20.0, 2.0]
step_list = [4.0, 4.0, 4.0, 20.0, 20.0, 20.0, 2.0]
naccept = 0
use_RT = True
RT_start = 0.1
RT_end = 0.01
cut = 1.6

#Slopes and offsets for conversion from shielding to shift
slope = {}
offset = {}
slope["H"] = 1.
offset["H"] = 30.36

#energy_constant = 1233197.53153
energy_constant = 0

initial_structure_original = os.path.abspath("../input_structures/{}/{}.pdb".format(molecule, molecule))

if molecule == "cocaine":
    initial_structure_original = '../input_structures/cocaine/COCAIN10_H_relaxed_out_cell_vc-relax_DFTBplus.pdb'
    atoms = 43
    #energy_constant = 276900
if molecule == "azd":
    initial_structure_original = '../input_structures/azd8329/azd8329_csp.vc-relax.pdb'
    atoms = 62
    #energy_constant = 376543
if molecule == "ritonavir": #space group 19
    initial_structure_original = '../input_structures/ritonavir/Ritonavir_polymorph_2_DFTB_vc-relax.pdb'
    #initial_structure_original = root + '/balodis/work/ritonavir/structure_files/Ritonavir_polymorph_2_DFTB_vc-relax.pdb'
    atoms = 98

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
    chi = ml.rmsd(y_exp,y_pred, offset["H"], slope["H"])
    return chi

# def calculate_C13(trial_crystal, krr_13C, representation_13C, trainsoaps_13C, model_numbers_13C, zeta_13C, molecule):
#     y_pred = ml.predict_shifts([trial_crystal], krr_13C, representation_13C, trainsoaps_13C, model_numbers_13C, zeta_13C, sp=6)
#     y_exp = ml.exp_rmsd_13C()
#     chi_13C = ml.rmsd(y_pred[:17],y_exp)
#     return chi_13C


#Read in the structure with the right conformation and coordinates
starting = read(initial_structure)

#Get the molecular volume
nr = starting.get_atomic_numbers()[:atoms]
rad = np.zeros(len(nr))
for l1 in range(0, len(rad)):
    rad[l1] = 3.0 / 4.0 * np.pi * vdw_radii[nr[l1]] ** 3
Vmol = sum(rad)*nr_molecules[choice]

# Get the lattice parameters and sites from the reference structure
lat = starting.get_cell_lengths_and_angles()
sites = starting.get_chemical_symbols()
# Creation of a big unit cell to extract intraatomic distances that are later used to confirm that no other short distances have been created
big_crystal = cr.create_crystal(starting, molecule, sg, atoms, sites, [400., 400., 400., 90., 90., 90.], [100, 100, 100],
                             [1, 0, 0, 0], [0, 0, 0, 0, 0, 0], nr_molecules[choice])
close_atoms = ase.geometry.get_duplicate_atoms(big_crystal, cutoff=cut, delete=False)

##
tot_trials = 0
n_success = 0

for k in range(structures):
    trial_crystal, lat, trans, q, starting_angles, n_failed = cr.generate_crystal(initial_structure, parameter_set, high_angle, low_angle, high_trans, low_trans, rotate_high, rotate_low, sg, atoms, nr_molecules[choice], molecule, cut, close_atoms, vol_high)
    tot_trials += n_failed + 1
    n_success += 1

    write(directory + name + '/' + experiment + "_" + str(k) + '_init_structure.cif', trial_crystal)
    print("Reasonable structure found after {} tries!".format(n_failed))

    #print starting_angles

    # 1H RMSE for the initial randomly generated crystal
    if H1:
        chi = calculate_1H(trial_crystal,krr,representation,trainsoaps,model_numbers,zeta,molecule)
        print chi
    else:
        chi = 0
    # 13C RMSE for the initial randomly generated crystal
    if C13:
        chi_13C = calculate_C13(trial_crystal, krr_13C, representation_13C, trainsoaps_13C, model_numbers_13C, zeta_13C, molecule)
    else:
        chi_13C = 0
    # Energy calculation for the initial randomly generated crystal
    if use_energy:
        energy = dftb.dftbplus_energy(directory+name+'/', directory + name + '/' + experiment+ "_" + str(k) + '_init_structure.cif', dftb_path, dispersion="D3")* 2625.50 + energy_constant
        print energy
    else:
        energy = 0

    #Saving the starting parameters
    chi_old = chi
    chi_13C_old = chi_13C
    # chi_list = [chi_old]
    # accepted_chi_list = [chi_old]
    # chi_13C_list = [chi_13C_old]
    # accepted_chi_13C_list = [chi_13C_old]

    energy_old = energy
    # energy_list = [energy_old]
    # accepted_energy_list = [energy_old]

    # accepted_conf_change = [conf_angles]
    #
    # cell_parameters = [lat]
    # cell_rotation = [rotation]
    # accepted_step_list = []

    trial_conf = [0,0,0,0,0,0]
    trial_conf_old = [0,0,0,0,0,0]
##
    sel_parameters = [0, 0, 0, 0, 0]
    acc_parameters = [0, 0, 0, 0, 0]
    for i in range(nloop):


        log_file = open(directory + name + '/' + experiment + '.log', "a")
        log_file.write("Structure: "+str(k) + ", loop: " + str(i)+ "\n")
        log_file.close()

        print i
        RT = RT_list[i]
        parameter = random.choice(parameter_set)
        rotation_count = 0

        print parameter
        if parameter in ["a", "b", "c"]:
            sel_parameters[0] += 1
        elif parameter in ["alpha", "beta", "gamma"]:
            sel_parameters[1] += 1
        elif parameter == "trans":
            sel_parameters[2] += 1
        elif parameter == "rot":
            sel_parameters[3] += 1
        elif parameter == "conf":
            sel_parameters[4] += 1
        else:
            raise ValueError("Unknown parameter: {}".format(parameter))

        while True:
            while True:
                try:
                    starting_copy=read(initial_structure)

                    trial_lat = copy.deepcopy(lat)
                    trial_trans = copy.deepcopy(trans)
                    #trial_rotation = copy.deepcopy(rotation)
                    trial_q = copy.deepcopy(q)
                    trial_conf = copy.deepcopy(trial_conf_old)

                    if parameter == 'a':
                        trial_lat[0] += step_list[0] * (2 * mc.gauss_number() - 1)
                    if parameter == 'b':
                        trial_lat[1] += step_list[1] * (2 * mc.gauss_number() - 1)
                    if parameter == 'c':
                        trial_lat[2] += step_list[2] * (2 * mc.gauss_number() - 1)
                    if parameter == 'alpha':
                        trial_lat[3] += step_list[3] * (2 * mc.gauss_number() - 1)
                    if parameter == 'beta':
                        trial_lat[4] += step_list[4] * (2 * mc.gauss_number() - 1)
                    if parameter == 'gamma':
                        trial_lat[5] += step_list[5] * (2 * mc.gauss_number() - 1)

                    if parameter == 'trans':
                        trial_trans[0] += step_list[6] * (2 * mc.gauss_number() - 1)
                        trial_trans[1] += step_list[6] * (2 * mc.gauss_number() - 1)
                        trial_trans[2] += step_list[6] * (2 * mc.gauss_number() - 1)

                    if parameter == 'rot':

                        #rotation_count += 1
                        #if rotation_count > rotation_cycles:
                        #    parameter = random.choice(parameter_set)
                        #    rotation_count = 0
                        #    print "rotation exception"

                        #trial_rotation[0] = random.random() * 360
                        #trial_rotation[1] = random.random() * 360
                        #trial_rotation[2] = random.random() * 360
                        
                        for i in range(4):
                            trial_q[i] += rot_amplitude * (random.random()-0.5)
                        trial_q /= np.linalg.norm(trial_q)

                    if parameter == 'conf':
                        #print "test conf"
                        new_angles = mc.conf_angles(angle_conf,'uniform')
                        trial_conf[0] += new_angles[0]
                        trial_conf[1] += new_angles[1]
                        trial_conf[2] += new_angles[2]
                        trial_conf[3] += new_angles[3]
                        trial_conf[4] += new_angles[4]
                        trial_conf[5] += new_angles[5]


                    vector1 = np.array(trial_conf)
                    vector2 = np.array(starting_angles)
                    trial_angles = vector1 + vector2
                    #trial_structure = cr.create_crystal(starting_copy, molecule, sg, atoms, sites, trial_lat, trial_trans, trial_rotation, trial_angles, nr_molecules[choice])
                    trial_structure = cr.create_crystal(starting_copy, molecule, sg, atoms, sites, trial_lat, trial_trans, trial_q, trial_angles, nr_molecules[choice])

                    break
                except Exception as e:
                    print(e)
                    pass
            
            n_overlap = 0
            if cr.check_for_overlap(trial_structure, cut, close_atoms, Vmol, vol_high):
                break

        write(directory + name + '/' + experiment + "_" + str(k) + '_trial_structure.cif', trial_structure)

        if H1:
            chi = calculate_1H(trial_structure, krr, representation, trainsoaps, model_numbers, zeta, molecule)
        else:
            chi = 0

        if C13:
            chi_13C = calculate_C13(trial_structure, krr_13C, representation_13C, trainsoaps_13C, model_numbers_13C,
                                    zeta_13C, molecule)
        else:
            chi_13C = 0

        if use_energy:
            energy = dftb.dftbplus_energy(directory+name+'/', directory + name + '/' + experiment + "_" + str(k) + '_trial_structure.cif', dftb_path, dispersion="D3")*2625.50 + energy_constant
        else:
            energy = 0

        # chi_list.append(chi)
        # chi_13C_list.append(chi_13C)
        # energy_list.append(energy)

        #print trial_trans, trial_rotation


        if chi + chi_13C / 10.0 + factor*energy <= chi_old + chi_13C_old / 10.0 +factor*energy_old:
            print chi, energy

            chi_old = chi
            chi_13C_old = chi_13C
            energy_old = energy
            lat = copy.deepcopy(trial_lat)
            trans = copy.deepcopy(trial_trans)
            #rotation = copy.deepcopy(trial_rotation)
            q = copy.deepcopy(trial_q)
            trial_conf_old = copy.deepcopy(trial_conf)
            
            if parameter in ["a", "b", "c"]:
                acc_parameters[0] += 1
            elif parameter in ["alpha", "beta", "gamma"]:
                acc_parameters[1] += 1
            elif parameter == "trans":
                acc_parameters[2] += 1
            elif parameter == "rot":
                acc_parameters[3] += 1
            elif parameter == "conf":
                acc_parameters[4] += 1


        elif random.random() <= math.exp((chi_old + chi_13C / 10.0 + factor*energy_old - chi - chi_13C / 10.0 - factor*energy) / RT):
            print chi, energy
            #print trial_lat, trial_trans, trial_rotation

            chi_old = chi
            chi_13C_old = chi_13C
            energy_old = energy
            lat = copy.deepcopy(trial_lat)
            trans = copy.deepcopy(trial_trans)
            #rotation = copy.deepcopy(trial_rotation)
            q = copy.deepcopy(trial_q)
            trial_conf_old = copy.deepcopy(trial_conf)
            
            if parameter in ["a", "b", "c"]:
                acc_parameters[0] += 1
            elif parameter in ["alpha", "beta", "gamma"]:
                acc_parameters[1] += 1
            elif parameter == "trans":
                acc_parameters[2] += 1
            elif parameter == "rot":
                acc_parameters[3] += 1
            elif parameter == "conf":
                acc_parameters[4] += 1
                
        #cell_parameters.append(lat)

    # #Final parameters for the MC run
    # accepted_chi_list = np.array(accepted_chi_list)
    # accepted_energy_list = np.array(accepted_energy_list)
    # np.save(directory + name + '/' + str(k) + '_accepted_chi_list.npy', accepted_chi_list)
    # np.save(directory + name + '/' + str(k) + '_accepted_energy_list.npy', accepted_energy_list)
    final_parameters = [chi_old,energy_old]
    final_parameters = np.array(final_parameters)
    np.save(directory + name + '/' + experiment + "_" + str(k) + '_final_parameters.npy', final_parameters)
    
    dat = np.array([sel_parameters, acc_parameters])
    dat = dat.T
    np.savetxt('data_{}_{}.dat'.format(k, RT_start), dat, delimiter=',', fmt='%d')



    starting_copy=read(initial_structure)
    vector1 = np.array(trial_conf_old)
    vector2 = np.array(starting_angles)
    final_angles = vector1+vector2
    #final_structure = cr.create_crystal(starting_copy, molecule, sg, atoms, sites, lat, trans, rotation, final_angles, nr_molecules[choice])
    final_structure = cr.create_crystal(starting_copy, molecule, sg, atoms, sites, lat, trans, q, final_angles, nr_molecules[choice])
    write(directory + name + '/' + experiment + "_" + str(k) + '_structure.cif', final_structure)
    #os.remove(directory + name + '/' + experiment + "_" + str(k) + '_trial_structure.cif')


    if simplex:

        guess = []
        constants = []
        guess_names = []
        constants_names = []

        if 'a' in parameter_set:
            guess.append(lat[0])
            guess_names.append("a")
        else:
            constants.append(lat[0])
            constants_names.append("a")
        if 'b' in parameter_set:
            guess.append(lat[1])
            guess_names.append("b")
        else:
            constants.append(lat[1])
            constants_names.append("b")
        if 'c' in parameter_set:
            guess.append(lat[2])
            guess_names.append("c")
        else:
            constants.append(lat[2])
            constants_names.append("c")
        if 'alpha' in parameter_set:
            guess.append(lat[3])
            guess_names.append("alpha")
        else:
            constants.append(lat[3])
            constants_names.append("alpha")
        if 'beta' in parameter_set:
            guess.append(lat[4])
            guess_names.append("beta")
        else:
            constants.append(lat[4])
            constants_names.append("beta")
        if 'gamma' in parameter_set:
            guess.append(lat[5])
            guess_names.append("gamma")
        else:
            constants.append(lat[5])
            constants_names.append("gamma")
        if 'trans' in parameter_set:
            guess.append(trans[0])
            guess.append(trans[1])
            guess.append(trans[2])
            guess_names.append("trans_x")
            guess_names.append("trans_y")
            guess_names.append("trans_z")
        else:
            constants.append(trans[0])
            constants.append(trans[1])
            constants.append(trans[2])
            constants_names.append("trans_x")
            constants_names.append("trans_y")
            constants_names.append("trans_z")
        if 'rot' in parameter_set:
            #guess.append(rotation[0])
            #guess.append(rotation[1])
            #guess.append(rotation[2])
            #guess_names.append("rot_x")
            #guess_names.append("rot_y")
            #guess_names.append("rot_z")
            guess.append(q[0])
            guess.append(q[1])
            guess.append(q[2])
            guess.append(q[3])
            guess_names.append("q_0")
            guess_names.append("q_1")
            guess_names.append("q_2")
            guess_names.append("q_3")
        else:
            #constants.append(rotation[0])
            #constants.append(rotation[1])
            #constants.append(rotation[2])
            #constants_names.append("rot_x")
            #constants_names.append("rot_y")
            #constants_names.append("rot_z")
            constants.append(q[0])
            constants.append(q[1])
            constants.append(q[2])
            constants.append(q[3])
            constants_names.append("q_0")
            constants_names.append("q_1")
            constants_names.append("q_2")
            constants_names.append("q_3")
        if 'conf' in parameter_set:
            guess.append(final_angles[0])
            guess.append(final_angles[1])
            guess.append(final_angles[2])
            guess.append(final_angles[3])
            guess.append(final_angles[4])
            guess.append(final_angles[5])
            guess_names.append("conf")
        else:
            constants.append(final_angles[0])
            constants.append(final_angles[1])
            constants.append(final_angles[2])
            constants.append(final_angles[3])
            constants.append(final_angles[4])
            constants.append(final_angles[5])
            constants_names.append("conf")



        all_param_names = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y',
                           'rot_z', 'conf']


        def optimisation_function(guess, constants, guess_names, constants_names, all_param_names, initial_structure, sg, atoms, sites, H1, use_energy, factor, krr, representation, trainsoaps, model_numbers,
                                  zeta,molecule,directory,name,experiment,k,energy_constant):


            #Make a list of the parameters
            all_params = []
            ind_guess = 0
            ind_const = 0
            for param in all_param_names:
                if param in guess_names:
                    if param == "conf":
                        if param in guess_names[:-1]:
                            raise ValueError("conf parameter should be in last position.")
                        all_params.extend(guess[ind_guess:])
                    else:
                        all_params.append(guess[ind_guess])
                        ind_guess += 1
                elif param in constants_names:
                    if param == "conf":
                        if param in constants_names[:-1]:
                            raise ValueError("conf parameter should be in last position.")
                        all_params.extend(constants[ind_const:])
                    else:
                        all_params.append(constants[ind_const])
                        ind_const += 1

            #print all_params

            #Load a structure and apply the parameters
            starting_copy = read(initial_structure)
            lat = starting_copy.get_cell_lengths_and_angles()
            lat[0] = all_params[0]
            lat[1] = all_params[1]
            lat[2] = all_params[2]
            lat[3] = all_params[3]
            lat[4] = all_params[4]
            lat[5] = all_params[5]
            trans = [all_params[6], all_params[7], all_params[8]]
            #rotation = [all_params[9], all_params[10], all_params[11]]
            #conf_angles = [all_params[12], all_params[13], all_params[14], all_params[15], all_params[16],all_params[17]]
            #trial_structure = cr.create_crystal(starting_copy, molecule, sg, atoms, sites, lat, trans, rotation, conf_angles, nr_molecules[choice])
            q = [all_params[9], all_params[10], all_params[11], all_params[12]]
            conf_angles = [all_params[13], all_params[14], all_params[15], all_params[16],all_params[17], all_params[18]]
            trial_structure = cr.create_crystal(starting_copy, molecule, sg, atoms, sites, lat, trans, q, conf_angles, nr_molecules[choice])
            write(directory + name + '/' + experiment + "_" + str(k) + '_opt_structure.cif', trial_structure)

            # Compute shifts and H_rmse
            if H1:
                chi = calculate_1H(trial_structure, krr, representation, trainsoaps, model_numbers, zeta, molecule)
            else:
                chi = 0


            if use_energy:
                energy = dftb.dftbplus_energy(directory + name + '/',
                                              directory + name + '/' + experiment + "_" + str(k) + '_opt_structure.cif', dftb_path, dispersion="D3") * 2625.50 + energy_constant
            else:
                energy = 0

            # chi_save = np.array(chi)
            # energy_save = np.array(energy)
            # np.save(directory + name + '/' + experiment + "_" + str(k) + '_opt_chi.npy', chi_save)
            # np.save(directory + name + '/' + experiment + "_" + str(k) + '_opt_energy.npy', energy_save)

            final_parameters = [chi, energy]
            final_parameters = np.array(final_parameters)
            np.save(directory + name + '/' + experiment + "_" + str(k) + '_final_opt_parameters.npy', final_parameters)

            print energy, chi

            return chi + factor*energy



        res = op.minimize(optimisation_function, guess, args=(constants, guess_names, constants_names, all_param_names, initial_structure, sg, atoms, sites,H1, use_energy, factor, krr, representation, trainsoaps, model_numbers,
                                  zeta,molecule,directory,name,experiment,k,energy_constant),method='Nelder-Mead',tol=1e-2,options={'fatol': 1e-2})

        print res

        ##


print("Structure generation lead to {:.2f}% of successful initial structures ({}/{})".format(float(n_success)/float(tot_trials)*100., n_success, tot_trials))





