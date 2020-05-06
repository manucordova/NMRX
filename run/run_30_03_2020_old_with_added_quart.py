##
#Last modified 30.03.2020
#Changed all starting copies to make sure nothing is overwriten

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
import mc_functions_v2 as mc
import ml_functions_v2 as ml
import dftb_v2 as dftb
import generate_crystals as cr


np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

##
random.seed()

#The choice of the space group
choice = 3
space_groups = [14,19,2,4,61,115,33,9,29,5]
nr_molecules = [4,4,2,2,8,4,4,2,4,2]
space_group_sym = ["M","O","Triclinic","M","O","Tetragonal","O","M","O","M"]
sg = space_groups[choice]

n_mol = nr_molecules[choice]


#Finding dftb+ path on the local machine
def which(pgm):
    path = os.getenv('PATH')
    for p in path.split(os.path.pathsep):
        p = os.path.join(p, pgm)
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p

dftb_path = which("dftb+")
cluster = False

# Primary parameters to change
molecule = "cocaine"
nloop = 3
structures = 2

simplex = False
use_energy = True
C13 = False #Not yet implemented for azd #think about including for bigger molecules like ritonavir
H1 = True
H_cutoff= 0.1
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
parameter_set = ['a','b','c','beta','trans','rot']
#parameter_set = ['conf']
directory = os.path.abspath('../data/test/') + "/"
vol_high = 3.0


# Secondary parameters to change
low_len = 2.0*nr_molecules[choice] #this needs to be changed later depending on the projections of the molecule
high_len = 7.5*nr_molecules[choice] #this needs to be changed later depending on the projections of the molecule
low_angle = 45
high_angle = 135
low_trans = 0 #this needs to be changed later depend on the selected cell lenghts and angles
high_trans = 7.5*nr_molecules[choice] #this needs to be changed later depend on the selected cell lenghts and angles
angle_conf = 180 #np.linspace(180, 1, nloop)
step_list = [2.0, 2.0, 2.0, 20.0, 20.0, 20.0, 2.0, 1.0]
#step_list = [4.0, 4.0, 4.0, 20.0, 20.0, 20.0, 2.0]
naccept = 0
use_RT = True
RT_start = 0.1
RT_end = 0.01
cut = 1.6


energy_constant = 0 #1233197.53153

initial_structure_original = os.path.abspath("../input_structures/{}/{}.pdb".format(molecule, molecule))

if molecule == "cocaine":
    initial_structure_original = '../input_structures/cocaine/COCAIN10_H_relaxed_out_cell_vc-relax_DFTBplus.pdb'
    atoms = 43
    # energy_constant = 276900
if molecule == "azd":
    initial_structure_original = '../input_structures/azd8329/azd8329_csp.vc-relax.pdb'
    atoms = 62
    # energy_constant = 376543
if molecule == "ritonavir":  # space group 19
    initial_structure_original = '../input_structures/ritonavir/Ritonavir_polymorph_2_DFTB_vc-relax.pdb'
    atoms = 98


if use_RT:
    RT_list = np.linspace(RT_start,RT_end,nloop)
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
starting_structure = read(initial_structure)


def check_for_overlap(trial_crystal, cut, close_atoms, Vmol, vol_high):
    """
    """
    overlapping_atoms = ase.geometry.get_duplicate_atoms(trial_crystal, cutoff=cut, delete=False)
    count = 0
    for aa in overlapping_atoms:
        for bb in close_atoms:
            if aa[0] == bb[0] and aa[1] == bb[1]:
                count += 1
    condition_1 = (len(overlapping_atoms) == count)

    Vcell = trial_crystal.get_volume()
    condition_2 = (Vcell > Vmol and Vcell < vol_high * Vmol)

    print condition_1, condition_2

    return condition_1 and condition_2

def calculate_1H(trial_crystal,krr,representation,trainsoaps,model_numbers,zeta,molecule):
    y_pred = ml.predict_shifts([trial_crystal], krr, representation, trainsoaps, model_numbers, zeta, sp=1)
    y_pred, y_exp = ml.exp_rmsd(y_pred,molecule=molecule)
    chi = ml.rmsd(y_exp,y_pred)
    return chi


def get_third_length_low(a, b, alpha, beta, gamma, Vmol):
    sqroot = 1+(2*np.cos(np.deg2rad(alpha))*np.cos(np.deg2rad(beta))*np.cos(np.deg2rad(gamma)))
    sqroot += -np.cos(np.deg2rad(alpha))**2-np.cos(np.deg2rad(beta))**2-np.cos(np.deg2rad(gamma))**2
    sqroot = np.sqrt(sqroot)
    return Vmol / (a*b*sqroot)
# def calculate_C13(trial_crystal, krr_13C, representation_13C, trainsoaps_13C, model_numbers_13C, zeta_13C, molecule):
#     y_pred = ml.predict_shifts([trial_crystal], krr_13C, representation_13C, trainsoaps_13C, model_numbers_13C, zeta_13C, sp=6)
#     y_exp = ml.exp_rmsd_13C()
#     chi_13C = ml.rmsd(y_pred[:17],y_exp)
#     return chi_13C

# Read in the structure with the right conformation and coordinates
starting = copy.deepcopy(starting_structure)

# Get the lattice parameters and sites from the reference structure
lat = starting.get_cell_lengths_and_angles()
sites = starting.get_chemical_symbols()
# Creation of a big unit cell to extract intraatomic distances that are later used to confirm that no other short distances have been created
big_crystal = cr.create_crystal(starting, molecule, sg, atoms, sites, [400., 400., 400., 90., 90., 90.],
                                [100, 100, 100],
                                [1, 0, 0, 0], [0, 0, 0, 0, 0, 0], nr_molecules[choice])
close_atoms = ase.geometry.get_duplicate_atoms(big_crystal, cutoff=cut, delete=False)

# Calculate the volume of the molecule
nr = big_crystal.get_atomic_numbers()
rad = np.zeros(len(nr))
for l1 in range(0, len(rad)):
    rad[l1] = 3.0 / 4.0 * np.pi * vdw_radii[nr[l1]] ** 3
Vmol = sum(rad)


accepted_structures = 0
runs = 0
##
for k in range(structures):


    #Create the starting structure and give it random parameters
    while True:
        while True:
            try:
                #Read in the structure with the right conformation and coordinates
                starting = copy.deepcopy(starting_structure)

                #Get the lattice parameters and sites from the reference structure
                lat = starting.get_cell_lengths_and_angles()
                sites = starting.get_chemical_symbols()

                # else:
                #     lat[2] = lat[1]
                if 'alpha' in parameter_set:
                    lat[3] = (high_angle - low_angle) * mc.gauss_number() + low_angle
                else:
                    lat[3] = 90.0 #take care to change this value if you want to explore M or T cells
                if 'beta' in parameter_set:
                    lat[4] = (high_angle - low_angle) * mc.gauss_number() + low_angle
                else:
                    lat[4] = 90.0 #take care to change this value if you want to explore M or T cells
                if 'gamma' in parameter_set:
                    lat[5] = (high_angle - low_angle) * mc.gauss_number() + low_angle
                else:
                    lat[5] = 90.0 #take care to change this value if you want to explore M or T cells


                if 'rot' in parameter_set:
                    quat = np.zeros(4)
                    quat[0] = 0.5-random.random()
                    quat[1] = 0.5-random.random()
                    quat[2] = 0.5-random.random()
                    quat[3] = 0.5-random.random()
                    quat /= np.linalg.norm(quat)
                else:
                    quat = [1,0,0,0]

                # #Randomly choose a, b, c, beta, placement, rotation and dihedrals
                # if 'a'in parameter_set:
                #     lat[0] = (high_len-low_len)*random.random()+low_len
                # if 'b' in parameter_set:
                #     lat[1] = (high_len-low_len)*random.random()+low_len
                # if 'c' in parameter_set:
                #     lat[2] = (high_len - low_len) * random.random() + low_len
                #     # c_low = Vmol/(lat[0]*lat[1])
                #     # c_high = vol_high*Vmol/(lat[0]*lat[1])
                #     # lat[2] = (c_high-c_low)*random.random()+c_low

                cell_l = ["a", "b", "c"]
                low_lens = cr.get_projections(starting, lat[3], lat[4], lat[5], atoms, quat)


                tmp_l = [0., 0.]


                for i, x in enumerate(np.random.permutation(cell_l)):
                    ind_l = cell_l.index(x)


                    if x in parameter_set:
                        if i < 2:
                            #lat[ind_l] = (high_len-low_len)*random.random()+low_len
                            lat[ind_l] = low_lens[ind_l]*((n_mol-1)*random.random()+1)
                            tmp_l[i] = lat[ind_l]
                        else:
                            low_c = get_third_length_low(tmp_l[0], tmp_l[1], lat[3], lat[4], lat[5], Vmol)
                            #sqroot = 1+(2*np.cos(np.deg2rad(lat[3]))*np.cos(np.deg2rad(lat[4]))*np.cos(np.deg2rad(lat[5])))
                            #sqroot += -np.cos(np.deg2rad(lat[3]))**2-np.cos(np.deg2rad(lat[4]))**2-np.cos(np.deg2rad(lat[5]))**2
                            #sqroot = np.sqrt(sqroot)
                            #low_c = Vmol / (tmp_vol*sqroot)
                            lat[ind_l] = (vol_high*low_c-low_c)*random.random()+low_c


                if 'trans' in parameter_set:
                    x = (high_trans - low_trans) * random.random() + low_trans
                    y = (high_trans - low_trans) * random.random() + low_trans
                    z = (high_trans - low_trans) * random.random() + low_trans
                    trans = [x, y, z]
                else:
                    trans = [0, 0, 0]


                if 'conf' in parameter_set:
                    starting_angles = mc.conf_angles(180, 'uniform')
                else:
                    starting_angles = [0, 0, 0, 0, 0, 0]


                #Overwrite the starting structure and create a trial structure for testing
                trial_crystal = cr.create_crystal(starting,molecule,sg,atoms,sites,lat,trans,quat,starting_angles, n_mol)
                write(directory + name + '/' + experiment + "_" + str(k) + '_test_structure.cif', trial_crystal)
                break
            except Exception as e:
                print(e)
                pass



        runs += 1


        # Check if the generated cell is good, if not, try again
        if check_for_overlap(trial_crystal,cut,close_atoms,Vmol,vol_high):
            break

    accepted_structures += 1
    write(directory + name + '/' + experiment + "_" + str(k) + '_init_structure.cif', trial_crystal)



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
        energy = dftb.dftbplus_energy(directory + name + '/', trial_crystal, dftb_path, dispersion="D3") * 2625.50 + energy_constant
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

    print lat
##
    for i in range(nloop):


        log_file = open(directory + name + '/' + experiment + '.log', "a")
        log_file.write("Structure: "+str(k) + ", loop: " + str(i)+ "\n")
        log_file.close()

        print i
        RT = RT_list[i]
        parameter = random.choice(parameter_set)
        rotation_count = 0

        print parameter

        while True:
            while True:
                try:
                    starting_copy=copy.deepcopy(starting_structure)

                    trial_lat = copy.deepcopy(lat)
                    trial_trans = copy.deepcopy(trans)
                    trial_quat = copy.deepcopy(quat)
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
                        for i in range(4):
                            trial_quat[i] += step_list[7] * (random.random() - 0.5)
                        trial_quat /= np.linalg.norm(trial_quat)

                    if parameter == 'conf':
                        #print "test conf"
                        new_angles = mc.conf_angles(angle_conf,'uniform')
                        trial_conf[0] = new_angles[0]
                        trial_conf[1] = new_angles[1]
                        trial_conf[2] = new_angles[2]
                        trial_conf[3] = new_angles[3]
                        trial_conf[4] = new_angles[4]
                        trial_conf[5] = new_angles[5]


                    vector1 = np.array(trial_conf)
                    vector2 = np.array(starting_angles)
                    trial_angles = vector1 + vector2
                    trial_structure = cr.create_crystal(starting_copy, molecule, sg, atoms, sites, trial_lat,
                                                        trial_trans, trial_quat, trial_angles, nr_molecules[choice])


                    break
                except Exception as e:
                    print(e)
                    pass

            if check_for_overlap(trial_structure, cut, close_atoms, Vmol, vol_high):
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
            energy = dftb.dftbplus_energy(directory + name + '/', trial_structure, dftb_path, dispersion="D3") * 2625.50 + energy_constant
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
            trial_conf_old = copy.deepcopy(trial_conf)
            quat = copy.deepcopy(trial_quat)


        elif random.random() <= math.exp((chi_old + chi_13C / 10.0 + factor*energy_old - chi - chi_13C / 10.0 - factor*energy) / RT):
            print chi, energy
            #print trial_lat, trial_trans, trial_rotation

            chi_old = chi
            chi_13C_old = chi_13C
            energy_old = energy
            lat = copy.deepcopy(trial_lat)
            trans = copy.deepcopy(trial_trans)
            trial_conf_old = copy.deepcopy(trial_conf)
            quat = copy.deepcopy(trial_quat)


        #cell_parameters.append(lat)

    # #Final parameters for the MC run
    # accepted_chi_list = np.array(accepted_chi_list)
    # accepted_energy_list = np.array(accepted_energy_list)
    # np.save(directory + name + '/' + str(k) + '_accepted_chi_list.npy', accepted_chi_list)
    # np.save(directory + name + '/' + str(k) + '_accepted_energy_list.npy', accepted_energy_list)
    final_parameters = [chi_old,energy_old]
    final_parameters = np.array(final_parameters)
    np.save(directory + name + '/' + experiment + "_" + str(k) + '_final_parameters.npy', final_parameters)


    if nloop > 0:
        starting_copy=read(initial_structure)
        vector1 = np.array(trial_conf_old)
        vector2 = np.array(starting_angles)
        final_angles = vector1+vector2
        final_structure = cr.create_crystal(starting_copy, molecule, sg, atoms, sites, lat, trans, quat, final_angles,
                                            nr_molecules[choice])
        write(directory + name + '/' + experiment + "_" + str(k) + '_structure.cif', final_structure)
        if nloop > 0:
            os.remove(directory + name + '/' + experiment + "_" + str(k) + '_trial_structure.cif')


    # if simplex:
    #
    #     guess = []
    #     constants = []
    #     guess_names = []
    #     constants_names = []
    #
    #     if 'a' in parameter_set:
    #         guess.append(lat[0])
    #         guess_names.append("a")
    #     else:
    #         constants.append(lat[0])
    #         constants_names.append("a")
    #     if 'b' in parameter_set:
    #         guess.append(lat[1])
    #         guess_names.append("b")
    #     else:
    #         constants.append(lat[1])
    #         constants_names.append("b")
    #     if 'c' in parameter_set:
    #         guess.append(lat[2])
    #         guess_names.append("c")
    #     else:
    #         constants.append(lat[2])
    #         constants_names.append("c")
    #     if 'alpha' in parameter_set:
    #         guess.append(lat[3])
    #         guess_names.append("alpha")
    #     else:
    #         constants.append(lat[3])
    #         constants_names.append("alpha")
    #     if 'beta' in parameter_set:
    #         guess.append(lat[4])
    #         guess_names.append("beta")
    #     else:
    #         constants.append(lat[4])
    #         constants_names.append("beta")
    #     if 'gamma' in parameter_set:
    #         guess.append(lat[5])
    #         guess_names.append("gamma")
    #     else:
    #         constants.append(lat[5])
    #         constants_names.append("gamma")
    #     if 'trans' in parameter_set:
    #         guess.append(trans[0])
    #         guess.append(trans[1])
    #         guess.append(trans[2])
    #         guess_names.append("trans_x")
    #         guess_names.append("trans_y")
    #         guess_names.append("trans_z")
    #     else:
    #         constants.append(trans[0])
    #         constants.append(trans[1])
    #         constants.append(trans[2])
    #         constants_names.append("trans_x")
    #         constants_names.append("trans_y")
    #         constants_names.append("trans_z")
    #     if 'rot' in parameter_set:
    #         guess.append(rotation[0])
    #         guess.append(rotation[1])
    #         guess.append(rotation[2])
    #         guess_names.append("rot_x")
    #         guess_names.append("rot_y")
    #         guess_names.append("rot_z")
    #     else:
    #         constants.append(rotation[0])
    #         constants.append(rotation[1])
    #         constants.append(rotation[2])
    #         constants_names.append("rot_x")
    #         constants_names.append("rot_y")
    #         constants_names.append("rot_z")
    #     if 'conf' in parameter_set:
    #         guess.append(final_angles[0])
    #         guess.append(final_angles[1])
    #         guess.append(final_angles[2])
    #         guess.append(final_angles[3])
    #         guess.append(final_angles[4])
    #         guess.append(final_angles[5])
    #         guess_names.append("conf")
    #     else:
    #         constants.append(final_angles[0])
    #         constants.append(final_angles[1])
    #         constants.append(final_angles[2])
    #         constants.append(final_angles[3])
    #         constants.append(final_angles[4])
    #         constants.append(final_angles[5])
    #         constants_names.append("conf")
    #
    #
    #
    #     all_param_names = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y',
    #                        'rot_z', 'conf']
    #
    #
    #     def optimisation_function(guess, constants, guess_names, constants_names, all_param_names, initial_structure, sg, atoms, sites, H1, use_energy, factor, krr, representation, trainsoaps, model_numbers,
    #                               zeta,molecule,directory,name,experiment,k,energy_constant):
    #
    #
    #         #Make a list of the parameters
    #         all_params = []
    #         ind_guess = 0
    #         ind_const = 0
    #         for param in all_param_names:
    #             if param in guess_names:
    #                 if param == "conf":
    #                     if param in guess_names[:-1]:
    #                         raise ValueError("conf parameter should be in last position.")
    #                     all_params.extend(guess[ind_guess:])
    #                 else:
    #                     all_params.append(guess[ind_guess])
    #                     ind_guess += 1
    #             elif param in constants_names:
    #                 if param == "conf":
    #                     if param in constants_names[:-1]:
    #                         raise ValueError("conf parameter should be in last position.")
    #                     all_params.extend(constants[ind_const:])
    #                 else:
    #                     all_params.append(constants[ind_const])
    #                     ind_const += 1
    #
    #         #print all_params
    #
    #         #Load a structure and apply the parameters
    #         starting_copy = read(initial_structure)
    #         lat = starting_copy.get_cell_lengths_and_angles()
    #         lat[0] = all_params[0]
    #         lat[1] = all_params[1]
    #         lat[2] = all_params[2]
    #         lat[3] = all_params[3]
    #         lat[4] = all_params[4]
    #         lat[5] = all_params[5]
    #         trans = [all_params[6], all_params[7], all_params[8]]
    #         rotation = [all_params[9], all_params[10], all_params[11]]
    #         conf_angles = [all_params[12], all_params[13], all_params[14], all_params[15], all_params[16],all_params[17]]
    #         trial_structure = create_crystal(starting_copy, molecule, sg, atoms, sites, lat, trans, rotation, conf_angles)
    #         write(directory + name + '/' + experiment + "_" + str(k) + '_opt_structure.cif', trial_structure)
    #
    #         # Compute shifts and H_rmse
    #         if H1:
    #             chi = calculate_1H(trial_structure, krr, representation, trainsoaps, model_numbers, zeta, molecule)
    #         else:
    #             chi = 0
    #
    #
    #         if use_energy:
    #             energy = dftb.dftbplus_energy(directory + name + '/', trial_structure, dftb_path, dispersion="D3") * 2625.50 + energy_constant
    #
    #         else:
    #             energy = 0
    #
    #         # chi_save = np.array(chi)
    #         # energy_save = np.array(energy)
    #         # np.save(directory + name + '/' + experiment + "_" + str(k) + '_opt_chi.npy', chi_save)
    #         # np.save(directory + name + '/' + experiment + "_" + str(k) + '_opt_energy.npy', energy_save)
    #
    #         final_parameters = [chi, energy]
    #         final_parameters = np.array(final_parameters)
    #         np.save(directory + name + '/' + experiment + "_" + str(k) + '_final_opt_parameters.npy', final_parameters)
    #
    #         print energy, chi
    #
    #         return chi + factor*energy
    #
    #
    #
    #     res = op.minimize(optimisation_function, guess, args=(constants, guess_names, constants_names, all_param_names, initial_structure, sg, atoms, sites,H1, use_energy, factor, krr, representation, trainsoaps, model_numbers,
    #                               zeta,molecule,directory,name,experiment,k,energy_constant),method='Nelder-Mead',tol=1e-2,options={'fatol': 1e-2})
    #
    #     print res

        ##








