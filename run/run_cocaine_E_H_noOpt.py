#####
# Last modified 29.06.2020
# Re-writing the script so that it is more general
# Authors: Manuel Cordova, Martins Balodis
#####



### Import general libraries
import numpy as np
import os
import sys
import shutil
import copy
import pickle as pk
import time
#import scipy.optimize as op



### Import Atomic Simulation Environment (ASE) libraries
import ase
import ase.io
#import ase.build
#import ase.spacegroup
#import ase.data



### Import custom libraries
# Path to the source files
sys.path.insert(0, "../src")
# Import source files
import new_mc_functions as mc
import new_ml_functions as ml
import new_energy_functions as en
import new_generate_crystals as cr

### TODO: insert full list of spacegroups, just ask for a choice
### Select space group, symmetry and number of molecules in the unit cell
### Space group of cocaine
sg = 4
n_mol = 2
sym = "M"



### Parameters for the chemical system under study
molecule = "cocaine"
experiment = sys.argv[1]
comment = "production_run"
# Directory for results
out_dir = "../data/cocaine_production_09_2020_noOpt/"
### TODO: Change to make general
if molecule == "cocaine":
    # Path to initial file
    initial_structure_original = '../input_structures/cocaine/Cocaine_gas_phase_conformer.xyz'
    # Number of atoms in one molecule
    n_atoms = 43
    # Number of dihedral angles
    n_conf = 5
    # Atoms and masks of the dihedral angles
    conf_params = {}
    conf_params["a1"] = [13, 19, 7, 2, 1]
    conf_params["a2"] = [8, 7, 18, 1, 14]
    conf_params["a3"] = [7, 18, 2, 14, 21]
    conf_params["a4"] = [18, 2, 3, 20, 15]
    conf_params["mask"] = [[9, 10, 11, 12, 13, 32, 33, 34, 35, 36], [8, 9, 10, 11, 12, 13, 19, 32, 33, 34, 35, 36], [7, 8, 9, 10, 11, 12, 13, 19, 32, 33, 34, 35, 36], [15, 20, 21, 37, 38, 39], [15, 37, 38, 39]]
    # energy_constant = 276900
if molecule == "azd":
    initial_structure_original = '../input_structures/azd8329/azd8329_csp.vc-relax.pdb'
    n_atoms = 62
    # energy_constant = 376543
if molecule == "ritonavir":  # space group 19
    initial_structure_original = '../input_structures/ritonavir/Ritonavir_polymorph_2_DFTB_vc-relax.pdb'
    n_atoms = 98
    # Number of dihedral angles
    n_conf = 23
    # Atoms and masks of the dihedral angles
    conf_params = {}
    conf_params["a1"] = [46, 45, 44, 6, 43, 49, 38, 11, 93, 36, 35, 20, 15, 25, 22, 21, 26, 8, 28, 29, 9, 32, 33]
    conf_params["a2"] = [45, 44, 6, 43, 11, 38, 37, 36, 4, 35, 34, 15, 14, 24, 21, 26, 8, 28, 9, 30, 32, 33, 10]
    conf_params["a3"] = [44, 6, 43, 11, 36, 37, 36, 35, 35, 34, 13, 14, 13, 23, 26, 8, 28, 9, 32, 32, 33, 10, 13]
    conf_params["a4"] = [6, 43, 11, 97, 72, 73, 11, 71, 34, 70, 76, 13, 76, 7, 88, 28, 2, 94, 30, 68, 3, 95, 76]
    conf_params["mask"] = [[0, 12, 46, 47, 55, 56], [0, 12, 45, 46, 47, 53, 54, 55, 56], [0, 12, 44, 45, 46, 47, 53, 54, 55, 56], [0, 5, 6, 12, 44, 45, 46, 47, 53, 54, 55, 56], [0, 5, 6, 12, 43, 44, 45, 46, 47, 53, 54, 55, 56, 98], [39, 40, 41, 42, 49, 50, 51, 52, 60, 75], [38, 39, 40, 41, 42, 49, 50, 51, 52, 60, 73, 74, 75], [0, 5, 6, 11, 12, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 60, 72, 73, 74, 75, 97], [93], [0, 4, 5, 6, 11, 12, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 60, 71, 72, 73, 74, 75, 93, 97], [0, 4, 5, 6, 11, 12, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 60, 69, 70, 71, 72, 73, 74, 75, 93, 97], [16, 17, 18, 19, 20, 79, 80, 81, 82, 96], [15, 16, 17, 18, 19, 20, 77, 78, 79, 80, 81, 82, 96], [25, 48, 57, 58, 59, 84, 85, 86, 87], [1, 7, 22, 23, 24, 25, 48, 57, 58, 59, 83, 84, 85, 86, 87], [1, 7, 21, 22, 23, 24, 25, 48, 57, 58, 59, 83, 84, 85, 86, 87, 88, 89], [1, 7, 21, 22, 23, 24, 25, 26, 27, 48, 57, 58, 59, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92], [1, 2, 7, 8, 21, 22, 23, 24, 25, 26, 27, 48, 57, 58, 59, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92], [1, 2, 7, 8, 21, 22, 23, 24, 25, 26, 27, 28, 48, 57, 58, 59, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94], [29, 31, 61, 62, 63, 64, 65, 66, 67], [1, 2, 7, 8, 9, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94], [1, 2, 3, 7, 8, 9, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 48, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94], [1, 2, 3, 7, 8, 9, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 48, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 94, 95]]



### Parameters for the simulation
# Set of parameters to change
parameter_set = ["a", "b", "c", "beta", "trans", "rot", "conf"]
# Whether random choices are between all parameters (True) or between groups of parameters (False, like cell lengths, cell angles, conformers)
weighted = True
# Initial step size of each parameter: unit cell length, unit cell angle, translation (fraction of unit cell length), rotation, dihedral angle
init_step_size = [2., 20., 0.05, 30., 40.]
# How much does the amplitude of a step change after each step (divide by this value when rejected, multiply when accepted)
step_factor = 2.
# Maximum volume (ratio w.r.t. sum of VDW spheres in the molecule * number of molecules in unit cell) for the initial unit cell
max_V_factor = 2.
# Use smart cell generation
smart_cell = True
# Minimum and maximum values of unit cell lengths and angles
cell_params_lims = [1., 50., 45., 135.]
# Number of structures to run
n_structures = 999
# Number of Monte-Carlo loops (only valid if the criterion for stopping the MC run is not variable, otherwise maximum number of loops)
n_loops = 4000
# Criterion for stopping the MC run. T=temperature, [TODO: Additional criteria and T profiles]
stopping_criterion = "T"
T_start = 1000.
T_stop = 1.
T_profile = "linear"
# Threshold for stopping the simulation
A_thresh = 1e-3
# Gas constant (kJ/(mol*K))
gas_cst = 8.314e-3
# Perform an optimization of the parameters at the end of the MC run
opt = False
# Maximum number of iterations of the optimization (Powell-like method)
n_opt = 10
# Form of the cost function. E=Energy, H=1H RMSE, C=13C RMSE, [TODO: harmonic constraints based on exp.]
cost_function = ["E", "H"]
# Factors and options for each part of the cost function
# (For energy: factor = Ha -> kJ/mol)
cost_factors = {"E":2625.50, "H":200.}
cost_options = {"E":{}, "H":{"cutoff": 0.1, "slope": -1., "offset": 30.36}}
# Available nuclei for ShiftML
symb2z = {"H":1, "C":6, "N":7, "O":8}
z2symb = {1:"H", 6:"C", 7:"N", 8:"O"}

### Experimental shifts
exp_shifts = {}
equivalent = {}
ambiguous = {}
# 1H
if molecule == "cocaine":
    # Experimental chemical shifts: the shifts should be in the order of the input structure
    cost_options["H"]["exp_shifts"] = [3.76, 3.78, 5.63, 3.06, 3.32, 3.49, 3.38, 2.91, 2.56, 2.12, 8.01, 8.01, 8.01, 8.01, 8.01, 3.78, 3.78, 3.78, 1.04, 1.04, 1.04]
    # Equivalent shifts, where computed shifts should be averaged
    cost_options["H"]["equivalent"] = [[10, 11, 12, 13, 14], [15, 16, 17], [18, 19, 20]]
    # Ambiguous shifts, where best matching criterion should be used
    cost_options["H"]["ambiguous"] = [[3, 4], [6, 7], [8, 9]]
    
# 13C
# TODO: check 13C

### Parameters for the system
# Name of the DFTB+ program
dftb_pgm_name = "dftb+"
# Is the simulation running on the cluster?
cluster = True
# Set a seed for random number generation
seed = None
# Write crystal structure at each MC step
write_intermediates = False
# Verbosity
verbose = True


###########################################################################################
### Initialization
###########################################################################################
print("Initializing the system...")
# Finding dftb+ path on the local machine
if "E" in cost_function:
    cost_options["E"]["dftb_path"] = en.which(dftb_pgm_name)
# If a seed is set:
if seed:
    np.random.seed(seed)
if stopping_criterion == "T":
    T_list = mc.generate_T_profile(T_start, T_stop, n_loops, profile=T_profile)
# Create output directory
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
# Generate name of experiment
if stopping_criterion == "T":
    name = "{}_loops_".format(n_loops)
    for c in cost_function:
        name += "{}_factor_{}_".format(c, cost_factors[c])
    name += "{}_{}".format(comment, experiment)
    if opt:
        name += "_opt"
else:
    raise ValueError("Experiment name not implemented yet for the selected criterion ().".format(stopping_criterion))
if not os.path.exists(out_dir + name):
    os.mkdir(out_dir + name)
if "E" in cost_function:
    if cluster:
        cost_options["E"]["directory"] = "/dev/shm/"
    else:
        cost_options["E"]["directory"] = out_dir + name + "/"
# Load ShiftML kernels
krr = {}
representation = {}
trainsoaps = {}
model_numbers = {}
zeta = {}
for c in cost_function:
    if c in symb2z.keys():
        cost_options[c]["krr"], cost_options[c]["rep"], cost_options[c]["tr"], cost_options[c]["m_num"], cost_options[c]["zeta"] = ml.load_kernel(symb2z[c])
print("Done!")


###########################################################################################
### Loading initial structure
###########################################################################################
print("Loading initial structure...")
# Generate random number to set as name of the structure
random_number = np.random.random()
# Load initial structure
initial_structure = ase.io.read(initial_structure_original)
# Generate gas-phase molecule
cr.write_gas_phase_conformer(initial_structure, n_atoms, out_dir + name + "/{}_single_molecule.xyz".format(random_number))
print("Done!")

### For all structures
for k in range(n_structures):

    ###########################################################################################
    ### Monte-Carlo
    ###########################################################################################

    # Load the starting structure
    starting_structure = copy.deepcopy(initial_structure)
    # Generate a trial crystal structure
    best_crystal, lat, trans, R, conf_angles = cr.generate_crystal(starting_structure, n_atoms, n_mol, sg, parameter_set, cell_params_lims, n_conf=n_conf, conf_params=conf_params, smart_cell=smart_cell, max_V_factor=max_V_factor, verbose=verbose)
    
    # Write the initial trial crystal structure
    ase.io.write(out_dir + name + "/init_crystal_{}.cif".format(k), best_crystal)
    
    # Initialize arrays for monitoring
    # Parameter monitoring
    all_params = {}
    sel_params = {}
    acc_params = {}
    Amp = {}
    for p in parameter_set:
        all_params[p] = [mc.current_parameter(p, lat, trans, R, conf_angles)]
        sel_params[p] = 0
        acc_params[p] = 0
        Amp[p] = 1.
    # Cost function monitoring
    all_costs = {}
    all_costs["Tot"] = []
    for c in cost_function:
        all_costs[c] = []
        
    # Compute initial cost function
    old_cost = mc.compute_cost(best_crystal, cost_function, cost_factors, cost_options)
    new_cost = copy.deepcopy(old_cost)
    new_cost["Tot"] = 1.
    all_costs["Tot"].append(old_cost["Tot"])
    for c in cost_function:
        all_costs[c].append(old_cost[c])

    if stopping_criterion == "T":
        # Initialize time monitoring
        start = time.time()
        # Temperature-based profile
        for i, T in enumerate(T_list):
            
            if verbose:
                print("Loop {}/{}, T = {} K".format(i+1, n_loops, T))
            
            # Select a random parameter to change
            if weighted:
                weighted_parameter_set = copy.deepcopy(parameter_set)
                if "conf" in parameter_set:
                    for _ in range(len(conf_angles)-1):
                        weighted_parameter_set.append("conf")
                param_to_change = np.random.choice(weighted_parameter_set)
            else:
                param_to_change = np.random.choice(parameter_set)
            sel_params[param_to_change] += 1
            # Update the selected parameter
            trial_lat, trial_trans, trial_R, trial_conf = mc.randomize(param_to_change, lat, trans, R, conf_angles, init_step_size, A=Amp[param_to_change])
            clash = False
            
            # Check if the cell parameters are within the bounds, otherwise reject the step
            if np.min(trial_lat[:3]) < cell_params_lims[0] or np.max(trial_lat[:3]) > cell_params_lims[1] or np.min(trial_lat[3:]) < cell_params_lims[2] or np.max(trial_lat[3:]) > cell_params_lims[3]:
                clash = True
                
            if not clash:
                # Generate the new trial crystal
                trial_crystal, clash = cr.create_crystal(starting_structure, trial_lat, trial_trans, trial_R, trial_conf, sg, n_atoms, conf_params=conf_params)
            
            accept = False
            # Check the structure for clashes
            if not clash:
                # Check intramolecular clashes
                clash = cr.check_clash(trial_crystal, n_atoms, pbc=True, clash_type="intra", factor=0.85)
                if not clash:
                    # Check intermolecular clashes
                    clash = cr.check_clash(trial_crystal, n_atoms, pbc=True, clash_type="inter", factor=0.85)

            if not clash:
                # Compute the new cost function
                new_cost = mc.compute_cost(trial_crystal, cost_function, cost_factors, cost_options)
                
                # If energy decreases, accept the step, otherwise accept it with Boltzmann probability
                if new_cost["Tot"] < old_cost["Tot"]:
                    accept = True
                elif np.random.random() < np.exp((old_cost["Tot"] - new_cost["Tot"]) / (gas_cst*T)):
                    accept = True
            
            # If the step is accepted, update the cost function, parameters, and unit cell
            if verbose:
                stop = time.time()
                dt = stop-start
                eta = dt / (i + 1) * (len(T_list)-i-1)
            if accept:
                Amp[param_to_change] *= step_factor
                acc_params[param_to_change] += 1
                if verbose:
                    print("parameter changed: {}, energy difference: {} kJ/mol, step accepted, new amplitude: {}, time elapsed: {:.2f} s, ETA {:.2f} s.".format(param_to_change, new_cost["Tot"]-old_cost["Tot"], Amp[param_to_change], dt, eta))
                old_cost = copy.deepcopy(new_cost)
                best_crystal = copy.deepcopy(trial_crystal)
                lat = copy.deepcopy(trial_lat)
                trans = copy.deepcopy(trial_trans)
                R = copy.deepcopy(trial_R)
                conf_angles = copy.deepcopy(trial_conf)
            else:
                Amp[param_to_change] /= step_factor
                if verbose:
                    print("parameter changed: {}, energy difference: {} kJ/mol, step rejected, new amplitude: {}, time elapsed: {:.2f} s, ETA {:.2f} s.".format(param_to_change, new_cost["Tot"]-old_cost["Tot"], Amp[param_to_change], dt, eta))
            # Append the cost function and parameters for monitoring
            all_costs["Tot"].append(old_cost["Tot"])
            for c in cost_function:
                all_costs[c].append(old_cost[c])
            for p in parameter_set:
                all_params[p].append(mc.current_parameter(p, lat, trans, R, conf_angles))
        
            # Stop the MC simulation if the maximum amplitude is lower than the threshold
            for p in parameter_set:
                converged = True
                if Amp[p] > A_thresh:
                    converged = False
            if converged:
                print("Convergence reached!")
                break
                
        
        # Save parameter list
        if verbose:
            with open(out_dir + name + "/params_crystal_{}.pickle".format(k), "wb") as f:
                pk.dump([all_costs, all_params, sel_params, acc_params], f)
            
        # Write the final crystal structure
        ase.io.write(out_dir + name + "/final_crystal_{}.cif".format(k), best_crystal)
        
    if verbose:
        for p in parameter_set:
            print(p, all_params[p][-1])
        for p in all_costs.keys():
            print(p, all_costs[p][-1])
            
    ###########################################################################################
    ### Simplex optimization
    ###########################################################################################
    if opt:
        opt_lat = copy.deepcopy(lat)
        opt_trans = copy.deepcopy(trans)
        opt_R = copy.deepcopy(R)
        opt_conf = copy.deepcopy(conf_angles)
        opt_crystal, opt_lat, opt_trans, opt_R, opt_conf = mc.iterative_opt(starting_structure, opt_lat, opt_trans, opt_R, opt_conf, sg, n_atoms, parameter_set, cost_function, cost_factors, cost_options, cell_params_lims, conf_params=conf_params, n_max=n_opt, verbose=verbose)
        
        opt_cost = mc.compute_cost(opt_crystal, cost_function, cost_factors, cost_options)
        
        opt_params ={}
        for p in parameter_set:
            opt_params[p] = mc.current_parameter(p, opt_lat, opt_trans, opt_R, opt_conf)
        
        # Save parameter list
        if verbose:
            with open(out_dir + name + "/opt_params_crystal_{}.pickle".format(k), "wb") as f:
                pk.dump([opt_cost, opt_params], f)
            
        # Write the final crystal structure
        ase.io.write(out_dir + name + "/opt_crystal_{}.cif".format(k), opt_crystal)
