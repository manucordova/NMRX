#mport libraries
import numpy as np
import os
import sys
import copy
import pickle as pk
import time
import ase.io

### Path to the source files
sys.path.insert(0, "../src")

### Import source files
import mc_functions as mc
import ml_functions as ml
import energy_functions as en
import generate_crystals as cr

### Select space group, symmetry and number of molecules in the unit cell
sg = 19
n_mol = 4
sym = "O"

### Parameters for the chemical system under study
molecule = "azd5718"
experiment = sys.argv[1]
comment = ""

# Directory for results
out_dir = "../data/2022_02_05_azd5718_H_E/"

if molecule == "cocaine":
    # Path to initial file
    initial_structure_original = '../input_structures/cocaine/COCAIN10_H_relaxed_out_cell_vc-relax_DFTBplus_gas.xyz'

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
    conf_params["mask"] = [[9, 10, 11, 12, 13, 32, 33, 34, 35, 36], [8, 9, 10, 11, 12, 13, 19, 32, 33, 34, 35, 36],
                           [7, 8, 9, 10, 11, 12, 13, 19, 32, 33, 34, 35, 36], [15, 20, 21, 37, 38, 39],
                           [15, 37, 38, 39]]

if molecule == "azd":
    initial_structure_original = '../input_structures/azd8329/azd8329_reference_relaxed_again_dftb.xyz'
    n_atoms = 62
    n_conf = 5  # amino fixed
    conf_params = {}
    conf_params["a1"] = [41, 40, 55, 31, 59]
    conf_params["a2"] = [58, 39, 52, 36, 37]
    conf_params["a3"] = [42, 41, 38, 56, 33]
    conf_params["a4"] = [43, 61, 39, 57, 32]
    conf_params["mask"] = [
        [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 43, 44, 45, 46, 47, 48, 49, 50, 51],
        [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 58, 6, 61],
        [53, 54, 55, 28, 29, 30, 23, 22, 24, 25, 26, 27], [4, 60, 37, 59, 33, 32, 31, 35, 34, 1, 0, 2, 3], [4, 60, 59]]

if molecule == "azd2":
    initial_structure_original = '../input_structures/azd8329/AZD8329_Form1_dftb_vc_relax.xyz'
    n_atoms = 62
    n_conf = 5  # amino fixed
    conf_params = {}
    conf_params["a1"] = [17, 4, 0, 38, 25]
    conf_params["a2"] = [7, 6, 22, 8, 9]
    conf_params["a3"] = [1, 0, 13, 25, 20]
    conf_params["a4"] = [11, 14, 58, 43, 15]
    conf_params["mask"] = [[3, 2, 17], [3, 2, 7, 17, 1, 11, 31, 18, 4, 12, 32, 19, 5],
                           [58, 59, 60, 61, 51, 52, 53, 54, 44, 45, 46, 47],
                           [3, 2, 7, 17, 1, 11, 31, 18, 4, 12, 32, 19, 5, 6, 0, 14, 38, 39, 22, 13, 58, 59, 60, 61, 51,
                            52, 53, 54, 44, 45, 46, 47],
                           [3, 2, 7, 17, 1, 11, 31, 18, 4, 12, 32, 19, 5, 6, 0, 14, 38, 39, 22, 13, 58, 59, 60, 61, 51,
                            52, 53, 54, 44, 45, 46, 47, 8, 25, 43, 10]]

if molecule == "piroxicam":
    initial_structure_original = '../input_structures/piroxicam/Piroxicam_H_only_opt.xyz'

    # Number of atoms in one molecule
    n_atoms = 36

    # Number of dihedral angles
    n_conf = 2

    # Atoms and masks of the dihedral angles
    conf_params = {}
    conf_params["a1"] = [2,13]
    conf_params["a2"] = [13,15]
    conf_params["a3"] = [15,18]
    conf_params["a4"] = [18,17]
    conf_params["mask"] = [[4,5,6,7,8,9,23,24,25,26,0,1,3,10,11,16,31,12,27,28,29,2,14],
                           [4,5,6,7,8,9,23,24,25,26,0,1,3,10,11,16,31,12,27,28,29,2,14,13]]


if molecule == "ampicillin":
    # Path to initial file
    initial_structure_original = '../input_structures/ampicillin/ampicillin.xyz'

    # Number of atoms in one molecule
    n_atoms = 43

    # Number of dihedral angles
    n_conf = 5

    # Atoms and masks of the dihedral angles
    conf_params = {}
    conf_params["a1"] = [38,14,13,40,36]
    conf_params["a2"] = [1,13,35,2,3]
    conf_params["a3"] = [11,35,2,3,4]
    conf_params["a4"] = [5,2,3,4,10]
    conf_params["mask"] = [[38,39],[6,7,8,9,10,24,25,26,27,28,4,20,36,21,22,23,3,40,2,19],[6,7,8,9,10,24,25,26,27,28,4,20,36,21,22,23,3,40],[6,7,8,9,10,24,25,26,27,28,4,20,36,21,22,23],[6,7,8,9,10,24,25,26,27,28]]

if molecule == "azd5718":
    # Path to initial file
    initial_structure_original = '../input_structures/azd5718/azd5718_all_atoms_opt_vc_single_molecule.xyz'

    # Number of atoms in one molecule
    n_atoms = 59

    # Number of dihedral angles
    n_conf = 6

    # Atoms and masks of the dihedral angles
    conf_params = {}
    conf_params["a1"] = [7,14,23,22,36,28]
    conf_params["a2"] = [9,18,21,25,28,37]
    conf_params["a3"] = [11,21,22,28,37,42]
    conf_params["a4"] = [12,23,25,36,42,44]
    conf_params["mask"] = [[0,1,2,3,4,5,6,7,8,10],
                           [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20],
                           [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,23],
                           [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,29,30,31,32,33,34,35,38,39,40,41],
                           [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35,36,38,39,40,41],
                           [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,38,39,40,41,43]]

### Parameters for the simulation

# Set of parameters to change
parameter_set = ["a", "b", "c", "trans", "rot", "conf"]

# Whether random choices are between all parameters (True) or between groups of parameters (False, like cell lengths, cell angles, conformers)
weighted = True

# Initial step size of each parameter: unit cell length, unit cell angle, translation (fraction of unit cell length), rotation, dihedral angle
init_step_size = [2., 20., 0.05, 30., 40.]

# Maximum absolute step size of each parameter: unit cell length, unit cell angle, translation (fraction of unit cell length), rotation, dihedral angle
max_step_size = [20., 360., 1., 360., 360.]

# How much does the amplitude of a step change after each step (divide by this value when rejected, multiply when accepted)
step_factor = 2.

# Maximum volume (ratio w.r.t. sum of VDW spheres in the molecule * number of molecules in unit cell) for the initial unit cell
max_V_factor = 2.0

# Use smart cell generation
smart_cell = True

# Minimum and maximum values of unit cell lengths and angles
cell_params_lims = [1., 50., 45., 135.]

# Number of structures to run
n_structures = 999

# Number of Monte-Carlo loops (oly valid if the criterion for stopping the MC run is not variable, otherwise maximum number of loops)
n_loops = 4000

# Criterion for stopping the MC run. T=temperature
stopping_criterion = "T"
T_start = 2500.
T_stop = 50.
T_profile = "linear"

# Threshold for stopping the simulation
A_thresh = 1e-3

# Gas constant (kJ/(mol*K))
gas_cst = 8.314e-3

# Perform an optimization of the parameters at the end of the MC run
opt = False

# Maximum number of iterations of the optimization (Powell-like method)
n_opt = 10

# Perform hydrogen position relaxation every n_dftb step (-1 for no relaxation)
n_dftb = -1

# Maximum number of DFTB relaxation steps
n_opt_dftb = 50

# Form of the cost function. E=Energy, H=1H RMSE, C=13C RMSE
cost_function = ["E", "H"]

# Factors and options for each part of the cost function

# (For energy: factor = Ha -> kJ/mol)
cost_factors = {"E": 2625.50, "H": 200.}
cost_options = {"E": {}, "H": {"cutoff": 0.1, "slope": -1., "offset": 30.36}}

# Available nuclei for ShiftML
symb2z = {"H": 1, "C": 6, "N": 7, "O": 8}
z2symb = {1: "H", 6: "C", 7: "N", 8: "O"}

### Experimental shifts

exp_shifts = {}
equivalent = {}
ambiguous = {}

# 1H

if molecule == "cocaine":
    # Experimental chemical shifts: the shifts should be in the order of the input structure
    cost_options["H"]["exp_shifts"] = [3.9780189569673787, 3.636220267098391, 5.50111356170509, 1.7017083744995354,
                                       2.97372302702669, 3.6334961628397053, 3.4831881678204475, 1.7656191040126927,
                                       2.051699996346425, 1.6921747247077903, 7.867125539838806, 7.461451020956069,
                                       7.57521557453952, 7.566502423701898, 7.354128212230904, 4.212966601516133,
                                       3.3785319903128546, 3.695539645312305, 0.513272767915403, 2.5564618319438317,
                                       1.3211352909471152]

    # Equivalent shifts, where computed shifts should be averaged
    cost_options["H"]["equivalent"] = [[10, 11, 12, 13, 14], [15, 16, 17], [18, 19, 20]]

    # Ambiguous shifts, where best matching criterion should be used
    cost_options["H"]["ambiguous"] = [[3, 4], [6, 7], [8, 9]]

if molecule == "azd":
    # TEMPORARY: Use ShiftML predicted shifts instead of experimental ones
    cost_options["H"]["exp_shifts"] = [6.632220950611853, 8.286975180205935, 9.121433672167317, 8.270713472628717,
                                       17.126875438690167, 7.653589397307449, 10.432536862591157, 2.7245088002520745,
                                       1.5879187616373542, 1.6147006083574098, 2.461963453682408,1.5705020959214195,
                                       0.8425845800186913, 0.5534996234277436, 1.6083846885650814, 2.2320254393122774,
                                       1.8017573363824155, 0.8325122637515392, -0.09148673146667718, 1.4149643865838328,
                                       1.631624444343963, 0.9416009242089238, 0.4209544296796288, 0.4209544296796288,
                                       0.4209544296796288, 0.7252132605568775, 0.7252132605568775, 0.7252132605568775,
                                       -0.4611561955528316, -0.4611561955528316, -0.4611561955528316]

    # Equivalent shifts, where computed shifts should be averaged
    cost_options["H"]["equivalent"] = [[22, 23, 24], [25, 26, 27], [28, 29, 30]]

    # Ambiguous shifts, where best matching criterion should be used
    cost_options["H"]["ambiguous"] = [[9, 10], [20, 21], [12, 13], [17, 18]]

if molecule == "azd2":
    cost_options["H"]["exp_shifts"] = [9.204499364822265, 7.621297852503048, 4.693370241015021, 7.892352372852706,
                                       1.410986703871579, 8.04781786216055, 4.130864655660265, 1.7950741200761122,
                                       1.5908689663372186, 0.18552860342953537, 0.9140897889972557, 6.535734200276945,
                                       1.2442325173457967, 2.589213934025505, -0.006275014798418965, 9.041262366438989,
                                       1.5797344773520443, 0.9591675704675175, 1.6359860773621193, 1.6359860773621193,
                                       1.6359860773621193, 0.5252608043051872, 2.0623891790050486, -0.23114035197289695,
                                       -0.23114035197289695, -0.23114035197289695, -0.0757847725061005,
                                       -0.11428546533861805, -0.6095694461503349, -0.6095694461503349,
                                       -0.6095694461503349]

    cost_options["H"]["equivalent"] = [[18, 19, 20], [23, 24, 25], [28, 29, 30]]
    cost_options["H"]["ambiguous"] = [[8,9],[13,14],[16,17],[26,27]]

if molecule == "piroxicam":
    cost_options["H"]["exp_shifts"] = [5.865601673264457, 7.428880221959776, 6.18975251281427,
                                       7.016938873454549, 1.899096346628318, 1.899096346628318,
                                       1.899096346628318, 9.447586919088724, 10.49088013482896,
                                       6.154818550816437, 6.1638568152936095, 6.0754941317973135, 7.982058165933022]
    cost_options["H"]["equivalent"] = [[4, 5, 6]]
    cost_options["H"]["ambiguous"] = []

if molecule == "ampicillin":
    cost_options["H"]["exp_shifts"] = [1.6,1.6,1.6,7.5,4.8,10.0,10.0,10.0,7.3,7.3,7.3,7.3,7.3,4.0,5.2,6.6,0.6,0.6,0.6]
    cost_options["H"]["equivalent"] = [[0, 1, 2],[5, 6, 7],[8,9,10,11,12],[16,17,18]]
    cost_options["H"]["ambiguous"] = []

if molecule == "azd5718":
    cost_options["H"]["exp_shifts"] = [1.2,1.2,1.2,10.6,5.8,6.9,7.3,6.7,7.0,3.9,1.6,0.0,1.7,
                                       1.6,1.6,-0.5,0.8,-0.5,0.8,7.7,7.6,1.7,2.7,6.9,1.9,2.7]
    cost_options["H"]["equivalent"] = [[0,1,2]]
    cost_options["H"]["ambiguous"] = [[5,6],[7,8,20],[11,12],[13,14],[15,16,17,18],[21,22],[24,25]]

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

# To follow how parameters change in each step
print_param_change = False

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

# Generate temperature profile
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
        cost_options[c]["krr"], cost_options[c]["rep"], cost_options[c]["tr"], cost_options[c]["m_num"], \
        cost_options[c]["zeta"] = ml.load_kernel(symb2z[c])

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
cr.write_gas_phase_conformer(initial_structure, n_atoms,
                             out_dir + name + "/{}_single_molecule.xyz".format(random_number))
print("Done!")

### For all structures

for k in range(n_structures):

    ###########################################################################################

    ### Monte-Carlo

    ###########################################################################################

    # Load the starting structure
    starting_structure = copy.deepcopy(initial_structure)

    # Generate a trial crystal structure
    best_crystal, lat, trans, R, conf_angles = cr.generate_crystal(starting_structure, n_atoms, n_mol, sg,
                                                                   parameter_set, cell_params_lims, n_conf=n_conf,
                                                                   conf_params=conf_params, smart_cell=smart_cell,
                                                                   max_V_factor=max_V_factor, verbose=verbose)

    # Write the initial trial crystal structure
    ase.io.write(out_dir + name + "/init_crystal_{}.cif".format(k), best_crystal)

    ## Initialize arrays for monitoring

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
            if i > 0 and n_dftb >= 0 and i % n_dftb == 0:
                print("Optimizing proton positions for {} steps...".format(n_opt_dftb))
                tmp_structure = en.dftb_relax(cost_options["E"]["directory"], best_crystal,
                                              cost_options["E"]["dftb_path"], n_opt)
                tmp_mol = cr.retrieve_initial_structure(tmp_structure, n_atoms, starting_structure, n_conf=n_conf,
                                                        conf_params=conf_params)
                tmp_crystal, clash = cr.create_crystal(tmp_mol, lat, trans, R, conf_angles, sg, n_atoms,
                                                       conf_params=conf_params)
                tmp_cost = mc.compute_cost(tmp_crystal, cost_function, cost_factors, cost_options)
                if tmp_cost["Tot"] < old_cost["Tot"]:
                    print("Cost decreased by {:.2f} kJ/mol".format(old_cost["Tot"] - tmp_cost["Tot"]))
                    old_cost = copy.deepcopy(tmp_cost)
                    starting_structure = copy.deepcopy(tmp_mol)
                    best_crystal = copy.deepcopy(tmp_crystal)

                else:
                    print("Optimization did not yield a lower energy structure.")

            if verbose:
                print("Loop {}/{}, T = {} K".format(i + 1, n_loops, T))

            # Select a random parameter to change

            if weighted:
                weighted_parameter_set = copy.deepcopy(parameter_set)
                if "conf" in parameter_set:
                    for _ in range(len(conf_angles) - 1):
                        weighted_parameter_set.append("conf")
                param_to_change = np.random.choice(weighted_parameter_set)

            else:
               param_to_change = np.random.choice(parameter_set)
            sel_params[param_to_change] += 1

            # Update the selected parameter
            trial_lat, trial_trans, trial_R, trial_conf = mc.randomize(param_to_change, lat, trans, R, conf_angles,
                                                                       init_step_size, A=Amp[param_to_change])

            clash = False

            # Check if the cell parameters are within the bounds, otherwise reject the step
            if np.min(trial_lat[:3]) < cell_params_lims[0] or np.max(trial_lat[:3]) > cell_params_lims[1] or np.min(
                    trial_lat[3:]) < cell_params_lims[2] or np.max(trial_lat[3:]) > cell_params_lims[3]:
                clash = True

            if not clash:
                # Generate the new trial crystal
                trial_crystal, clash = cr.create_crystal(starting_structure, trial_lat, trial_trans, trial_R,
                                                         trial_conf, sg, n_atoms, conf_params=conf_params)

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
                elif np.random.random() < np.exp((old_cost["Tot"] - new_cost["Tot"]) / (gas_cst * T)):
                    accept = True
            # If the step is accepted, update the cost function, parameters, and unit cell

            if verbose:
                stop = time.time()
                dt = stop - start
                eta = dt / (i + 1) * (len(T_list) - i - 1)

            if accept:
                Amp[param_to_change] *= step_factor
                # If the amplitude makes a step go over the maximum step size, reduce it
                Amp = mc.normalize_amplitude(Amp, param_to_change, init_step_size, max_step_size, verbose=verbose)
                acc_params[param_to_change] += 1

                if verbose:
                    print(
                        "parameter changed: {}, energy difference: {} kJ/mol, step accepted, new amplitude: {}, time elapsed: {:.2f} s, ETA {:.2f} s.".format(
                            param_to_change, new_cost["Tot"] - old_cost["Tot"], Amp[param_to_change], dt, eta))
                old_cost = copy.deepcopy(new_cost)
                best_crystal = copy.deepcopy(trial_crystal)
                lat = copy.deepcopy(trial_lat)
                trans = copy.deepcopy(trial_trans)
                R = copy.deepcopy(trial_R)
                conf_angles = copy.deepcopy(trial_conf)

            else:
                Amp[param_to_change] /= step_factor
                # If the amplitude makes a step go over the maximum step size, reduce it
                Amp = mc.normalize_amplitude(Amp, param_to_change, init_step_size, max_step_size, verbose=verbose)
                if verbose:
                    print(
                        "parameter changed: {}, energy difference: {} kJ/mol, step rejected, new amplitude: {}, time elapsed: {:.2f} s, ETA {:.2f} s.".format(
                            param_to_change, new_cost["Tot"] - old_cost["Tot"], Amp[param_to_change], dt, eta))

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

            if write_intermediates:
                ase.io.write(out_dir + name + "/intermediate_crystal_{}.cif".format(i), trial_crystal)
            if print_param_change:
                print("a: " + str(all_params["a"][-1]), "b: " + str(all_params["b"][-1]),
                      "c: " + str(all_params["c"][-1]), "alpha: " + str(all_params["alpha"][-1]),
                      "beta: " + str(all_params["beta"][-1]), "gamma: " + str(all_params["gamma"][-1]),
                      "trans: " + str(all_params["trans"][-1]), "rot: " + str(all_params["rot"][-1]),
                      "conf: " + str(all_params["conf"][-1]))

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
