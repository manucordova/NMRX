#####
# Last modified 29.06.2020
# Re-writing the script so that it is more general
# Authors: Manuel Cordova, Martins Balodis
#####



### Import libraries
import numpy as np
import sys
import copy
import scipy.optimize as op
#import ase
#import ase.io
#import ase.spacegroup
#import fnmatch

### Import custom libraries
sys.path.insert(0, "../src")
# Import source files
import new_ml_functions as ml
import new_energy_functions as en
import new_generate_crystals as cr

# Available nuclei for ShiftML
symb2z = {"H":1, "C":6, "N":7, "O":8}


def generate_T_profile(T_init, T_final, N, profile="linear", **kwargs):
    """
    Generate temperature profile
    
    Inputs:     - T_init        Initial temperature [K]
                - T_final       Final temperature [K]
                - N             Number of points in temperature profile
                - profile       Form of the temperature profile
                - **kwargs      Additional arguments to define the form of the profile
    
    Outputs:    - T_profile     Array of temperatures
    """
    if profile == "linear":
        T_profile = np.linspace(T_init, T_final, N)
    else:
        raise ValueError("Temperature profile not implemented yet! ({})".format(form))
    return T_profile



def compute_cost_part(struct, cost_type, cost_options):
    """
    Compute one part of the cost function
    
    Inputs:     - struct            Crystal structure
                - cost_type         Cost function part
                - cost_options      Dictionary of options specific to each cost function part
                - exp_shifts        Experimental chemical shifts
    
    Outputs:    - cost              Value of the cost computed
    """
    
    if cost_type == "E":
        cost = en.dftbplus_energy(cost_options["directory"], struct, cost_options["dftb_path"])
    elif cost_type in symb2z.keys():
        cost = ml.shift_rmsd(struct, cost_type, cost_options)
    else:
        raise ValueError("Unknown cost function type: {}".format(cost_type))

    return cost



def compute_cost(struct, cost_function, cost_factors, cost_options):
    """
    Compute the cost function
    
    Inputs:     - struct            Crystal structure
                - cost_function     Array of the cost function parts
                - cost_factors      Dictionary of the factor of each cost function part
                - cost_options      Dictionary of options specific to each cost function part
    
    Outputs:    - cost              Dictionary of the total cost function and of its parts
    """
    
    cost = {}
    cost["Tot"] = 0
    
    for c in cost_function:
        cost[c] = compute_cost_part(struct, c, cost_options[c])
        cost["Tot"] += cost[c]*cost_factors[c]
    
    return cost



def generate_random_unit_vector():
    """
    Generate a random unit vector
    
    Outputs:    - v     Random unit vector
    """
    
    v = np.zeros(3)
    v[0] = np.random.normal(loc=0.0, scale=1.0)
    v[1] = np.random.normal(loc=0.0, scale=1.0)
    v[2] = np.random.normal(loc=0.0, scale=1.0)
    
    return v / np.linalg.norm(v)



def randomize(param_to_change, lat, trans, R, conf_angles, step_list, A=1.):
    """
    Randomize a randomly selected parameter, with amplitude given by the step in step list and scaled by A.
    
    Inputs:     - parameter_set     Set of parameters
                - param_to_change   Parameter to randomly change
                - lat               Unit cell parameters
                - trans             Translation vector
                - R                 Rotation matrix
                - conf_angles       List of conformational angles
                - step_list         List of step size for each parameters
                - A                 Amplitude of the change
    
    Outputs:    - new_lat           Updated unit cell parameters
                - new_trans         Updated translation vector
                - new_R             Updated rotation matrix
                - new_conf          Updated conformational angles
    """
    
    new_lat = copy.deepcopy(lat)
    new_trans = copy.deepcopy(trans)
    new_R = copy.deepcopy(R)
    new_conf = copy.deepcopy(conf_angles)
    
    if param_to_change == "a":
        new_lat[0] = lat[0] + (1 - (2 * np.random.random())) * step_list[0] * A
    if param_to_change == "b":
        new_lat[1] = lat[1] + (1 - (2 * np.random.random())) * step_list[0] * A
    if param_to_change == "c":
        new_lat[2] = lat[2] + (1 - (2 * np.random.random())) * step_list[0] * A
    
    if param_to_change == "alpha":
        new_lat[3] = lat[3] + (1 - (2 * np.random.random())) * step_list[1] * A
    if param_to_change == "beta":
        new_lat[4] = lat[4] + (1 - (2 * np.random.random())) * step_list[1] * A
    if param_to_change == "gamma":
        new_lat[5] = lat[5] + (1 - (2 * np.random.random())) * step_list[1] * A
    
    if param_to_change == "trans":
        v = generate_random_unit_vector()
        new_trans += v * step_list[2] * A
        change = True
        while change:
            change = False
            for i in range(3):
                if new_trans[i] >= 1.:
                    change = True
                    new_trans[i] -= 1.
                if new_trans[i] < 0.:
                    change = True
                    new_trans[i] += 1.
    
    if param_to_change == "rot":
        new_R = cr.generate_random_rot(step_list[3] * A)
        new_R = new_R.dot(R)
    
    if param_to_change == "conf":
        conf_list = list(range(len(conf_angles)))
        i = np.random.choice(conf_list)
        new_conf[i] += (1 - (2 * np.random.random())) * step_list[4] * A
        while new_conf[i] < 0.:
            new_conf[i] += 360.
        while new_conf[i] >= 360.:
            new_conf[i] -= 360.
    
    return new_lat, new_trans, new_R, new_conf



def current_parameter(p, lat, trans, R, conf_angles):
    """
    Obtain the current value of parameter p.
    
    Inputs:     - p             Parameter to monitor
                - lat           Unit cell parameters
                - trans         Translation vector
                - R             Rotation matrix
                - conf_angles   List of conformational angles
                
    outputs:    - v             Value of parameter p
    """
    
    if p == "a":
        return lat[0]
    if p == "b":
        return lat[1]
    if p == "c":
        return lat[2]
    
    if p == "alpha":
        return lat[3]
    if p == "beta":
        return lat[4]
    if p == "gamma":
        return lat[5]
    
    if p == "trans":
        return trans
    
    if p == "rot":
        return R
    
    if p == "conf":
        return conf_angles
    
    raise ValueError("Unknown parameter: {}".format(p))
    return



def to_minimize(x, struct, lat, trans, R, conf_angles, sg, n_atoms, parameter_set, cost_function, cost_factors, cost_options, conf_params, verbose):
    """
    Function to minimize in the simplex algorithm
    
    Inputs:     - x                     List of values of the parameters to optimize
                - struct                Initial crystal structure
                - lat                   Starting lattice parameters
                - trans                 Starting translation vector (fractions of the unit cell)
                - R                     Starting rotation matrix
                - conf_angles           Starting conformational angles
                - sg                    Space group of the crystal
                - n_atoms               Number of atoms in the molecule
                - parameter_set         Set of parameters to optimize
                - cost_function         Form of the cost function
                - cost_factors          Factors of each part of the cost function
                - cost_options          Options for the parts of the cost function
                - conf_params           Atoms and mask in conformer angles
                - verbose               Verbosity level
                
    Outputs:    - cost["Tot"]           Value of the cost function
    """
    new_lat = copy.deepcopy(lat)
    new_trans = copy.deepcopy(trans)
    new_R = copy.deepcopy(R)
    new_conf = copy.deepcopy(conf_angles)
    
    print(x)
    
    # Set optimized parameters
    k = 0
    # Lattice parameters
    for p in ["a", "b", "c", "alpha", "beta", "gamma"]:
        if p in parameter_set:
            new_lat[k] = x[k]
            k += 1
    # Translation
    if "trans" in parameter_set:
        new_trans = x[k:k+3]
        k += 3
    # Rotation
    if "rot" in parameter_set:
        r = np.array(x[k:k+4])
        r[:3] /= np.linalg.norm(r[:3])
        new_R = cr.rotation_matrix(x[k:k+4])
        new_R = new_R.dot(R)
        k += 4
    # Conformation
    if "conf" in parameter_set:
        new_conf = x[k:]
    
    new_cryst, clash = cr.create_crystal(struct, new_lat, new_trans, new_R, new_conf, sg, n_atoms, conf_params=conf_params)
    
    if not clash:
        # Check intramolecular clashes
        clash = cr.check_clash(new_cryst, n_atoms, pbc=True, clash_type="intra", factor=0.85)
        if not clash:
            # Check intermolecular clashes
            clash = cr.check_clash(new_cryst, n_atoms, pbc=True, clash_type="inter", factor=0.85)
    
    if clash:
        print("CLASH")
        cost = {}
        cost["Tot"] = 1e12
    else:
        cost = compute_cost(new_cryst, cost_function, cost_factors, cost_options)
        
    print(cost)
    
    return cost["Tot"]



def simplex_opt(struct, lat, trans, R, conf_angles, sg, n_atoms, parameter_set, cost_function, cost_factors, cost_options, conf_params=None, verbose=False):
    """
    Optimize the cost function with respect to parameters
    
    Inputs:     - struct                Initial crystal structure
                - lat                   Starting lattice parameters
                - trans                 Starting translation vector (fractions of the unit cell)
                - R                     Starting rotation matrix
                - conf_angles           Starting conformational angles
                - sg                    Space group of the crystal
                - n_atoms               Number of atoms in the molecule
                - parameter_set         Set of parameters to optimize
                - cost_function         Form of the cost function
                - cost_factors          Factors of each part of the cost function
                - cost_options          Options for the parts of the cost function
                - conf_params           Atoms and mask in conformer angles
                - verbose               Verbosity level
                
    Outputs:    - opt_crystal           Optimized crystal structure
                - opt_lat               Optimized lattice parameters
                - opt_trans             Optimized translation vector
                - opt_R                 Optimized rotation matrix
                - opt_conf              Optimized conformational angles
    """
    
    # Set initial parameters
    x0 = []
    bounds = []
    # Lattice parameters
    for k, p in enumerate(["a", "b", "c", "alpha", "beta", "gamma"]):
        if p in parameter_set:
            x0.append(lat[k])
            bounds.append((0, 2*lat[k]))
    # Translation
    if "trans" in parameter_set:
        x0.extend(list(trans))
        bounds.extend([(0., 1.), (0., 1.), (0., 1.)])
    # Rotation
    if "rot" in parameter_set:
        x0.extend([1., 0., 0., 0.])
        bounds.extend([(-1., 1.), (-1., 1.), (-1., 1.), (-180., 180.)])
    # Conformation
    if "conf" in parameter_set:
        x0.extend(list(conf_angles))
        for i in range(len(conf_angles)):
            bounds.append((0., 360.))

    # Optimize cost function
    res = op.minimize(to_minimize, x0, args=(struct, lat, trans, R, conf_angles, sg, n_atoms, parameter_set, cost_function, cost_factors, cost_options, conf_params, verbose), method="TNC", bounds=bounds, options={"eps":1e-4, "ftol":1e-2})
    
    opt_lat = copy.deepcopy(lat)
    opt_trans = copy.deepcopy(trans)
    opt_R = copy.deepcopy(R)
    opt_conf = copy.deepcopy(conf_angles)
    
    # Set optimized parameters
    k = 0
    # Lattice parameters
    for p in ["a", "b", "c", "alpha", "beta", "gamma"]:
        if p in parameter_set:
            opt_lat[k] = res.x[k]
            k += 1
    # Translation
    if "trans" in parameter_set:
        opt_trans = res.x[k:k+3]
        k += 3
    # Rotation
    if "rot" in parameter_set:
        opt_R = cr.rotation_matrix(res.x[k:k+4])
        opt_R = opt_R.dot(R)
        k += 4
    # Conformation
    if "conf" in parameter_set:
        opt_conf = res.x[k:]
    
    # Generate optimized crystal
    opt_crystal, _ = cr.create_crystal(struct, opt_lat, opt_trans, opt_R, opt_conf, sg, n_atoms, conf_params=conf_params)
    
    return opt_crystal, opt_lat, opt_trans, opt_R, opt_conf
