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
    elif cost_type == "D":
        cost = en.compute_distance_constraints(struct, cost_options["n_atoms"], cost_options["pairs"], cost_options["thresh"], cost_options["exponent"], cost_options["contact"], cost_options["c_type"])
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
        # Regularize translation vector such that each component is always within [0, 1[
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
        # Regularize conformational angles such that they are always within [0, 360[
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
    
    # Set optimized parameters
    k = 0
    # Lattice parameters
    for m, p in enumerate(["a", "b", "c", "alpha", "beta", "gamma"]):
        if p in parameter_set:
            new_lat[m] = x[k]
            k += 1
    # Translation
    if "trans" in parameter_set:
        new_trans = x[k:k+3]
        k += 3
    # Rotation
    if "rot" in parameter_set:
        rx = x[k]
        ry = x[k+1]
        rz = x[k+2]
        Rx = cr.rotation_matrix([1., 0., 0., rx])
        Ry = cr.rotation_matrix([0., 1., 0., ry])
        Rz = cr.rotation_matrix([0., 0., 1., rz])
        new_R = Rx.dot(Ry.dot(Rz.dot(R)))
        k += 3
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
    
    return cost["Tot"]



def simplex_opt(struct, lat, trans, R, conf_angles, sg, n_atoms, parameter_set, cost_function, cost_factors, cost_options, cell_params_lims, conf_params=None, verbose=False):
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
    
    # Set initial parameters and boundaries
    x0 = []
    bounds = []
    # Lattice parameters
    for k, p in enumerate(["a", "b", "c", "alpha", "beta", "gamma"]):
        if p in parameter_set:
            x0.append(lat[k])
            if p in ["a", "b", "c"]:
                bounds.append((cell_params_lims[0], cell_params_lims[1]))
            elif p in ["alpha", "beta", "gamma"]:
                bounds.append((cell_params_lims[2], cell_params_lims[3]))
            else:
                raise ValueError("Unknown parameter: {}".format(p))
    # Translation
    if "trans" in parameter_set:
        x0.extend(list(trans))
        bounds.extend([(0., 1.), (0., 1.), (0., 1.)])
    # Rotation
    if "rot" in parameter_set:
        x0.extend([0., 0., 0.])
        bounds.extend([(-180., 180.), (-180., 180.), (-180., 180.)])
    # Conformation
    if "conf" in parameter_set:
        x0.extend(list(conf_angles))
        for _ in range(len(conf_angles)):
            bounds.append((0., 360.))

    # Optimize cost function
    res = op.minimize(to_minimize, x0, args=(struct, lat, trans, R, conf_angles, sg, n_atoms, parameter_set, cost_function, cost_factors, cost_options, conf_params, verbose), method="Nelder-Mead", options={"fatol":1e-2})
    
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



def optimize_lat_param(struct, p, best_cost, bounds, lat, trans, R, conf_angles, sg, n_atoms, cost_function, cost_factors, cost_options, conf_params=None, step=1e-2, ftol=1e-2, xtol=1e-4, thresh_type="abs", verbose=False):
    """
    Optimize the cost function with respect to all lattice parameters, iteratively.
    
    Inputs:     - struct                Input crystal structure
                - p                     Cell parameter to change
                - best_cost             Initial cost
                - bounds                Bounds for the parameter
                - lat                   Initial lattice parameters
                - trans                 Translation vector
                - R                     Rotation matrix
                - conf_angles           Conformational angles
                - sg                    Space group of the crystal
                - n_atoms               Number of atoms in the molecules
                - cost_function         Form of the cost function
                - cost_factors          Factors of each part of the cost function
                - cost_options          Options for the parts of the cost function
                - conf_params           Atoms and masks in conformer angles
                - step                  Step size (fraction of the range of parameter)
                - ftol                  Tolerance for convergence of the cost function
                - xtol                  Tolerance for convergence of the step size
                - thresh_type           Set to "abs" for absolute convergence of the cost function,
                                            "rel" for relative convergence
                
    Outputs:    - best_lat              Optimized lattice parameters
                - best_cost             Optimized cost function
    """
    
    best_lat = copy.deepcopy(lat)
    lat_ind = {"a":0, "b":1, "c":2, "alpha":3, "beta":4, "gamma":5}
    
    if verbose:
        print("  Optimizing parameter {}. Initial value {}, step size {}".format(p, lat[lat_ind[p]], step))
    
    # Display warning if the initial parameters are outside the bounds
    if lat[lat_ind[p]] > bounds[1] or lat[lat_ind[p]] < bounds[0]:
        print("  WARNING: the initial parameters are outside the bounds")
    
    # If the step size is too small, return the original lattice parameter
    if step < xtol:
        return best_lat, best_cost

    # Obtain step size
    span = bounds[1]-bounds[0]
    dx = span * step
    
    # Generate steps in positive and negative directions
    tmp_lat_p = copy.copy(best_lat)
    tmp_lat_p[lat_ind[p]] += dx
    tmp_lat_m = copy.copy(best_lat)
    tmp_lat_m[lat_ind[p]] -= dx
    
    # Restrict parameters to be within the bounds
    if tmp_lat_p[lat_ind[p]] > bounds[1]:
        tmp_lat_p[lat_ind[p]] = bounds[1]
    if tmp_lat_p[lat_ind[p]] < bounds[0]:
        tmp_lat_p[lat_ind[p]] = bounds[0]
    if tmp_lat_m[lat_ind[p]] > bounds[1]:
        tmp_lat_m[lat_ind[p]] = bounds[1]
    if tmp_lat_m[lat_ind[p]] < bounds[0]:
        tmp_lat_m[lat_ind[p]] = bounds[0]
    
    # Generate modified crystals
    cryst_p, clash_p = cr.create_crystal(struct, tmp_lat_p, trans, R, conf_angles, sg, n_atoms, conf_params=conf_params)
    cryst_m, clash_m = cr.create_crystal(struct, tmp_lat_m, trans, R, conf_angles, sg, n_atoms, conf_params=conf_params)
    
    # Compute cost of the modified crystals
    if clash_p:
        tmp_cost_p["Tot"] = 1e12
    else:
        tmp_cost_p = compute_cost(cryst_p, cost_function, cost_factors, cost_options)
    if clash_m:
        tmp_cost_m["Tot"] = 1e12
    else:
        tmp_cost_m = compute_cost(cryst_m, cost_function, cost_factors, cost_options)
    
    # If no direction is found, decrease the step size by two and try again
    if tmp_cost_p["Tot"] >= best_cost["Tot"]-1e-6 and tmp_cost_m["Tot"] >= best_cost["Tot"]-1e-6:
        if verbose:
            print("  Minimum found, decreasing step size")
        return optimize_lat_param(struct, p, best_cost, bounds, lat, trans, R, conf_angles, sg, n_atoms, cost_function, cost_factors, cost_options, conf_params=conf_params, step=step/2., ftol=ftol, xtol=xtol, thresh_type=thresh_type, verbose=verbose)
        
    # If positive direction is found, increase the parameter until energy rises
    elif tmp_cost_p["Tot"] < best_cost["Tot"] and tmp_cost_p["Tot"] < tmp_cost_m["Tot"]:
        if verbose:
            print("  Increasing the parameter decreases the cost")
        # While the energy decreases
        while tmp_cost_p["Tot"] < best_cost["Tot"]:
            # Update the best parameter and associated cost
            best_cost = copy.copy(tmp_cost_p)
            best_lat = copy.copy(tmp_lat_p)
            # Generate a new parameter
            tmp_lat_p[lat_ind[p]] += dx
            # Generate the corresponding crystal
            cryst_p, clash_p = cr.create_crystal(struct, tmp_lat_p, trans, R, conf_angles, sg, n_atoms, conf_params=conf_params)
            # Compute cost
            if clash_p:
                tmp_cost_p["Tot"] = 1e12
            else:
                tmp_cost_p = compute_cost(cryst_p, cost_function, cost_factors, cost_options)
            if verbose:
                print("  Cost before this step: {:.2f}, cost after: {:.2f}".format(best_cost["Tot"], tmp_cost_p["Tot"]))
        # Restart the optimization with reduced step size
        return optimize_lat_param(struct, p, best_cost, bounds, best_lat, trans, R, conf_angles, sg, n_atoms, cost_function, cost_factors, cost_options, conf_params=conf_params, step=step/2., ftol=ftol, xtol=xtol, thresh_type=thresh_type, verbose=verbose)
        
    # If negative direction is found, increase the parameter until energy rises
    elif tmp_cost_m["Tot"] < best_cost["Tot"] and tmp_cost_m["Tot"] < tmp_cost_p["Tot"]:
        if verbose:
            print("  Decreasing the parameter decreases the cost")
        # While the energy decreases
        while tmp_cost_m["Tot"] < best_cost["Tot"]:
            # Update the best parameter and associated cost
            best_cost = copy.copy(tmp_cost_m)
            best_lat = copy.copy(tmp_lat_m)
            # Generate a new parameter
            tmp_lat_m[lat_ind[p]] -= dx
            # Generate the corresponding crystal
            cryst_m, clash_m = cr.create_crystal(struct, tmp_lat_m, trans, R, conf_angles, sg, n_atoms, conf_params=conf_params)
            # Compute cost
            if clash_m:
                tmp_cost_m["Tot"] = 1e12
            else:
                tmp_cost_m = compute_cost(cryst_m, cost_function, cost_factors, cost_options)
            if verbose:
                print("  Cost before this step: {:.2f}, cost after: {:.2f}".format(best_cost["Tot"], tmp_cost_m["Tot"]))
        # Restart the optimization with reduced step size
        return optimize_lat_param(struct, p, best_cost, bounds, best_lat, trans, R, conf_angles, sg, n_atoms, cost_function, cost_factors, cost_options, conf_params=conf_params, step=step/2., ftol=ftol, xtol=xtol, thresh_type=thresh_type, verbose=verbose)
    
    else:
        raise ValueError("Error during the optimization of {}".format(p))
        return



def roll_zeropad(a, shift, axis=None):
    """
    Roll array elements along a given axis. Elements off the end of the array are treated as zeros.

    Inputs:     - a         Input array
                - shift     Number of places by which elements are shifted
                - axis      Axis along which elements are shifted
                
    Outputs:    - res       Shifted array
    """
    
    a = np.asanyarray(a)
    if shift == 0: return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift,n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift,n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res



def optimize_trans(struct, best_cost, bounds, lat, trans, R, conf_angles, sg, n_atoms, cost_function, cost_factors, cost_options, conf_params=None, step=1e-2, ftol=1e-2, xtol=1e-4, thresh_type="abs", verbose=False):
    """
    Optimize the cost function with respect to the translation vector.
    
    Inputs:     - struct                Input crystal structure
                - best_cost             Initial cost
                - bounds                Bounds for the parameter
                - lat                   Lattice parameters
                - trans                 Initial translation vector
                - R                     Rotation matrix
                - conf_angles           Conformational angles
                - sg                    Space group of the crystal
                - n_atoms               Number of atoms in the molecules
                - cost_function         Form of the cost function
                - cost_factors          Factors of each part of the cost function
                - cost_options          Options for the parts of the cost function
                - conf_params           Atoms and masks in conformer angles
                - step                  Step size (fraction of the range of parameter)
                - ftol                  Tolerance for convergence of the cost function
                - xtol                  Tolerance for convergence of the step size
                - thresh_type           Set to "abs" for absolute convergence of the cost function,
                                            "rel" for relative convergence
                
    Outputs:    - best_trans            Optimized translation vector
                - best_cost             Optimized cost function
    """
    
    if verbose:
        print("  Optimizing translation, step size {}".format(step))
        print("  Initial translation vector: {:.4f}, {:.4f}, {:.4f}".format(trans[0], trans[1], trans[2]))
    
    best_trans = copy.deepcopy(trans)
    
    # If the step size is too small, return the original lattice parameter
    if step < xtol:
        return best_trans, best_cost

    # Obtain step size
    span = bounds[1]-bounds[0]
    dx = span * step
    
    # Initialize grid of steps and costs
    trans_grid = np.zeros((3,3,3,3))
    cost_grid = np.zeros((3,3,3))
    
    # Initialize optimized translation vector and cost
    opt_trans = copy.deepcopy(best_trans)
    opt_inds = (1,1,1)
    opt_cost = copy.deepcopy(best_cost)
    cost_grid[1,1,1] = best_cost["Tot"]
    
    # Check that any direction yields an improvement
    improve = False
    for i in range(3):
        # Positive displacement along axis i
        tmp_trans = copy.deepcopy(best_trans)
        tmp_trans[i] += dx
        tmp_cryst, clash = cr.create_crystal(struct, lat, tmp_trans, R, conf_angles, sg, n_atoms, conf_params=conf_params)
        if clash:
            tmp_cost ={}
            tmp_cost["Tot"] = 1e12
        else:
            tmp_cost = compute_cost(tmp_cryst, cost_function, cost_factors, cost_options)
        
        # If we found a new minimum, update the optimized cost, translation vector and index
        if tmp_cost["Tot"] < opt_cost["Tot"]:
            opt_cost = copy.deepcopy(tmp_cost)
            opt_trans = copy.deepcopy(tmp_trans)
            if i == 0:
                opt_inds =(2,1,1)
            elif i == 1:
                opt_inds =(1,2,1)
            elif i == 2:
                opt_inds =(1,1,2)
        
        # Set cost grid values already computed
        if i == 0:
            cost_grid[2, 1, 1] = tmp_cost["Tot"]
        elif i == 1:
            cost_grid[1, 2, 1] = tmp_cost["Tot"]
        elif i == 2:
            cost_grid[1, 1, 2] = tmp_cost["Tot"]
        
        if tmp_cost["Tot"] < best_cost["Tot"]:
            improve = True
            
        # Negative displacement along axis i
        tmp_trans = copy.deepcopy(best_trans)
        tmp_trans[i] -= dx
        tmp_cryst, clash = cr.create_crystal(struct, lat, tmp_trans, R, conf_angles, sg, n_atoms, conf_params=conf_params)
        if clash:
            tmp_cost ={}
            tmp_cost["Tot"] = 1e12
        else:
            tmp_cost = compute_cost(tmp_cryst, cost_function, cost_factors, cost_options)
            
        # If we found a new minimum, update the optimized cost, translation vector and index
        if tmp_cost["Tot"] < opt_cost["Tot"]:
            opt_cost = copy.deepcopy(tmp_cost)
            opt_trans = copy.deepcopy(tmp_trans)
            if i == 0:
                opt_inds =(0,1,1)
            elif i == 1:
                opt_inds =(1,0,1)
            elif i == 2:
                opt_inds =(1,1,0)
        
        # Set cost grid values already computed
        if i == 0:
            cost_grid[0, 1, 1] = tmp_cost["Tot"]
        elif i == 1:
            cost_grid[1, 0, 1] = tmp_cost["Tot"]
        elif i == 2:
            cost_grid[1, 1, 0] = tmp_cost["Tot"]
        
        if tmp_cost["Tot"] < best_cost["Tot"]:
            improve = True
    
    if not improve:
        return optimize_trans(struct, best_cost, bounds, lat, best_trans, R, conf_angles, sg, n_atoms, cost_function, cost_factors, cost_options, conf_params=conf_params, step=step/2., ftol=ftol, xtol=xtol, thresh_type=thresh_type, verbose=verbose)
            
    # Optimize the translation vector
    while opt_cost["Tot"] < best_cost["Tot"]-1e-6:
    
        if verbose:
            print("  Optimized translation vector: {:.4f}, {:.4f}, {:.4f}".format(opt_trans[0], opt_trans[1], opt_trans[2]))
            print("  Cost before this step: {:.2f}, cost after: {:.2f}".format(best_cost["Tot"], opt_cost["Tot"]))
            
        best_trans = copy.deepcopy(opt_trans)
        best_cost = copy.deepcopy(opt_cost)
        
        # Update grid of translation vectors
        trans_grid[0,:,:,0] = best_trans[0] - dx
        trans_grid[1,:,:,0] = best_trans[0]
        trans_grid[2,:,:,0] = best_trans[0] + dx
        trans_grid[:,0,:,1] = best_trans[1] - dx
        trans_grid[:,1,:,1] = best_trans[1]
        trans_grid[:,2,:,1] = best_trans[1] + dx
        trans_grid[:,:,0,2] = best_trans[2] - dx
        trans_grid[:,:,1,2] = best_trans[2]
        trans_grid[:,:,2,2] = best_trans[2] + dx
        
        # Normalize the translation vectors to be within [0, 1[
        trans_grid[np.where(trans_grid >= 1.)] -= 1.
        trans_grid[np.where(trans_grid < 0.)] += 1.
        
        # Center grid of costs around the best cost
        for i in range(3):
            r = 1-opt_inds[i]
            cost_grid = roll_zeropad(cost_grid, r, axis=i)
        # Update the cost grid elements that are newly introduced
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if cost_grid[i, j, k] == 0.:
                        tmp_cryst, clash = cr.create_crystal(struct, lat, trans_grid[i, j, k], R, conf_angles, sg, n_atoms, conf_params=conf_params)
                        if clash:
                            tmp_cost = {}
                            tmp_cost["Tot"] = 1e12
                        else:
                            tmp_cost = compute_cost(tmp_cryst, cost_function, cost_factors, cost_options)
                        cost_grid[i, j, k] = tmp_cost["Tot"]
        
        # Get minimum cost and associated translation vector index
        min_cost = np.min(cost_grid)
        best_inds = np.where(np.isclose(cost_grid, min_cost, atol=1e-6, rtol=1e-12))
        num_1 = -1
        opt_inds = (1,1,1)
        for i in range(len(best_inds[0])):
            # if the best cost is in the central index, restart the optimization with a reduced step size
            if best_inds[0][i] == 1 and best_inds[1][i] == 1 and best_inds[2][i] == 1:
                print("  Minimum found, decreasing step size")
                return optimize_trans(struct, best_cost, bounds, lat, opt_trans, R, conf_angles, sg, n_atoms, cost_function, cost_factors, cost_options, conf_params=conf_params, step=step/2., ftol=ftol, xtol=xtol, thresh_type=thresh_type, verbose=verbose)
            # Get indices that are closest to the center as the central indices
            n = 0
            for j in range(3):
                if best_inds[j][i] == 1:
                    n += 1
            if n > num_1:
                num_1 = n
                opt_inds = (best_inds[0][i], best_inds[1][i], best_inds[2][i])
        
        opt_trans = copy.deepcopy(trans_grid[opt_inds])
        opt_cryst, clash = cr.create_crystal(struct, lat, opt_trans, R, conf_angles, sg, n_atoms, conf_params=conf_params)
        opt_cost = compute_cost(opt_cryst, cost_function, cost_factors, cost_options)
        
    return optimize_trans(struct, opt_cost, bounds, lat, opt_trans, R, conf_angles, sg, n_atoms, cost_function, cost_factors, cost_options, conf_params=conf_params, step=step/2., ftol=ftol, xtol=xtol, thresh_type=thresh_type, verbose=verbose)
    
    
    
def optimize_rot(struct, best_cost, bounds, lat, trans, R, conf_angles, sg, n_atoms, cost_function, cost_factors, cost_options, conf_params=None, step=1e-2, ftol=1e-2, xtol=1e-4, thresh_type="abs", verbose=False, N_max=10):
    """
    Optimize the cost function with respect to the rotation matrix.
    
    Inputs:     - struct                Input crystal structure
                - best_cost             Initial cost
                - bounds                Bounds for the parameter
                - lat                   Lattice parameters
                - trans                 Initial translation vector
                - R                     Rotation matrix
                - conf_angles           Conformational angles
                - sg                    Space group of the crystal
                - n_atoms               Number of atoms in the molecules
                - cost_function         Form of the cost function
                - cost_factors          Factors of each part of the cost function
                - cost_options          Options for the parts of the cost function
                - conf_params           Atoms and masks in conformer angles
                - step                  Step size (fraction of the range of parameter)
                - ftol                  Tolerance for convergence of the cost function
                - xtol                  Tolerance for convergence of the step size
                - thresh_type           Set to "abs" for absolute convergence of the cost function,
                                            "rel" for relative convergence
                
    Outputs:    - best_R                Optimized rotation matrix
                - best_cost             Optimized cost function
    """
    
    if verbose:
        print("  Optimizing rotation. Step size: {}".format(step))
    
    best_R = copy.deepcopy(R)
    converged = False
    N = 0
    
    if step < xtol:
        return best_R, best_cost
    
    # Generate initial rotation
    while not converged:
        N += 1
        # Generate random rotation
        r = cr.generate_random_unit_vector()
        r = np.append(r, step*bounds[1])
        opt_R = cr.rotation_matrix(r)
        opt_R = opt_R.dot(best_R)
        
        # Get associated cost function
        opt_cryst, clash = cr.create_crystal(struct, lat, trans, opt_R, conf_angles, sg, n_atoms, conf_params=conf_params)
        if clash:
            opt_cost = {}
            opt_cost["Tot"] = 1e12
        else:
            opt_cost = compute_cost(opt_cryst, cost_function, cost_factors, cost_options)
        
        
        # If the energy decreases, update the rotation matrix
        if opt_cost["Tot"] < best_cost["Tot"]:
            if verbose:
                print("  Decrease in the cost function found after {} tries ({:.2f} -> {:.2f})".format(N, best_cost["Tot"], opt_cost["Tot"]))
            best_cost = copy.deepcopy(opt_cost)
            best_R = copy.deepcopy(opt_R)
            N = 0
        
        # After N_max iterations without decrease of the cost function, reduce the step size
        if N > N_max:
            return optimize_rot(struct, best_cost, bounds, lat, trans, best_R, conf_angles, sg, n_atoms, cost_function, cost_factors, cost_options, conf_params=conf_params, step=step/2., ftol=ftol, xtol=xtol, thresh_type=thresh_type, verbose=verbose, N_max=N_max)
    
    return opt_R, opt_cost



def optimize_conf(struct, i, best_cost, bounds, lat, trans, R, conf_angles, sg, n_atoms, cost_function, cost_factors, cost_options, conf_params=None, step=1e-2, ftol=1e-2, xtol=1e-4, thresh_type="abs", verbose=False):
    """
    Optimize the cost function with respect to all lattice parameters, iteratively.

    Inputs:     - struct                Input crystal structure
                - i                     Index of the conformer angle to change
                - best_cost             Initial cost
                - bounds                Bounds for the parameter
                - lat                   Initial lattice parameters
                - trans                 Translation vector
                - R                     Rotation matrix
                - conf_angles           Conformational angles
                - sg                    Space group of the crystal
                - n_atoms               Number of atoms in the molecules
                - cost_function         Form of the cost function
                - cost_factors          Factors of each part of the cost function
                - cost_options          Options for the parts of the cost function
                - conf_params           Atoms and masks in conformer angles
                - step                  Step size (fraction of the range of parameter)
                - ftol                  Tolerance for convergence of the cost function
                - xtol                  Tolerance for convergence of the step size
                - thresh_type           Set to "abs" for absolute convergence of the cost function,
                                            "rel" for relative convergence
                
    Outputs:    - best_conf             Optimized conformational angles
                - best_cost             Optimized cost function
    """

    best_conf = copy.deepcopy(conf_angles)

    if verbose:
        print("  Optimizing conformer angle {}/{}. Initial value {}, step size {}".format(i+1, len(conf_angles), conf_angles[i], step))

    # If the step size is too small, return the original conformer angle
    if step < xtol:
        return best_conf, best_cost

    # Obtain step size
    span = bounds[1]-bounds[0]
    dx = span * step

    # Generate steps in positive and negative directions
    tmp_conf_p = copy.copy(best_conf)
    tmp_conf_p[i] += dx
    tmp_conf_m = copy.copy(best_conf)
    tmp_conf_m[i] -= dx

    # Restrict parameters to be within the bounds
    if tmp_conf_p[i] > bounds[1]:
        tmp_conf_p[i] -= bounds[1]
    if tmp_conf_p[i] < bounds[0]:
        tmp_conf_p[i] += bounds[0]
    if tmp_conf_m[i] > bounds[1]:
        tmp_conf_m[i] -= bounds[1]
    if tmp_conf_m[i] < bounds[0]:
        tmp_conf_m[i] += bounds[0]

    # Generate modified crystals
    cryst_p, clash_p = cr.create_crystal(struct, lat, trans, R, tmp_conf_p, sg, n_atoms, conf_params=conf_params)
    cryst_m, clash_m = cr.create_crystal(struct, lat, trans, R, tmp_conf_m, sg, n_atoms, conf_params=conf_params)

    # Compute cost of the modified crystals
    if clash_p:
        tmp_cost_p["Tot"] = 1e12
    else:
        tmp_cost_p = compute_cost(cryst_p, cost_function, cost_factors, cost_options)
    if clash_m:
        tmp_cost_m["Tot"] = 1e12
    else:
        tmp_cost_m = compute_cost(cryst_m, cost_function, cost_factors, cost_options)

    # If no direction is found, decrease the step size by two and try again
    if tmp_cost_p["Tot"] >= best_cost["Tot"]-1e-6 and tmp_cost_m["Tot"] >= best_cost["Tot"]-1e-6:
        if verbose:
            print("  Minimum found, decreasing step size")
        return optimize_conf(struct, i, best_cost, bounds, lat, trans, R, conf_angles, sg, n_atoms, cost_function, cost_factors, cost_options, conf_params=conf_params, step=step/2., ftol=ftol, xtol=xtol, thresh_type=thresh_type, verbose=verbose)
        
    # If positive direction is found, increase the parameter until energy rises
    elif tmp_cost_p["Tot"] < best_cost["Tot"] and tmp_cost_p["Tot"] < tmp_cost_m["Tot"]:
        if verbose:
            print("  Increasing the parameter decreases the cost")
        # While the energy decreases
        while tmp_cost_p["Tot"] < best_cost["Tot"]:
            # Update the best parameter and associated cost
            best_cost = copy.copy(tmp_cost_p)
            best_conf = copy.copy(tmp_conf_p)
            # Generate a new parameter
            tmp_conf_p[i] += dx
            # Generate the corresponding crystal
            cryst_p, clash_p = cr.create_crystal(struct, lat, trans, R, tmp_conf_p, sg, n_atoms, conf_params=conf_params)
            # Compute cost
            if clash_p:
                tmp_cost_p["Tot"] = 1e12
            else:
                tmp_cost_p = compute_cost(cryst_p, cost_function, cost_factors, cost_options)
            if verbose:
                print("  Cost before this step: {:.2f}, cost after: {:.2f}".format(best_cost["Tot"], tmp_cost_p["Tot"]))
        # Restart the optimization with reduced step size
        return optimize_conf(struct, i, best_cost, bounds, lat, trans, R, best_conf, sg, n_atoms, cost_function, cost_factors, cost_options, conf_params=conf_params, step=step/2., ftol=ftol, xtol=xtol, thresh_type=thresh_type, verbose=verbose)
        
    # If negative direction is found, increase the parameter until energy rises
    elif tmp_cost_m["Tot"] < best_cost["Tot"] and tmp_cost_m["Tot"] < tmp_cost_p["Tot"]:
        if verbose:
            print("  Decreasing the parameter decreases the cost")
        # While the energy decreases
        while tmp_cost_m["Tot"] < best_cost["Tot"]:
            # Update the best parameter and associated cost
            best_cost = copy.copy(tmp_cost_m)
            best_conf = copy.copy(tmp_conf_m)
            # Generate a new parameter
            tmp_conf_m[i] -= dx
            # Generate the corresponding crystal
            cryst_m, clash_m = cr.create_crystal(struct, lat, trans, R, tmp_conf_m, sg, n_atoms, conf_params=conf_params)
            # Compute cost
            if clash_m:
                tmp_cost_m["Tot"] = 1e12
            else:
                tmp_cost_m = compute_cost(cryst_m, cost_function, cost_factors, cost_options)
            if verbose:
                print("  Cost before this step: {:.2f}, cost after: {:.2f}".format(best_cost["Tot"], tmp_cost_m["Tot"]))
        # Restart the optimization with reduced step size
        return optimize_conf(struct, i, best_cost, bounds, lat, trans, R, best_conf, sg, n_atoms, cost_function, cost_factors, cost_options, conf_params=conf_params, step=step/2., ftol=ftol, xtol=xtol, thresh_type=thresh_type, verbose=verbose)

    else:
        raise ValueError("Error during the optimization of conformer angle {}".format(i+1))
        return









def iterative_opt(struct, lat, trans, R, conf_angles, sg, n_atoms, parameter_set, cost_function, cost_factors, cost_options, cell_params_lims, conf_params=None, n_max=100, verbose=False, step=1e-2, ftol=1., xtol=1e-4, thresh_type="abs"):
    """
    Optimize the cost function with respect to all variable parameters.

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
                - conf_params           Atoms and masks in conformer angles
                - n_max                 Maximum number of optimization iterations
                - verbose               Verbosity level
                - step                  Step size (fraction of range of each parameter)
                - ftol                  Tolerance for convergence of the cost function
                - xtol                  Tolerance for convergence of the step size
                - thresh_type           Set to "abs" for absolute convergence of the cost function,
                                            "rel" for relative convergence
                
    Outputs:    - opt_crystal           Optimized crystal structure
                - opt_lat               Optimized lattice parameters
                - opt_trans             Optimized translation vector
                - opt_R                 Optimized rotation matrix
                - opt_conf              Optimized conformational angles
    """

    # Generate initial crystal
    init_cryst, _ = cr.create_crystal(struct, lat, trans, R, conf_angles, sg, n_atoms, conf_params=conf_params)
    # Get initial cost
    best_cost = compute_cost(init_cryst, cost_function, cost_factors, cost_options)
    converged = False
    
    opt_lat = copy.deepcopy(lat)
    opt_trans = copy.deepcopy(trans)
    opt_R = copy.deepcopy(R)
    opt_conf = copy.deepcopy(conf_angles)
    tmp_cost = copy.deepcopy(best_cost)
    
    k = 0
    while not converged:
        k += 1
        best_cost = copy.deepcopy(tmp_cost)
        # Optimize cell parameters
        for p in ["a", "b", "c", "alpha", "beta", "gamma"]:
            if p in ["a", "b", "c"]:
                bounds = cell_params_lims[0], cell_params_lims[1]
            elif p in ["alpha", "beta", "gamma"]:
                bounds = cell_params_lims[2], cell_params_lims[3]
            if p in parameter_set:
                opt_lat, tmp_cost = optimize_lat_param(struct, p, tmp_cost, bounds, opt_lat, opt_trans, opt_R, opt_conf, sg, n_atoms, cost_function, cost_factors, cost_options, conf_params=conf_params, step=step, ftol=ftol, xtol=xtol, thresh_type=thresh_type, verbose=verbose)
        
        # Optimize translation
        if "trans" in parameter_set:
            bounds = [0., 1.]
            opt_trans, tmp_cost = optimize_trans(struct, tmp_cost, bounds, opt_lat, opt_trans, opt_R, opt_conf, sg, n_atoms, cost_function, cost_factors, cost_options, conf_params=conf_params, step=step, ftol=ftol, xtol=xtol, thresh_type=thresh_type, verbose=verbose)
        
        # TODO: optimize rotation
        if "rot" in parameter_set:
            bounds = [0., 100.]
            opt_R, tmp_cost = optimize_rot(struct, tmp_cost, bounds, opt_lat, opt_trans, opt_R, opt_conf, sg, n_atoms, cost_function, cost_factors, cost_options, conf_params=conf_params, step=step, ftol=ftol, xtol=xtol, thresh_type=thresh_type, verbose=verbose)
        
        # TODO: optimize conformation
        if "conf" in parameter_set:
            bounds = [0., 360.]
            for i in range(len(conf_angles)):
                opt_conf, tmp_cost = optimize_conf(struct, i, tmp_cost, bounds, opt_lat, opt_trans, opt_R, opt_conf, sg, n_atoms, cost_function, cost_factors, cost_options, conf_params=conf_params, step=step, ftol=ftol, xtol=xtol, thresh_type=thresh_type, verbose=verbose)
        
        # Check for convergence (relative threshold)
        print("Cost before this optimization iteration: {:.2f}, cost after: {:.2f}".format(best_cost["Tot"], tmp_cost["Tot"]))
        if thresh_type == "rel":
            if np.abs((tmp_cost["Tot"]-best_cost["Tot"])/best_cost["Tot"]) < ftol:
                converged = True
                print("Converged!")
        # Check for convergence (absolute threshold)
        elif thresh_type == "abs":
            if np.abs(tmp_cost["Tot"]-best_cost["Tot"]) < ftol:
                converged = True
                print("Converged!")
        
        if not converged and k >= n_max:
            converged = True
            print("Max number of optimization runs reached!")

    # Generate optimized crystal
    opt_crystal, _ = cr.create_crystal(struct, opt_lat, opt_trans, opt_R, opt_conf, sg, n_atoms, conf_params=conf_params)

    return opt_crystal, opt_lat, opt_trans, opt_R, opt_conf
