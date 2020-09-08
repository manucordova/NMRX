#####
# Last modified 29.06.2020
# Re-writing the script so that it is more general
# Authors: Manuel Cordova, Martins Balodis
#####



# Import libraries
import numpy as np
import os
import sys
import json
import itertools as it

# Import ShiftML libraries
sys.path.insert(0, os.path.abspath("../src"))
from sML.krr_modules import *
from ml_tools.utils import get_mae,get_rmse,get_sup,get_spearman,get_score

# Available nuclei for ShiftML
symb2z = {"H":1, "C":6, "N":7, "O":8}


def load_kernel(sp=1):
    """
    Load ShiftML kernel

    Inputs:     - sp                Species (atomic number of nucleus to load)

    Outputs:    - krrs              [?]
                - representation    [?]
                - trainsoaps        [?]
                - model_numbers     [?]
                - zeta              [?]
    """
    # Loading the models
    print("Loading the model for : {}".format(sp))

    kernelDir = os.path.abspath("../src/ShiftML_kernels/") + "/"

    with open(kernelDir + 'sparse_{}_20000_model_numbers.json'.format(sp), 'r') as f:
        model_numbers = json.load(f)
    # Load up FPS selection
    trainsoaps = np.load(kernelDir + 'sparse_{}_20000_soap_vectors.npy'.format(sp))[:20000]
    krrs = np.load(kernelDir + 'sparse_{}_20000_model.npy'.format(sp), allow_pickle=True, encoding="latin1")
    zeta = model_numbers['hypers']['zeta']
    global_species = [1, 6, 7, 8, 16]
    nocenters = copy(global_species)
    nocenters.remove(sp)
    soap_params = model_numbers['soaps']
    representation = RawSoapQUIP(**soap_params)

    print("The model is ready")

    return krrs, representation, trainsoaps, model_numbers, zeta



def predict_shifts(frames_test, krrs, representation, trainsoaps, model_numbers, zeta, sp=1):
    # Loading test properties
    
    ### THIS LINE FUCKS UP THE MEMORY USAGE
    rawsoaps = representation.transform(frames_test)
    ###
    #rawsoaps = np.load("tmp.npy")
    
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss/1024/1024)
    
    #print("Loaded SOAPS")

    x_test = rawsoaps[:, np.array(model_numbers['space'])]

    x_test /= np.linalg.norm(x_test, axis=1)[:, np.newaxis]
    Ktest = np.dot(x_test, trainsoaps.T) ** zeta

    y_pred = bootstrap_krr_predict_PP(Ktest, krrs, model_numbers['alpha_bs'], model_numbers['indices_bs']) + model_numbers['y_offset']

    y_best = y_pred.mean(axis=0)
    y_err  = y_pred.std(axis=0)

    return y_best, y_err
    
    
    
def best_fit_given_params(exp, shields, equivs, ambiguous, a, b):
    """
    Best fit experimental and computed chemical shifts
    
    Inputs:     - exp           Experimental shifts
                - shields       Computed shieldings
                - equivs        Equivalent nuclei
                - ambiguous     Ambiguous nuclei
                - a             Slope for the conversion from shielding to shift
                - b             Offset for the conversion from shielding to shift
    """
    
    # Initialize arrays of computed shifts, shieldings, errors, and shift RMSE
    comp_shifts = np.zeros(len(exp))
    discarded = []
    final_shields = np.zeros(len(exp))
    shields_errs = np.zeros(len(exp))
    rmse = np.inf
    # For all shifts
    for i, (shift, shield) in enumerate(zip(exp, shields)):
        # If the shift is already written: do not overwrite it
        if i not in discarded:
            unique = True
            # Managing equivalent nuclei
            for equiv in equivs:
                if i in equiv:
                    unique = False
                    # For all equivalent nuclei: consider the average shift
                    for j in equiv:
                        final_shields[j] = np.mean([shields[k] for k in equiv])
                        shields_errs[j] = np.std([shields[k] for k in equiv])/np.sqrt(float(len(equiv)))
                        discarded.append(j)
            # For unique nuclei: append the shielding
            if unique:
                final_shields[i] = shield
                shields_errs[i] = 0.
                discarded.append(i)
    
    # Obtain temporary RMSE (without considering ambiguities yet)
    best_rmse = np.sqrt(np.mean(np.square([exp[k] - a*final_shields[k] - b for k in range(len(exp))])))
    
    # Optimize shift RMSE with respect to ambiguous assignment
    change = True
    while change:
        change = False
        for unsure in ambiguous:
            tmp_shields = np.copy(final_shields)
            # Permute ambiguous assignment
            for perm in it.permutations(unsure):
                for i in range(len(perm)):
                    tmp_shields[unsure[i]] = shields[perm[i]]
                tmp_rmse = np.sqrt(np.mean(np.square([exp[k] - a*tmp_shields[k] - b for k in range(len(exp))])))
                if tmp_rmse < best_rmse:
                    final_shields = np.copy(tmp_shields)
                    best_rmse = tmp_rmse
                    change = True
    comp_shifts = [a*final_shields[k] + b for k in range(len(exp))]
    shifts_errs = np.abs(a)*shields_errs
    return comp_shifts, shifts_errs, final_shields, best_rmse
    
    
    
def shift_rmsd(struct, e, options):
    """
    Compute chemical shift RMSD
    
    Inputs:     - struct        Crystal structure
                - e             Nucleus to consider
                - options       Options
                - exp_shifts    Experimental shifts
                
    Outputs:    - rmsd
    """
    
    N = len(options["exp_shifts"])
    # Predict shieldings
    
    preds, errs = predict_shifts([struct], options["krr"], options["rep"], options["tr"], options["m_num"], options["zeta"], sp=symb2z[e])
    
    # Convert to shifts
    _, _, _, rmse = best_fit_given_params(options["exp_shifts"], preds[:N], options["equivalent"], options["ambiguous"], options["slope"], options["offset"])
    
    # If the chemical shift RMSE is smaller than the cutoff value set, return zero
    if rmse < options["cutoff"]:
        return options["cutoff"]
    
    return rmse
