##
##

import autograd
import numpy as np
import copy
import ase
from ase.io import read,write
from ase.visualize import view
import sys,os
from glob import glob
from copy import copy
from tqdm import tqdm_notebook
import cPickle as pck

import scipy.optimize as op

cluster = False
if cluster:
    root = "/home"
else:
    root = "/Users"


sys.path.insert(0,root+'/balodis/gap/ml_tools/')
sys.path.insert(0,root+'/balodis/gap/sML/modules/')
from krr_modules import *
from ml_tools.utils import get_mae,get_rmse,get_sup,get_spearman,get_score
import json



def load_kernel(sp=1):
    # Loading the models
    print "Loading the model for : ", sp

    kernelDir = root+'/balodis/gap/data/'

    with open(kernelDir + 'sparse_{}_20000_model_numbers.json'.format(sp), 'r') as f:
        model_numbers = json.load(f)
    # Load up FPS selection
    trainsoaps = np.load(kernelDir + 'sparse_{}_20000_soap_vectors.npy'.format(sp))[:20000]
    krrs = np.load(kernelDir + 'sparse_{}_20000_model.npy'.format(sp))
    zeta = model_numbers['hypers']['zeta']
    global_species = [1, 6, 7, 8, 16]
    nocenters = copy(global_species)
    nocenters.remove(sp)
    soap_params = model_numbers['soaps']
    representation = RawSoapQUIP(**soap_params)

    print "The model is ready"

    return krrs, representation, trainsoaps, model_numbers, zeta


def predict_shifts(frames_test, krrs, representation, trainsoaps, model_numbers, zeta, sp=1):
    # Loading test properties
    rawsoaps = representation.transform(frames_test)
    # print "Loaded SOAPS for {}".format(z2symb[sp])

    x_test = rawsoaps[:, np.array(model_numbers['space'])]
    x_test /= np.linalg.norm(x_test, axis=1)[:, np.newaxis]
    Ktest = np.dot(x_test, trainsoaps.T) ** zeta

    y_pred = bootstrap_krr_predict_PP(Ktest, krrs, model_numbers['alpha_bs'], model_numbers['indices_bs']) + \
             model_numbers['y_offset']
    y_best = y_pred.mean(axis=0)
    # y_err  = y_pred.std(axis=0)
    return y_best


# here we average the methyls (and count them 1 time), the CH2 protons are assigned as best match in the individual groups
def exp_rmsd(y_calc,molecule="cocaine"):

    if molecule=="cocaine":
        y_exp = [3.76, 3.78, 5.63,
                 3.49, 3.06,
                 3.32,
                 2.91, 3.38,
                 2.56, 2.12,
                 8.01, 8.01, 8.01, 8.01, 8.01,
                 3.78, 1.04]

        # label_exp=[1,2,3,4,5,6,7,8,9,10,[11,12,13],14,15,16,17,18,[19,20,21]]
        # label_calc=[1,2,3,4,[5,6],[7,8],[9,10],[11,12,13],[14,15,16,17,18],[19,20,21]]

        y_calc = np.array(y_calc[0:len(y_calc) / 2])
        # y_calc=np.array(y_calc[np.arange(0,len(y_calc),len(y_calc)/21)])

        y_sort = []
        for k1 in range(3):
            y_sort.append(y_calc[k1])

        y_sel = y_calc[3:5]
        y_sort.append(np.min(y_sel))
        y_sort.append(np.max(y_sel))

        y_sort.append(y_calc[5])

        y_sel = y_calc[6:8]
        y_sort.append(np.max(y_sel))
        y_sort.append(np.min(y_sel))

        y_sel = y_calc[8:10]
        y_sort.append(np.min(y_sel))
        y_sort.append(np.max(y_sel))

        for k1 in range(10, 15):
            # y_sort.append(np.mean(y_calc[13:18]))
            y_sort.append(y_calc[k1])

        y_sel = y_calc[15:18]
        for k1 in range(1):
            y_sort.append(np.mean(y_sel))

        y_sel = y_calc[18:21]
        for k1 in range(1):
            y_sort.append(np.mean(y_sel))


    if molecule == "azd":
        y_exp = [6.92,
                 8.69,
                 9.01,
                 8.47,
                 15.37,
                 7.73,
                 9.64,
                 2.90,
                 1.78,
                 1.88,
                 1.88,
                 1.8,
                 1.6,
                 0.44,
                 1.54,
                 1.88,
                 1.88,
                 0.8,
                 0.8,
                 1.0,
                 1.74,
                 1.74,
                 0.73,
                 0.73,
                 0.73]

        y_calc = np.array(y_calc[0:len(y_calc) / 2])

        y_sort = []
        for k1 in range(22):
            y_sort.append(y_calc[k1])
        y_sel = y_calc[22:25]
        for k1 in range(1):
            y_sort.append(np.mean(y_sel))
        y_sel = y_calc[25:28]
        for k1 in range(1):
            y_sort.append(np.mean(y_sel))
        y_sel = y_calc[28:31]
        for k1 in range(1):
            y_sort.append(np.mean(y_sel))

    if molecule == "azd":
        y_exp = [6.92,
                 8.69,
                 9.01,
                 8.47,
                 15.37,
                 7.73,
                 9.64,
                 2.90,
                 1.78,
                 1.88,
                 1.88,
                 1.8,
                 1.6,
                 0.44,
                 1.54,
                 1.88,
                 1.88,
                 0.8,
                 0.8,
                 1.0,
                 1.74,
                 1.74,
                 0.73,
                 0.73,
                 0.73]

        y_calc = np.array(y_calc[0:len(y_calc) / 2])

        y_sort = []
        for k1 in range(22):
            y_sort.append(y_calc[k1])
        y_sel = y_calc[22:25]
        for k1 in range(1):
            y_sort.append(np.mean(y_sel))
        y_sel = y_calc[25:28]
        for k1 in range(1):
            y_sort.append(np.mean(y_sel))
        y_sel = y_calc[28:31]
        for k1 in range(1):
            y_sort.append(np.mean(y_sel))

    if molecule == "ritonavir":

        # Only protons attached to carbons, ch2 groups averaged, aromatic groups averaged
        ar_av = np.mean([y_calc[29], y_calc[30], y_calc[31], y_calc[32], y_calc[46],
                         y_calc[0], y_calc[1], y_calc[2], y_calc[10], y_calc[25]])

        y_sort = [np.mean([y_calc[7], y_calc[8], y_calc[9]]), y_calc[34], np.mean([y_calc[35], y_calc[36], y_calc[37]]),
                  y_calc[33], np.mean([y_calc[40], y_calc[41], y_calc[42]]),
                  np.mean([y_calc[38], y_calc[39]]), np.mean([y_calc[38], y_calc[39]]),
                  y_calc[18], y_calc[14], np.mean([y_calc[15], y_calc[16], y_calc[17]]),
                  np.mean([y_calc[11], y_calc[12], y_calc[13]]), y_calc[26], np.mean([y_calc[27], y_calc[28]]),
                  np.mean([y_calc[27], y_calc[28]]),
                  ar_av, ar_av, ar_av, ar_av, ar_av, np.mean([y_calc[19], y_calc[20]]),
                  np.mean([y_calc[19], y_calc[20]]),
                  y_calc[21], y_calc[22], np.mean([y_calc[23], y_calc[24]]), np.mean([y_calc[23], y_calc[24]]),
                  ar_av, ar_av, ar_av, ar_av, ar_av,
                  np.mean([y_calc[3], y_calc[4]]), np.mean([y_calc[3], y_calc[4]]), y_calc[5], y_calc[6]]

        # Experimental shifts ordered, but not calibrated correclty
        y_exp = [2.05, 3.22, 1.79,
                 4.22, 2.73,
                 np.mean([3.3, 4.53]), np.mean([3.3, 4.53]),
                 3.44, 2.40, 1.76,
                 1.00, 3.86, 2.82, 2.82,
                 5.21, 5.21, 5.21, 5.21, 5.21, np.mean([2.08, 2.85]), np.mean([2.08, 2.85]),
                 3.61, 3.88, np.mean([2.07, 3.10]), np.mean([2.07, 3.10]),
                 5.21, 5.21, 5.21, 5.21, 5.21,
                 3.7, 3.7, 5.52, 6.04]

    return np.array(y_sort), np.array(y_exp)


# def lin(x,a):
#     return a-x

def lin(x,a,b):
    return a-b*x

def rmsd(y_sort,y_exp):
    popt, pcov = op.curve_fit(lin,y_sort,y_exp)
    #rmsd = np.sqrt( sum( (y_exp-lin(y_sort,*popt))**2 )/len(y_sort))
    #rmsd = np.sqrt( sum( (y_exp-lin(y_sort,30.36))**2 )/len(y_sort))
    rmsd = np.sqrt(sum((y_exp - lin(y_sort,33.17,1.88)) ** 2) / len(y_sort))
    return rmsd

krr, representation, trainsoaps, model_numbers, zeta = load_kernel(1)

# xyz = read("/Users/balodis/work/ritonavir/results/test/1_loops_0.005_factor__H1_True_C13_False_test_1/1_0_init_structure.cif")
# y_pred = predict_shifts([xyz], krr, representation, trainsoaps, model_numbers, zeta, sp=1)
# y_sort, y_exp = exp_rmsd(y_pred,molecule='ritonavir')
# print y_sort
# print y_exp
# rms = rmsd(y_exp,y_sort)
# print rms

def exp_rmsd_13C(molecule="cocaine"):

    if molecule == "cocaine":
        y_exp = [65.95,
                 50.16,
                 66.70,
                 36.66,
                 62.63,
                 25.62,
                 25.62,
                 165.94,
                 129.37,
                 131.50,
                 133.50,
                 134.53,
                 133.50,
                 131.50,
                 172.18,
                 50.16,
                 41.52]
    return y_exp

##Testing
#Test
# krr, representation, trainsoaps, model_numbers, zeta = load_kernel(1)
#
# xyz = read("/Users/balodis/work/cocaine/structure_files/COCAIN10_H_relaxed_out_cell.pdb")
# y_pred = predict_shifts([xyz], krr, representation, trainsoaps, model_numbers, zeta, sp=1)
# y_sort, y_exp = exp_rmsd(y_pred)
# #popt, pcov = op.curve_fit(lin,y_sort,y_exp)
# #rmsd = np.sqrt( sum( (y_exp-lin(y_sort,*popt))**2 )/len(y_sort) )
# rms = rmsd(y_sort,y_exp)
# print rms

