
#--------------------------------------------------------------------------------
# IMPORT LIBRARIES
#--------------------------------------------------------------------------------

import numpy as np
import ase
from ase.io import read,write
import sys,os
from copy import copy
import quippy as qp
sys.path.insert(0,'/Users/manuelcordova/Desktop/Work/ml_tools/')
from ml_tools.descriptors.quippy_interface import RawSoapQUIP
from ml_tools.models.KRR import KRR,KRR_PP
from ml_tools.kernels.kernels import KernelPower,KernelPP
from ml_tools.utils import get_rmse
from ml_tools.compressor.idx import IDXFilter
from scipy.linalg import eigh
from scipy.linalg import eigvalsh as evs

#--------------------------------------------------------------------------------
# FUNCTION DEFINITIONS
#--------------------------------------------------------------------------------

def pca(X,Threshold=0.99,space=False):
    """  Calculates the PCA projections of set X onto itself, returns D eigenvectors if needed for later embedding """
    mX = X - X.mean(axis=0) # remove the means of each feature
    cov = np.cov(mX.T)# Has to contain the covariance matrix between the feature. We will diagonalise it.
    e,v = eigh(cov)
    e = e[::-1]
    v = np.fliplr(v)
    esum= e/e.sum()
    var = [esum[0]]
    
    for ee in esum[1:]:
        var.append(ee+var[-1])
    var = np.array(var)
    until = np.where(var<Threshold)[0][-1]
    
    print("You have reduced the initial space of ", X.shape[1], " dimensions to ", until)
    if space:
        return np.dot(mX,v[:,:until]), v[:,:until]
    else:
        return np.dot(mX,v[:,:until])

def do_fps(x, d=0,robust=False):
    """This will target a number of FPS that reaches a density comparable with the average
    minimum distance in the distribution of X """
    if robust == False:
        if d == 0 : d = len(x)
        n = len(x)
        iy = np.zeros(d, int)
        # faster evaluation of Euclidean distance
        n2 = np.sum(x**2,axis=1)
        iy[0] = 0
        dl = n2 + n2[iy[0]] - 2* np.dot(x, x[iy[0]])
        dss = []
        for i in range(1,d):
            iy[i] = np.argmax(dl)
            nd = n2 + n2[iy[i]] - 2*np.dot(x,x[iy[i]])
            dl = np.minimum(dl, nd)
            dss.append(max(dl))
    else: 
        # First, we take a good model estimate 
        print("Will implement something smarter")
    return iy,dss

def bootstrap_krr_train(ktrain,ytrain,delta,jitter=1e-8,nmodels=10):
    # ktrain :
    # ytrain :
    # nmodels : number of random KRR models to build for error estimation
    # delta : this is the sigma_n provided in the shiftML paper
    # jitter : jitter for Cholesky decomposition
    ntrain = len(ytrain) # number of training configurations
    itrain = np.arange(ntrain) # indices of training configurations
    
    nrs = ntrain/2 # 2-fold cross validation split
    
    ypred  = np.zeros((nmodels, ntrain)) 
    krr = []
    irs = np.zeros((nmodels, nrs), int)
    omegas = np.zeros((nmodels, ntrain))
    
    #evaluation of optimal jitter
    #sigma = np.std(ytrain)/np.trace(ktrain)*ktrain.shape[0]*delta
    sigma = 1.0/np.sqrt(np.trace(ktrain)/ktrain.shape[0])/delta
    k = np.eye(ktrain.shape[0])*sigma**2 + ktrain
    evals = evs(k)
    if np.min(evals) < 0.0:
        jitter = 2*np.abs(np.min(evals))
    print("jitter ",jitter)
    del k
    
    training_model_indices = []
    for j in range(nmodels):
        print(j)
        # pick ntrs CV training points from among the full training set
        irs[j] = np.random.choice(itrain, nrs, replace=False)
        invirs = np.setdiff1d(itrain, irs[j])
        training_model_indices.append(irs[j])
        training_model_indices.append(invirs)
        #First half model using the irs indices 
        krr.append(KRR(jitter,delta))
        # Perform KRR
        krr[-1].fit(ktrain[irs[j],:][:,irs[j]],ytrain[irs[j]])
        # predict on invirs training structures
        ypred[j,invirs] = krr[-1].predict(ktrain[:,irs[j]])[invirs]
        omegas[j,irs[j]] = krr[-1].alpha[0]
        
        #Second half model using the complementary irs indices (invirs)
        krr.append(KRR(jitter,delta))
        # Perform KRR
        krr[-1].fit(ktrain[invirs,:][:,invirs],ytrain[invirs])
        # predict on irs training structures
        ypred[j,irs[j]] = krr[-1].predict(ktrain[:,invirs])[irs[j]]
        omegas[j,invirs] = krr[-1].alpha[0]

    ybest = np.mean(ypred,axis=0)
    yerr = np.std(ypred,axis=0)
    
    alpha = np.sqrt(np.mean((ybest - ytrain)**2/yerr**2))
    
    # return the set of nmodels KRR models, the subsampling correction alpha, and the
    # final predictions for the training set allowing an error estimate for the training
    # configurations for outlier detection
    return krr,alpha,ybest,yerr*alpha,np.array(training_model_indices),omegas
    
def bootstrap_krr_predict(ktest,krr,alpha,irs):
    # ktest : rect kernel between training and test configurations
    # krr : set of KRR models from bootstrap_krr_train()
    # alpha : subsampling correction alpha from bootstrap_krr_train()
    #yref_sp
    nmodels = len(krr)
    ntest = len(ktest) # EAE might need the transpose here!!!
    ypred = np.zeros((nmodels, ntest)) 
    # for each KRR model
    for j in range(nmodels):
        # predict on all training structures
        ypred[j] = krr[j].predict(ktest[:,irs[j]])
            
    # final prediction before correction
    ypred_final = ypred * 0.0 + np.mean(ypred,axis=0)
    # if all models agree no correction is made, if they disagree a correction is made 
    ypred_final += alpha * (ypred - np.mean(ypred,axis=0))
    
    # return final predictions for the test set and corresponding error
    return ypred_final

def get_index_of_frame(db,sp,suspicious_centers):
    
    i_list = {}
    abs_idx = 0
    blacklist = []
    frm_list = {}
    for i_f,d in enumerate(db):
        n_of_sp = len(np.where(d.numbers==sp)[0])
        i_list[i_f] = np.arange(abs_idx,abs_idx+n_of_sp)
        for n in  np.arange(abs_idx,abs_idx+n_of_sp,1):
            frm_list[n] = i_f
        abs_idx += n_of_sp
        
    for susp in suspicious_centers:
        if susp not in blacklist:
            frame_containing_susp = frm_list[susp]
            for ll in i_list[frame_containing_susp]:
                blacklist.append(ll)
        else:
            next
    return list(set(blacklist))


def best_sigma_cv(kmm,kmn,y,delta,ddelta,jitter):
    """ Finds the best value of sigma such that
    regulariser = np.std(y_filt)/delta * np.sqrt(jitter)
    regularises well our problem.
    """
    krr_dummy,alpha_dummy,ypred,yerr,indices_dummy,omega_dummy = bootstrap_krr_train_PP(kmm,kmn,y,delta,jitter,nmodels)
    score = get_rmse(y,ypred)
    bestscore = score
    currdelta = delta
    bestdelta = delta
    counter = 0
    while True:
        currdelta *= ddelta
        # Train the sparsified model
        krr_dummy,alpha_dummy,ypred,yerr,indices_dummy,omega_dummy = bootstrap_krr_train_PP(kmm,kmn,y,currdelta,jitter,nmodels)
        score = get_rmse(y,ypred)
        if (score > bestscore): break
        counter += 1
        bestscore = score
        bestdelta = currdelta

    if (counter == 0):
        currdelta = delta
        while True:
            currdelta /= ddelta
            # Train the sparsified model
            krr_dummy,alpha_dummy,ypred,yerr,indices_dummy,omega_dummy = bootstrap_krr_train_PP(kmm,kmn,y,currdelta,jitter,nmodels)
            score = get_rmse(y,ypred)
            if (score > bestscore): break
            bestscore = score
            bestdelta = currdelta

    print("Best result we get is :", bestscore," with a sigma of :", bestdelta)
    return bestdelta,bestscore



def bootstrap_krr_train_PP(kmm,kmn,ytrain,delta,jitter=1e-8,nmodels=10):
    """Prepares the resampling scheme for building an uncertainty estimate 
    enriched KRR regression model using projected processes.
    Kmm : Active kernel
    Kmn : Passive kernel
    ytrain : All properties
    delta  : intrinsic regularisation
    jitter : numerical stabiliser
    nmodels: number of resampling routines to be used for uncertainty convergence
    """
    ntrain = len(ytrain) # number of training configurations
    itrain = np.arange(ntrain) # indices of training configurations
    
    nrs = ntrain/2 # 2-fold cross validation split
    print(nrs)
    ypred  = np.zeros((nmodels, ntrain)) 
    krr = []
    irs = np.zeros((nmodels, nrs), int)
    omegas = np.zeros((2*nmodels, kmm.shape[0]))
    
    #evaluation of optimal jitter
    sigma = 1.0/np.sqrt(np.trace(kmm)/kmm.shape[0])/delta
    k = np.eye(kmm.shape[0])*sigma**2*kmm + np.dot(kmn,kmn.T)
    evals = evs(k)
    if np.min(evals) < 0.0:
        jitter = 2*np.abs(np.min(evals))
    print("jitter ",jitter)
    del k
    
    training_model_indices = []
    for j in range(nmodels):
        print(j)
        # pick ntrs CV training points from among the full training set
        irs[j] = np.random.choice(itrain, nrs, replace=False)
        invirs = np.setdiff1d(itrain, irs[j])
        training_model_indices.append(irs[j])
        training_model_indices.append(invirs)
        
        #First half model using the irs indices 
        krr.append(KRR_PP(jitter,delta))
        # Perform KRR
        krr[-1].fit(kmm,kmn[:,irs[j]],ytrain[irs[j]])
        # predict on invirs training structures
        ypred[j,invirs] = krr[-1].predict(kmn.T)[invirs]
        #omegas[j,irs[j]] = krr[-1].alpha[0]
        omegas[2*j] = krr[-1].alpha[0]
        #Second half model using the complementary irs indices (invirs)
        krr.append(KRR_PP(jitter,delta))
        # Perform KRR
        krr[-1].fit(kmm,kmn[:,invirs],ytrain[invirs])
        # predict on irs training structures
        ypred[j,irs[j]] = krr[-1].predict(kmn.T)[irs[j]]
        #omegas[j,invirs] = krr[-1].alpha[0]
        omegas[2*j+1] = krr[-1].alpha[0]

    ybest = np.mean(ypred,axis=0)
    yerr = np.std(ypred,axis=0)
    
    alpha = np.sqrt(np.mean((ybest - ytrain)**2/yerr**2))
    
    # return the set of nmodels KRR models, the subsampling correction alpha, and the
    # final predictions for the training set allowing an error estimate for the training
    # configurations for outlier detection
    return krr,alpha,ybest,yerr*alpha,np.array(training_model_indices),omegas

def bootstrap_krr_predict_PP(ktest,krr,alpha,irs):
    # ktest : rect kernel between training and test configurations
    # krr : set of KRR models from bootstrap_krr_train()
    # alpha : subsampling correction alpha from bootstrap_krr_train()
    #yref_sp
    nmodels = len(krr)
    ntest = len(ktest) # EAE might need the transpose here!!!
    ypred = np.zeros((nmodels, ntest)) 
    # for each KRR model
    for j in range(nmodels):
        # predict on all training structures
        ypred[j] = krr[j].predict(ktest)
            
    # final prediction before correction
    ypred_final = ypred * 0.0 + np.mean(ypred,axis=0)
    # if all models agree no correction is made, if they disagree a correction is made 
    ypred_final += alpha * (ypred - np.mean(ypred,axis=0))
    
    # return final predictions for the test set and corresponding error
    return ypred_final
