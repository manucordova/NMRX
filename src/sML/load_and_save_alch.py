#!/usr/bin/python2 -u

import math
import numpy as np
import quippy as qp
import re
import argparse
import sys
import gc
from numba import njit, prange
from scipy import sparse as sp
from string import Template

##########################################################################################

def order_soap(soap, species, nspecies, nab, subspecies, nsubspecies, nsubab, \
               nmax, lmax, nn, u):

    p = np.zeros((nsubspecies, nsubspecies, nn))

    #translate the fingerprints from QUIP
    counter = 0
    p = np.zeros((nsubspecies, nsubspecies, nmax, nmax, lmax + 1))
    rs_index = [(i%nmax, (i - i%nmax)/nmax) for i in range(nmax*nsubspecies)]
    for i in range(nmax*nsubspecies):
        for j in range(i + 1):
            if i != j: mult = np.sqrt(0.5)
            else: mult = 1.0
            for k in range(lmax + 1):
                n1, s1 = rs_index[i]
                n2, s2 = rs_index[j]
                p[s1, s2, n1, n2, k] = soap[counter]*mult
                if s1 == s2: p[s1, s2, n2, n1, k] = soap[counter]*mult
                counter += 1
    for s1 in range(nsubspecies):
        for s2 in range(s1):
            p[s2, s1] = p[s1, s2].transpose((1, 0, 2))

    p = p.reshape((nsubspecies, nsubspecies, nn))
    p_full = np.zeros((nspecies, nspecies, nn))
    indices = [species.index(i) for i in subspecies]
    for i, j in enumerate(indices):
        for k, l in enumerate(indices):
            p_full[j, l] = p[i, k]
    p = p_full

    #we have an alchemical calculation
    if u is not None:
        nalch = u.shape[0]
        p = np.dot(u, np.dot(u, p))
        p = p.reshape((nalch, nalch, nn))

        p2 = np.zeros((nalch*(nalch + 1)/2, nn))
        k = 0
        for i in range(nalch):
            for j in range(i, nalch):
                if i == j: mult = 1.0
                else: mult = np.sqrt(2.0)
                p2[k] = p[i, j]*mult
                k += 1

    #we have a delta kernel calculation
    else:
        p2 = np.zeros((nab, nn))
        k = 0
        for i in range(nspecies):
            for j in range(i, nspecies):
                if i == j: mult = 1.0
                else: mult = np.sqrt(2.0)
                p2[k] = p[i, j]*mult
                k += 1

    return p2.ravel()

##########################################################################################

def indices(nframes, width=0, aod='asc'):

    if width == 0: width = nframes
    #would use nonlocal variables rather than a dictionary in python3
    d = {'k':0, 'l':0}
    n = nframes/width
    x = [i*width for i in range(n + 1)]
    if x[n] != nframes: 
        x += [nframes]
        n += 1

    #return ascending indices (top to bottom)
    if aod == 'asc':
        def inner():
            ind = range(x[d['k']], x[d['k'] + 1])
            d['k'] += 1 
            if d['k'] >= n: 
                d['l'] += 1
                d['k'] = d['l']
            return ind

    #return descending indices (right to left)
    else:
        def inner():
            ind = range(x[n - d['k'] + d['l']- 1], x[n - d['k'] + d['l']])
            d['k'] += 1 
            if d['k'] >= n: 
                d['l'] += 1
                d['k'] = d['l']
            return ind

    return n, inner

##########################################################################################

def get_soaps(soapstr, rc, species, nspecies, nmax, lmax, nn, nab, u, sparse, fastavg):

    #we have an alchemical calculation
    if u is not None : nalch = u.shape[0]

    def inner(frames):
        soaps_list = []
        n = len(frames)
        for i in range(n):

            frame = frames[i]
            frame.set_cutoff(rc)
            frame.calc_connect()
            subspecies = sorted(list(set([atom.number for atom in frame if atom.number in species])))
            nsubspecies = len(subspecies)
            nsubab = nsubspecies*(nsubspecies + 1)/2
            speciesstr = '{'+re.sub('[\[,\]]', '', str(subspecies))+'}'
            soapstr2 = soapstr.substitute(nspecies=nsubspecies, ncentres=nsubspecies, \
                                          species=speciesstr, centres=speciesstr)
            desc = qp.descriptors.Descriptor(soapstr2)
            soap = desc.calc(frame, grad=False)['descriptor']
            nenv = soap.shape[0]
            if u is not None: 
                soaps = np.zeros((nenv, nn*nalch*(nalch + 1)/2))
            else: 
                soaps = np.zeros((nenv, nn*nab))
            for j in range(nenv):
                soaps[j] = order_soap(soap[j], species, nspecies, nab, subspecies, \
                                      nsubspecies, nsubab, nmax, lmax, nn, u)
            if fastavg == True:
                if sparse == True: soaps_list.append(sp.lil_matrix(soaps.mean(axis=0)))
                else: soaps_list.append(soaps.mean(axis=0))
            else:
                if sparse == True: soaps_list.append(sp.csr_matrix(soaps))
                else: soaps_list.append(soaps)
        return soaps_list

    return inner

##########################################################################################

@njit(parallel=True)
def get_average(kernel, env_kernel, ids1, ids2, zeta):
    for ii in prange(len(ids1)-1):
        for jj in prange(len(ids2)-1):
            counter = 0
            for l1 in range(ids1[ii], ids1[ii+1]):
                for l2 in range(ids2[jj], ids2[jj+1]):
                    counter += 1
                    kernel[ii, jj] += math.pow(env_kernel[l1, l2], zeta)
            kernel[ii, jj] /= counter

##########################################################################################
##########################################################################################

def main(suffix, fxyz, rc, species, nmax, lmax, awidth, nframes, fu, sparse, width, \
         cutoff_dexp, cutoff_scale, fastavg, ntmax):

    suffix = str(suffix)
    if suffix != '': suffix = '_'+suffix
    fxyz = str(fxyz)
    cutoff = float(rc)
    species = sorted([int(species) for species in species.split()])
    nmax = int(nmax)
    lmax = int(lmax)
    awidth = float(awidth)
    nframes = int(nframes)
    fu = str(fu)
    sparse = bool(sparse)
    width = int(width)
    if nframes == 0: nframes = None
    nspecies = len(species)
    nn = nmax**2*(lmax + 1)
    nab = nspecies*(nspecies+1)/2
    cutoff_dexp = int(cutoff_dexp)
    cutoff_scale = float(cutoff_scale)
    fastavg = bool(fastavg)
    ntmax = int(ntmax)

    if fu != '':
        try:
            u = np.load(fu)
        except:
            print("Cannot find the u data")
            sys.exit()
    else:
        u = None

    frames = qp.AtomsList(fxyz, stop=nframes)
    nframes = len(frames)

    speciesstr = '{'+re.sub('[\[,\]]', '', str(species))+'}'

    soapstr = Template('average=F normalise=T soap cutoff_dexp=$cutoff_dexp \
                        cutoff_scale=$cutoff_scale central_reference_all_species=F \
                        central_weight=1.0 covariance_sigma0=0.0 atom_sigma=$awidth \
                        cutoff=$rc cutoff_transition_width=0.5 n_max=$nmax l_max=$lmax \
                        n_species=$nspecies species_Z=$species n_Z=$ncentres Z=$centres')
    soapstr = soapstr.safe_substitute(rc=rc, nmax=nmax, lmax=lmax, awidth=awidth, \
                                      cutoff_dexp=cutoff_dexp, cutoff_scale=cutoff_scale)
    soapstr = Template(soapstr)

    glob_kernel = np.zeros((3, nframes, nframes))

    #calculate the upper triangle of the global kernel in n*(n + 1)/2 blocks.
    #work from top to bottom and right to left so ind1 is asc and ind2 desc.
    #for n = 3 this calculates 1/3 of the SOAP vectors twice, the rest once.
    #RAM permitting, n should be minimised (e.g. n = 1, width = nframes) for speed
    #since the efficiency scales E(n) = 1 - (n - 1)*(n - 2)/(n*(n + 1))
    #e.g. E(n = 10) = 0.35 times slowdown compared to n = 1 (max efficiency) for
    #1/5 of the RAM usage.
    n, ind1 = indices(nframes, width=min(nframes, width), aod='asc')
    n, ind2 = indices(nframes, width=min(nframes, width), aod='desc')
    gsoaps = get_soaps(soapstr, rc, species, nspecies, nmax, lmax, nn, nab, u, sparse, fastavg)
    
    counter = 0
    for i in range(n):

        inds1 = ind1()
        #have we reached the end of the training set?
        if ntmax != 0 and inds1[0] >= ntmax: break
        #the second loop has already calculated the soaps we want
        if i > 0: soaps_list1 = soaps_list2
        #unless the second loop has not calculated anything yet 
        else: soaps_list1 = gsoaps([frames[l] for l in inds1])

        if fastavg == True:

            #diagonal
            if sparse == True:
                soaps1 = sp.vstack(soaps_list1).tocsr()
                glob_kernel[0, inds1[0]:inds1[-1]+1, inds1[0]:inds1[-1]+1] = soaps1.dot(soaps1.T).toarray()
            else:
                soaps1 = np.vstack(soaps_list1)
                glob_kernel[0, inds1[0]:inds1[-1]+1, inds1[0]:inds1[-1]+1] = np.dot(soaps1, soaps1.T)

            counter += 1
            print('Calculated block %i of %i' % (counter, n*(n + 1)/2))
            sys.stdout.flush()

            #off-diagonal
            for j in range(n - i - 1):
                inds2 = ind2()
                soaps_list2 = gsoaps([frames[l] for l in inds2])
                if sparse == True:
                    soaps1 = sp.vstack(soaps_list1).tocsr()
                    soaps2 = sp.vstack(soaps_list2).tocsr()
                    glob_kernel[0, inds1[0]:inds1[-1]+1, inds2[0]:inds2[-1]+1] = soaps1.dot(soaps2.T).toarray()
                else:
                    soaps1 = np.vstack(soaps_list1)
                    soaps2 = np.vstack(soaps_list2)
                    glob_kernel[0, inds1[0]:inds1[-1]+1, inds2[0]:inds2[-1]+1] = np.dot(soaps1, soaps2.T)

                counter += 1
                print('Calculated block %i of %i' % (counter, n*(n + 1)/2))
                sys.stdout.flush()

        else:

            #diagonal
            nenvs1 = [0] + [soaps_list1[k].shape[0] for k in range(len(soaps_list1))]
            nenvs1 = np.cumsum(nenvs1)

            if sparse == False: 
                sl_stacked1 = np.vstack(soaps_list1)
                del soaps_list1
                env_kernel = np.dot(sl_stacked1, sl_stacked1.T)
            else: 
                sl_stacked1 = sp.vstack(soaps_list1)
                del soaps_list1
                env_kernel = sl_stacked1.dot(sl_stacked1.T).toarray()

            for zeta in range(3):
                glob_block = np.zeros((len(nenvs1)-1, len(nenvs1)-1))
                get_average(glob_block, env_kernel, nenvs1, nenvs1, zeta+1)
                glob_kernel[zeta, inds1[0]:inds1[-1]+1, inds1[0]:inds1[-1]+1] = glob_block

            counter += 1
            print('Calculated block %i of %i' % (counter, n*(n + 1)/2))
            sys.stdout.flush()
            gc.collect()

            #off-diagonal
            for j in range(n - i - 1):

                inds2 = ind2()
                soaps_list2 = gsoaps([frames[l] for l in inds2])
                nenvs2 = [0] + [soaps_list2[k].shape[0] for k in range(len(soaps_list2))]
                nenvs2 = np.cumsum(nenvs2)

                if sparse == False: 
                    sl_stacked2 = np.vstack(soaps_list2)
                    env_kernel = np.dot(sl_stacked1, sl_stacked2.T)
                else: 
                    sl_stacked2 = sp.vstack(soaps_list2)
                    env_kernel = sl_stacked1.dot(sl_stacked2.T).toarray()

                for zeta in range(3):
                    glob_block = np.zeros((len(nenvs1)-1, len(nenvs2)-1))
                    get_average(glob_block, env_kernel, nenvs1, nenvs2, zeta+1)
                    glob_kernel[zeta, inds1[0]:inds1[-1]+1, inds2[0]:inds2[-1]+1] = glob_block[:, :]

                counter += 1
                print('Calculated block %i of %i' % (counter, n*(n + 1)/2))
                sys.stdout.flush()
                gc.collect()

        #cycle the second set of indices because we have avoided i == j
        inds2 = ind2()

    if fastavg == True:

        for i in range(nframes):
            for j in range(i, nframes):
                glob_kernel[0, j, i] = glob_kernel[0, i, j]

        for zeta in range(1):
            np.save('kernel'+suffix+'_zeta'+str(zeta+1)+'.npy', glob_kernel[zeta])

    else:

        for i in range(nframes):
            for j in range(i, nframes):
                glob_kernel[:, j, i] = glob_kernel[:, i, j]

        for zeta in range(3):
            np.save('kernel'+suffix+'_zeta'+str(zeta+1)+'.npy', glob_kernel[zeta])

##########################################################################################
##########################################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-fxyz', type=str, help='Location of xyz file')
    parser.add_argument('-species', type=str, help='List of elements e.g. "1 2 3"')
    parser.add_argument('--suffix', type=str, default='', help='Filename suffix')
    parser.add_argument('--rc', type=float, default=3.0, help='Cutoff radius')
    parser.add_argument('--nmax', type=int, default=9, help='Maximum radial label')
    parser.add_argument('--lmax', type=int, default=9, help='Maximum angular label')
    parser.add_argument('--awidth', type=float, default=0.3, help='Atom width')
    parser.add_argument('--nframes', type=int, default=0, help='Number of frames')
    parser.add_argument('--fu', type=str, default='', help='Location of u file')
    parser.add_argument('--sparse', type=int, default=0, help='1 for sparse arrays')
    parser.add_argument('--width', type=int, default=10**6, help='Width of kernel blocks')
    parser.add_argument('--cutoff_dexp', type=int, default=0, help='Witch\'s exponent')
    parser.add_argument('--cutoff_scale', type=float, default=1.0, help='Witch\'s scale')
    parser.add_argument('--fastavg', type=int, default=1, help='1 for fast average')
    parser.add_argument('--ntmax', type=int, default=0, help='Number of training structures')
    args = parser.parse_args()

    main(args.suffix, args.fxyz, args.rc, args.species, args.nmax, args.lmax, \
         args.awidth, args.nframes, args.fu, args.sparse, args.width, args.cutoff_dexp, \
         args.cutoff_scale, args.fastavg, args.ntmax)
