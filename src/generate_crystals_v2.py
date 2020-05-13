##
##

import autograd
import numpy as np
import copy
import random
import ase
from ase.io import read,write
from ase.visualize import view
from ase.build import minimize_rotation_and_translation
from ase.spacegroup import crystal
from ase.data import chemical_symbols
from ase.data import atomic_numbers, vdw_radii
import sys,os
from glob import glob
from tqdm import tqdm_notebook
import cPickle as pck

import scipy.optimize as op

import mc_functions as mc

def get_close_atoms(struct, cutoff=0.1):
    close_atoms = []
    d = struct.get_all_distances(mic=True)
    #close atoms -> sum of VdW radii - 0.5
    dup = np.nonzero(d < cutoff)
    for i, j in zip(dup[0], dup[1]):
        if j > i:
            close_atoms.append([i, j])
    return close_atoms

def rotate_molecule(structure, center, quat, atoms):
    
    # Quaternion rotation
    R = np.zeros((3,3))
    R[0,0] = quat[0]**2 + quat[1]**2 - quat[2]**2 - quat[3]**2
    R[0,1] = 2*(quat[1]*quat[2] - quat[0]*quat[3])
    R[0,2] = 2*(quat[1]*quat[3] + quat[0]*quat[2])
    R[1,0] = 2*(quat[1]*quat[2] + quat[0]*quat[3])
    R[1,1] = quat[0]**2 - quat[1]**2 + quat[2]**2 - quat[3]**2
    R[1,2] = 2*(quat[2]*quat[3] - quat[0]*quat[1])
    R[2,0] = 2*(quat[1]*quat[3] - quat[0]*quat[2])
    R[2,1] = 2*(quat[2]*quat[3] + quat[0]*quat[1])
    R[2,2] = quat[0]**2 - quat[1]**2 - quat[2]**2 + quat[3]**2
    
    xyz = structure.get_positions()
    
    for i in range(atoms):
        xyz[i] -= center
        xyz[i] = R.dot(xyz[i])
        xyz[i] += center
    
    structure.set_positions(xyz)
    
    return structure

def create_crystal(structure,molecule,sg,atoms,sites,lat,trans,quat,conf_angles, n_mol):
#def create_crystal(structure,molecule,sg,atoms,sites,lat,trans,rotation,conf_angles, n_mol):

    structure.translate(trans)

    mass = structure[0:atoms].get_center_of_mass()
    
    structure = rotate_molecule(structure, mass, quat, atoms)
    
#    structure.rotate(rotation[0], v='x', center=mass)
#    structure.rotate(rotation[1], v='y', center=mass)
#    structure.rotate(rotation[2], v='z', center=mass)

    structure = mc.change_conformation(structure,conf_angles[0],conf_angles[1],conf_angles[2],conf_angles[3],conf_angles[4],conf_angles[5],molecule)

    structure.set_cell(lat, scale_atoms=False)

    scaled = structure.get_scaled_positions(wrap=False)
    trial_crystal = crystal(symbols=sites, basis=scaled[:atoms], spacegroup=sg, cell=lat, symprec=0.001, pbc=True)
    trial_crystal = mc.rewrite_coordinate(trial_crystal, n_mol, molecule=molecule)


    return trial_crystal

def get_third_length_low(a, b, alpha, beta, gamma, Vmol):
    sqroot = 1+(2*np.cos(np.deg2rad(alpha))*np.cos(np.deg2rad(beta))*np.cos(np.deg2rad(gamma)))
    sqroot += -np.cos(np.deg2rad(alpha))**2-np.cos(np.deg2rad(beta))**2-np.cos(np.deg2rad(gamma))**2
    sqroot = np.sqrt(sqroot)
    return Vmol / (a*b*sqroot)

def get_projections(struct, alpha, beta, gamma, atoms, quat):
    unit_a = np.array([1,0,0])
    unit_b = np.zeros(3)
    unit_b[0] = np.cos(np.deg2rad(gamma))
    unit_b[1] = np.sin(np.deg2rad(gamma))
    unit_c = np.zeros(3)
    unit_c[0] = np.cos(np.deg2rad(beta))
    unit_c[1] = (np.cos(np.deg2rad(alpha)) - np.cos(np.deg2rad(beta)) * np.cos(np.deg2rad(gamma))) / np.sin(np.deg2rad(gamma))
    unit_c[2] = np.sqrt(1. - unit_c[0]**2 - unit_c[1]**2)
    
    s = copy.deepcopy(struct)
    mass = s[0:atoms].get_center_of_mass()
    s = rotate_molecule(s, mass, quat, atoms)
    xyz = s.get_positions()[:atoms]
    
    proj_a = np.zeros(atoms)
    proj_b = np.zeros(atoms)
    proj_c = np.zeros(atoms)
    
    for i, pos in enumerate(xyz):
        proj_a[i] = pos.T.dot(unit_a)
        proj_b[i] = pos.T.dot(unit_b)
        proj_c[i] = pos.T.dot(unit_c)
    
    la = np.max(proj_a)-np.min(proj_a)
    lb = np.max(proj_b)-np.min(proj_b)
    lc = np.max(proj_c)-np.min(proj_c)
    
    return [la, lb, lc]



def check_for_overlap(trial_crystal, cut, close_atoms, Vmol, vol_high,old_overlap=False):
    """
    """
    if old_overlap==False:
        overlapping_atoms = get_close_atoms(trial_crystal, cutoff=cut)
    else:
        overlapping_atoms = ase.geometry.get_duplicate_atoms(trial_crystal, cutoff=cut, delete=False)

    count = 0
    for aa in overlapping_atoms:
        for bb in close_atoms:
            if aa[0] == bb[0] and aa[1] == bb[1]:
                count += 1
    condition_1 = (len(overlapping_atoms) == count)

    Vcell = trial_crystal.get_volume()
    condition_2 = (Vcell > Vmol and Vcell < vol_high * Vmol)

    return condition_1 and condition_2

Vs = []

def generate_crystal(starting, parameter_set, high_angle, low_angle, high_trans, low_trans, rotate_high, rotate_low, sg, atoms, n_mol, molecule, cut, close_atoms, vol_high,old_overlap=False,smart_cell = True):
    
    # Calculate the volume of the molecule
    nr = starting.get_atomic_numbers()[:atoms]
    rad = np.zeros(len(nr))
    for l1 in range(0, len(rad)):
        rad[l1] = 3.0 / 4.0 * np.pi * vdw_radii[nr[l1]] ** 3
    Vmol = sum(rad)*n_mol
    
    n_failed = 0
    
    while True:
        while True:
            n_failed += 1
            try:
                starting_copy = copy.deepcopy(starting)
                #Read in the structure with the right conformation and coordinates
                #starting = read(initial_structure)

                #Get the lattice parameters and sites from the reference structure
                lat = starting.get_cell_lengths_and_angles()
                sites = starting.get_chemical_symbols()

                #Randomly choose a, b, c, beta, placement, rotation and dihedrals
                # else:
                #     lat[2] = lat[1]
                if 'alpha' in parameter_set:
                    lat[3] = (high_angle - low_angle) * mc.gauss_number() + low_angle
                # else:
                #     lat[3] = 90.0
                if 'beta' in parameter_set:
                    lat[4] = (high_angle - low_angle) * mc.gauss_number() + low_angle
                # else:
                #     lat[4] = 90.0
                if 'gamma' in parameter_set:
                    lat[5] = (high_angle - low_angle) * mc.gauss_number() + low_angle
                # else:
                #     lat[5] = 90.0

                if 'rot' in parameter_set:
                    quat = np.zeros(4)
                    quat[0] = 0.5-random.random()
                    quat[1] = 0.5-random.random()
                    quat[2] = 0.5-random.random()
                    quat[3] = 0.5-random.random()
                    quat /= np.linalg.norm(quat)
                    #rx = (rotate_high - rotate_low) * random.random() + rotate_low
                    #ry = (rotate_high - rotate_low) * random.random() + rotate_low
                    #rz = (rotate_high - rotate_low) * random.random() + rotate_low
                    #rotation = [rx, ry, rz]
                else:
                    quat = [1,0,0,0]
                    #rotation = [0, 0, 0]

                if smart_cell:
                    cell_l = ["a", "b", "c"]
                    low_lens = get_projections(starting, lat[3], lat[4], lat[5], atoms, quat)
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
                else:
                    if 'a' in parameter_set:
                        lat[0] = (7.5*n_mol - 2.0*n_mol) * random.random() + 2.0*n_mol
                    if 'b' in parameter_set:
                        lat[1] = (7.5*n_mol - 2.0*n_mol) * random.random() + 2.0*n_mol
                    if 'c' in parameter_set:
                        lat[2] = (7.5*n_mol - 2.0*n_mol) * random.random() + 2.0*n_mol
                            
#                if 'a'in parameter_set:
#                    lat[0] = (high_len-low_len)*random.random()+low_len
#                if 'b' in parameter_set:
#                    lat[1] = (high_len-low_len)*random.random()+low_len
#                if 'c' in parameter_set:
#                    sqroot = 1+(2*np.cos(np.deg2rad(lat[3]))*np.cos(np.deg2rad(lat[4]))*np.cos(np.deg2rad(lat[5])))
#                    sqroot += -np.cos(np.deg2rad(lat[3]))**2-np.cos(np.deg2rad(lat[4]))**2-np.cos(np.deg2rad(lat[5]))**2
#                    sqroot = np.sqrt(sqroot)
#                    low_c = Vmol / (lat[0]*lat[1]*sqroot)
#                    lat[2] = (vol_high*low_c-low_c)*random.random()+low_c

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
                trial_crystal = create_crystal(starting_copy,molecule,sg,atoms,sites,lat,trans,quat,starting_angles, n_mol)
                break
            except Exception as e:
                print(e)
                pass

        #print check_for_overlap(trial_crystal,cut,close_atoms,Vmol,vol_high)
        Vs.append(trial_crystal.get_volume())

        # Check if the generated cell is good, if not, try again
        if check_for_overlap(trial_crystal,cut,close_atoms,Vmol,vol_high,old_overlap):
            print "Cell relative volume: " + str((trial_crystal.get_volume())/Vmol)
            break

    return trial_crystal, lat, trans, quat, starting_angles, n_failed
