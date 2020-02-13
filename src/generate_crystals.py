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
from copy import copy
from tqdm import tqdm_notebook
import cPickle as pck

import scipy.optimize as op

import mc_functions as mc

def create_crystal(structure,molecule,sg,atoms,sites,lat,trans,rotation,conf_angles, n_mol):

    structure.translate(trans)

    mass = structure[0:atoms].get_center_of_mass()
    structure.rotate(rotation[0], v='x', center=mass)
    structure.rotate(rotation[1], v='y', center=mass)
    structure.rotate(rotation[2], v='z', center=mass)

    structure = mc.change_conformation(structure,conf_angles[0],conf_angles[1],conf_angles[2],conf_angles[3],conf_angles[4],conf_angles[5],molecule)

    structure.set_cell(lat, scale_atoms=False)

    scaled = structure.get_scaled_positions(wrap=False)
    trial_crystal = crystal(symbols=sites, basis=scaled[:atoms], spacegroup=sg, cell=lat, symprec=0.001, pbc=True)
    trial_crystal = mc.rewrite_coordinate(trial_crystal, n_mol, molecule=molecule)


    return trial_crystal

def get_third_length(a, b, alpha, beta, gamma):
    
    return c

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

    return condition_1 and condition_2

def generate_crystal(initial_structure, parameter_set, high_len, low_len, high_angle, low_angle, high_trans, low_trans, rotate_high, rotate_low, sg, atoms, n_mol, molecule, cut, close_atoms, vol_high):
    
    # Calculate the volume of the molecule
    starting = read(initial_structure)
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
                #Read in the structure with the right conformation and coordinates
                starting = read(initial_structure)

                #Get the lattice parameters and sites from the reference structure
                lat = starting.get_cell_lengths_and_angles()
                sites = starting.get_chemical_symbols()

                #Randomly choose a, b, c, beta, placement, rotation and dihedrals
                # else:
                #     lat[2] = lat[1]
                if 'alpha' in parameter_set:
                    lat[3] = (high_angle - low_angle) * mc.gauss_number() + low_angle
                else:
                    lat[3] = 90.0
                if 'beta' in parameter_set:
                    lat[4] = (high_angle - low_angle) * mc.gauss_number() + low_angle
                else:
                    lat[4] = 90.0
                if 'gamma' in parameter_set:
                    lat[5] = (high_angle - low_angle) * mc.gauss_number() + low_angle
                else:
                    lat[5] = 90.0
                    
                if 'a'in parameter_set:
                    lat[0] = (high_len-low_len)*random.random()+low_len
                if 'b' in parameter_set:
                    lat[1] = (high_len-low_len)*random.random()+low_len
                if 'c' in parameter_set:
                    sqroot = 1+(2*np.cos(np.deg2rad(lat[3]))*np.cos(np.deg2rad(lat[4]))*np.cos(np.deg2rad(lat[5])))
                    sqroot += -np.cos(np.deg2rad(lat[3]))**2-np.cos(np.deg2rad(lat[4]))**2-np.cos(np.deg2rad(lat[5]))**2
                    sqroot = np.sqrt(sqroot)
                    low_c = Vmol / (lat[0]*lat[1]*sqroot)
                
                    lat[2] = (vol_high*low_c-low_c)*random.random()+low_c

                if 'trans' in parameter_set:
                    x = (high_trans - low_trans) * random.random() + low_trans
                    y = (high_trans - low_trans) * random.random() + low_trans
                    z = (high_trans - low_trans) * random.random() + low_trans
                    trans = [x, y, z]
                else:
                    trans = [0, 0, 0]

                if 'rot' in parameter_set:
                    rx = (rotate_high - rotate_low) * random.random() + rotate_low
                    ry = (rotate_high - rotate_low) * random.random() + rotate_low
                    rz = (rotate_high - rotate_low) * random.random() + rotate_low
                    rotation = [rx, ry, rz]
                else:
                    rotation = [0, 0, 0]

                if 'conf' in parameter_set:
                    starting_angles = mc.conf_angles(180, 'uniform')
                else:
                    starting_angles = [0, 0, 0, 0, 0, 0]


                #Overwrite the starting structure and create a trial structure for testing
                trial_crystal = create_crystal(starting,molecule,sg,atoms,sites,lat,trans,rotation,starting_angles, n_mol)
                break
            except Exception as e:
                print(e)
                pass

        #print check_for_overlap(trial_crystal,cut,close_atoms,Vmol,vol_high)

        # Check if the generated cell is good, if not, try again
        if check_for_overlap(trial_crystal,cut,close_atoms,Vmol,vol_high):
            break
    
    return trial_crystal, lat, trans, rotation, starting_angles, n_failed
