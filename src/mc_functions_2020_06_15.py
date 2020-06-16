# coding=utf-8

import ase
from ase.io import read,write
import fnmatch
import numpy as np
from ase.spacegroup import crystal
import random


def change_labels_from_0_to_1(in_file,out_file):
    """Ase starts numbering from 0, so this file helps to translate the numbering that starts from 1"""
    dump_in = open(in_file,"r")
    dump_out = open(out_file,"w")
    i = 1
    for line in dump_in:
        if fnmatch.fnmatch(line,'*ATOM*'):
            old_line = line
            if i == 10:
                old_line = old_line.replace(' ','',1)
            new_line = old_line.replace(str(i-1),str(i),1)
            dump_out.write(new_line)
            i = i + 1
        else:
            dump_out.write(line)


def convert_xyz(file_list,input_dir,output_dir,extension='pdb'):
    """Helper function to convert xyz files to other formats"""
    dump_in = open(file_list,"r")
    for line in dump_in:
        in_mol = read(input_dir + line[:-1],index='0')
        in_mol.write(output_dir + line[:-4] + extension)
    dump_in.close()


def extract_crystal_parameters(xyzPath,k):
    fin = open(xyzPath + 'asym_unit/cocain_{}.xyz'.format(k), 'r')
    data = fin.readlines()
    fin.close()

    # extract asym.unit information from file
    nn = int(data[0])
    uc = data[1].split(':')
    sg = int(uc[1].split()[0])
    uc = uc[-1].split()
    lat = []
    for i1 in range(0, 9):
        lat.append(float(uc[i1]))
    lat = np.array(lat).reshape((3, 3))
    sites = []
    frac = []
    for dd in data[2:]:
        dd = dd.split()
        sites.append(dd[0])
        ff = []
        for i1 in range(1, 4):
            ff.append(float(dd[i1]))
        frac.append(ff)
    return sites, frac, sg, lat

def create_crystal(sites,frac,sg,lat):
    return crystal(symbols=sites, basis=frac, spacegroup=sg, cell=lat, symprec=0.001)



def rewrite_coordinate_old(test,molecule="cocaine"):
    """A function to deal with change in coordinates after creating a crystal with ASE"""
    coordinates = test.get_positions()

    if molecule=="cocaine":
        mapping_atom = range(85)[0::2]
        new_sites = ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                     'C', 'C', 'C', 'C', 'N', 'O', 'O', 'O', 'O', 'H', 'H', 'H', 'H', 'H',
                     'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'C',
                     'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                     'C', 'C', 'C', 'C', 'N', 'O', 'O', 'O', 'O', 'H', 'H', 'H', 'H', 'H',
                     'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H']
        new_coordinates = [0] * 86
        i = 0
        for line in coordinates:
            if i in mapping_atom:
                new_coordinates[mapping_atom.index(i)] = [line[0], line[1], line[2]]
            else:
                new_coordinates[mapping_atom.index(i - 1) + 43] = [line[0], line[1], line[2]]
            i = i + 1

    if molecule=="azd":
        mapping_atom = range(124)[0::2]
        new_sites = ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'N', 'N', 'N', 'O', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'N', 'N', 'N', 'O', 'O', 'O']
        new_coordinates = [0] * 124
        i = 0
        for line in coordinates:
            if i in mapping_atom:
                new_coordinates[mapping_atom.index(i)] = [line[0], line[1], line[2]]
            else:
                new_coordinates[mapping_atom.index(i - 1) + 62] = [line[0], line[1], line[2]]
            i = i + 1

    if molecule=="azd":
        mapping_atom = range(124)[0::2]
        new_sites = ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'N', 'N', 'N', 'O', 'O', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'N', 'N', 'N', 'O', 'O', 'O']
        new_coordinates = [0] * 124
        i = 0
        for line in coordinates:
            if i in mapping_atom:
                new_coordinates[mapping_atom.index(i)] = [line[0], line[1], line[2]]
            else:
                new_coordinates[mapping_atom.index(i - 1) + 62] = [line[0], line[1], line[2]]
            i = i + 1

    test.set_positions(new_coordinates)
    test.set_chemical_symbols(new_sites)

    return test


def rewrite_coordinate(test,nr_molecules,molecule="cocaine"):
    """A function to deal with change in coordinates after creating a crystal with ASE"""
    coordinates = test.get_positions()

    if molecule=="cocaine":
        atoms = 43
        new_sites = ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                     'C', 'C', 'C', 'C', 'N', 'O', 'O', 'O', 'O', 'H', 'H', 'H', 'H', 'H',
                     'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H']*nr_molecules
    if molecule == "azd":
        atoms = 62
        new_sites = ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                     'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                     'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'N',
                     'N', 'N', 'O', 'O', 'O'] * nr_molecules
    if molecule == "ritonavir":
        atoms = 98
        new_sites = ['S', 'S', 'O', 'O', 'O', 'O', 'O', 'N', 'N', 'N', 'N', 'N', 'N', 'C', 'C', 'C', 'C', 'C', 'C',
                     'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
                     'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                     'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                     'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                     'H', 'H', 'H'] * nr_molecules

    mapping_atom = range(atoms * nr_molecules - 1)[0::nr_molecules]

    new_coordinates = [0] * atoms * nr_molecules
    i = 0
    for line in coordinates:
        for j in range(nr_molecules):
            #print i, j
            if i - j in mapping_atom:
                new_coordinates[mapping_atom.index(i - j) + atoms * j] = [line[0], line[1], line[2]]
        i = i + 1

    test.set_positions(new_coordinates)
    test.set_chemical_symbols(new_sites)
    return test





def write_asym(frac_coord, space_group, lattice, sites, nr_mol, file_name):
    frac = frac_coord
    lat = lattice
    cnr = nr_mol
    Nat = len(sites)
    sg = space_group

    # write asym. unit as xyz (in comment line put lattice-params & space-group & nr of original conf)
    data = '{}\n#SG: {} CONV: {} CELL (Angstrom): '.format(Nat, sg, cnr)
    for i1 in range(0, 3):
        data += ' {:f} {:f} {:f}'.format(lat[i1][0], lat[i1][1], lat[i1][2])
    data += '\n'
    for i1 in range(0, Nat):
        data += '{}\t{:f}\t{:f}\t{:f}\n'.format(sites[i1], frac[i1][0], frac[i1][1], frac[i1][2])
    fout = open(file_name, 'w')
    fout.write(data)
    fout.close()

def gauss_number(mean=0.5):
    while True:
        # Checks
        number = np.random.normal(mean, 0.2)
        if number > 0 and number < 1:
            break
    return number







def conf_angles(angle,randomizer="gauss",molecule="cocaine"):
    high_angle = angle
    low_angle = -angle

    if randomizer == "uniform":
        angle_1 = (high_angle - low_angle) * random.random() + low_angle
        angle_2 = (high_angle - low_angle) * random.random() + low_angle
        angle_3 = (high_angle - low_angle) * random.random() + low_angle
        angle_4 = (high_angle - low_angle) * random.random() + low_angle
        angle_5 = (high_angle - low_angle) * random.random() + low_angle
        angle_6 = (high_angle - low_angle) * random.random() + low_angle
        angle_7 = (high_angle - low_angle) * random.random() + low_angle
        angle_8 = (high_angle - low_angle) * random.random() + low_angle
        angle_9 = (high_angle - low_angle) * random.random() + low_angle
        angle_10 = (high_angle - low_angle) * random.random() + low_angle
        angle_11 = (high_angle - low_angle) * random.random() + low_angle
        angle_12 = (high_angle - low_angle) * random.random() + low_angle
        angle_13 = (high_angle - low_angle) * random.random() + low_angle
        angle_14 = (high_angle - low_angle) * random.random() + low_angle
        angle_15 = (high_angle - low_angle) * random.random() + low_angle
        angle_16 = (high_angle - low_angle) * random.random() + low_angle
        angle_17 = (high_angle - low_angle) * random.random() + low_angle
        angle_18 = (high_angle - low_angle) * random.random() + low_angle
        angle_19 = (high_angle - low_angle) * random.random() + low_angle
        angle_20 = (high_angle - low_angle) * random.random() + low_angle
        angle_21 = (high_angle - low_angle) * random.random() + low_angle
        angle_22 = (high_angle - low_angle) * random.random() + low_angle
        angle_23 = (high_angle - low_angle) * random.random() + low_angle


    else:
        angle_1 = (high_angle - low_angle) * gauss_number() + low_angle
        angle_2 = (high_angle - low_angle) * gauss_number() + low_angle
        angle_3 = (high_angle - low_angle) * gauss_number() + low_angle
        angle_4 = (high_angle - low_angle) * gauss_number() + low_angle
        angle_5 = (high_angle - low_angle) * gauss_number() + low_angle
        angle_6 = (high_angle - low_angle) * gauss_number() + low_angle
        angle_7 = (high_angle - low_angle) * random.random() + low_angle
        angle_8 = (high_angle - low_angle) * random.random() + low_angle
        angle_9 = (high_angle - low_angle) * random.random() + low_angle
        angle_10 = (high_angle - low_angle) * random.random() + low_angle
        angle_11 = (high_angle - low_angle) * random.random() + low_angle
        angle_12 = (high_angle - low_angle) * random.random() + low_angle
        angle_13 = (high_angle - low_angle) * random.random() + low_angle
        angle_14 = (high_angle - low_angle) * random.random() + low_angle
        angle_15 = (high_angle - low_angle) * random.random() + low_angle
        angle_16 = (high_angle - low_angle) * random.random() + low_angle
        angle_17 = (high_angle - low_angle) * random.random() + low_angle
        angle_18 = (high_angle - low_angle) * random.random() + low_angle
        angle_19 = (high_angle - low_angle) * random.random() + low_angle
        angle_20 = (high_angle - low_angle) * random.random() + low_angle
        angle_21 = (high_angle - low_angle) * random.random() + low_angle
        angle_22 = (high_angle - low_angle) * random.random() + low_angle
        angle_23 = (high_angle - low_angle) * random.random() + low_angle

    return [angle_1,angle_2,angle_3,angle_4,angle_5,angle_6,angle_7,angle_8,angle_9,angle_10,angle_11,angle_12,angle_13,
            angle_14,angle_15,angle_16,angle_17,angle_18,angle_19,angle_20,angle_21,angle_22,angle_23]


def change_conformation(abc, angle_1, angle_2, angle_3, angle_4, angle_5, angle_6, angle_7, angle_8, angle_9, angle_10,
                        angle_11, angle_12, angle_13, angle_14, angle_15, angle_16, angle_17, angle_18, angle_19,
                        angle_20, angle_21, angle_22, angle_23, molecule):

    mask = np.zeros(len(abc))
    if molecule == "cocaine":
        for k1 in [8, 9, 10, 11, 12, 13, 32, 33, 34, 35, 36]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=13, a2=8, a3=7, a4=18, angle=angle_1, mask=mask)

        mask = np.zeros(len(abc))
        for k1 in [8, 9, 10, 11, 12, 13, 32, 33, 34, 35, 36, 19, 7]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=19, a2=7, a3=18, a4=2, angle=angle_2, mask=mask)

        mask = np.zeros(len(abc))
        for k1 in [8, 9, 10, 11, 12, 13, 32, 33, 34, 35, 36, 19, 7, 18]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=7, a2=18, a3=2, a4=3, angle=angle_3, mask=mask)

        mask = np.zeros(len(abc))
        for k1 in [20, 14, 21, 15, 37, 38, 39]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=2, a2=1, a3=14, a4=20, angle=angle_4, mask=mask)

        mask = np.zeros(len(abc))
        for k1 in [21, 15, 37, 38, 39]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=1, a2=14, a3=21, a4=15, angle=angle_5, mask=mask)

    if molecule == "azd":

        mask = np.zeros(len(abc))
        for k1 in [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,42,43,44,45,46,47,48,49,50,51]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=6, a2=58, a3=42, a4=7, angle=angle_1, mask=mask)

        mask = np.zeros(len(abc))
        for k1 in [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,42,43,44,45,46,47,48,49,50,51,58,6]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=61, a2=41, a3=58, a4=6, angle=angle_2, mask=mask)

        mask = np.zeros(len(abc))
        for k1 in [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,42,43,44,45,46,47,48,49,50,51,58,6,41,61]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=40, a2=39, a3=41, a4=61, angle=angle_3, mask=mask)

        mask = np.zeros(len(abc))
        for k1 in [52,53,54,55,28,29,30,23,22,24,25,26,27]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=55, a2=52, a3=38, a4=39, angle=angle_4, mask=mask)

        mask = np.zeros(len(abc))
        for k1 in [4, 60, 37, 59, 33, 32, 31, 36, 35, 34, 1, 0, 2, 3]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=31, a2=36, a3=56, a4=57, angle=angle_5, mask=mask)

        mask = np.zeros(len(abc))
        for k1 in [4, 60, 37, 59]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=59, a2=37, a3=33, a4=32, angle=angle_6, mask=mask)

    if molecule == "ritonavir":
        # Dih1
        mask = np.zeros(len(abc))
        for k1 in [56, 47, 12, 0, 46, 55, 45]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=46, a2=45, a3=44, a4=6, angle=angle_1, mask=mask)

        # Dih2
        mask = np.zeros(len(abc))
        for k1 in [56, 47, 12, 0, 46, 55, 45, 44, 53, 54]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=45, a2=44, a3=6, a4=43, angle=angle_2, mask=mask)

        # Dih3
        mask = np.zeros(len(abc))
        for k1 in [56, 47, 12, 0, 46, 55, 45, 44, 53, 54, 6]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=44, a2=6, a3=43, a4=11, angle=angle_3, mask=mask)

        # Dih4
        mask = np.zeros(len(abc))
        for k1 in [56, 47, 12, 0, 46, 55, 45, 44, 53, 54, 6, 43, 5]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=6, a2=43, a3=11, a4=97, angle=angle_4, mask=mask)

        # Dih5
        mask = np.zeros(len(abc))
        for k1 in [56, 47, 12, 0, 46, 55, 45, 44, 53, 54, 6, 43, 5, 11, 97]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=43, a2=11, a3=36, a4=72, angle=angle_5, mask=mask)

        # Dih6
        mask = np.zeros(len(abc))
        for k1 in [38, 39, 40, 41, 42, 50, 51, 52, 49, 60, 75]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=49, a2=38, a3=37, a4=73, angle=angle_6, mask=mask)

        # Dih7
        mask = np.zeros(len(abc))
        for k1 in [38, 39, 40, 41, 42, 50, 51, 52, 49, 60, 75, 37, 73, 74]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=38, a2=37, a3=36, a4=11, angle=angle_7, mask=mask)

        # Dih8
        mask = np.zeros(len(abc))
        for k1 in [38, 39, 40, 41, 42, 50, 51, 52, 49, 60, 75, 37, 73, 74, 56, 47, 12, 0, 46, 55, 45, 44, 53, 54, 6, 43,
                   5, 11, 97, 36, 72]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=11, a2=36, a3=35, a4=71, angle=angle_8, mask=mask)

        # Dih9
        mask = np.zeros(len(abc))
        for k1 in [93, 4]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=93, a2=4, a3=35, a4=34, angle=angle_9, mask=mask)

        # Dih10
        mask = np.zeros(len(abc))
        for k1 in [38, 39, 40, 41, 42, 50, 51, 52, 49, 60, 75, 37, 73, 74, 56, 47, 12, 0, 46, 55, 45, 44, 53, 54, 6, 43,
                   5, 11, 97, 36, 72, 35, 71, 4, 93]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=36, a2=35, a3=34, a4=70, angle=angle_10, mask=mask)

        # Dih11
        mask = np.zeros(len(abc))
        for k1 in [38, 39, 40, 41, 42, 50, 51, 52, 49, 60, 75, 37, 73, 74, 56, 47, 12, 0, 46, 55, 45, 44, 53, 54, 6, 43,
                   5, 11, 97, 36, 72, 35, 71, 4, 93, 34, 69, 70]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=35, a2=34, a3=13, a4=76, angle=angle_11, mask=mask)

        # Dih12
        mask = np.zeros(len(abc))
        for k1 in [15, 16, 17, 18, 19, 20, 79, 96, 80, 81, 82]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=21, a2=15, a3=14, a4=13, angle=angle_12, mask=mask)

        # Dih13
        mask = np.zeros(len(abc))
        for k1 in [15, 16, 17, 18, 19, 20, 79, 96, 80, 81, 82, 14, 77, 78]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=15, a2=14, a3=13, a4=76, angle=angle_13, mask=mask)

        # Dih14
        mask = np.zeros(len(abc))
        for k1 in [48, 59, 57, 58, 25, 85, 86, 87, 24, 84]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=25, a2=24, a3=23, a4=7, angle=angle_14, mask=mask)

        # Dih15
        mask = np.zeros(len(abc))
        for k1 in [48, 59, 57, 58, 25, 85, 86, 87, 24, 84, 23, 7, 1, 21, 22, 83]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=22, a2=21, a3=26, a4=88, angle=angle_15, mask=mask)

        # Dih16 #12k tries, around an hour
        mask = np.zeros(len(abc))
        for k1 in [48, 59, 57, 58, 25, 85, 86, 87, 24, 84, 23, 7, 1, 21, 22, 83, 26, 89, 88]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=21, a2=26, a3=8, a4=28, angle=angle_16, mask=mask)

        # Dih17
        mask = np.zeros(len(abc))
        for k1 in [48, 59, 57, 58, 25, 85, 86, 87, 24, 84, 23, 7, 1, 21, 22, 83, 26, 89, 88, 8, 27, 90, 91, 92]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=26, a2=8, a3=28, a4=2, angle=angle_17, mask=mask)

        # Dih18
        mask = np.zeros(len(abc))
        for k1 in [48, 59, 57, 58, 25, 85, 86, 87, 24, 84, 23, 7, 1, 21, 22, 83, 26, 89, 88, 8, 27, 90, 91, 92, 28, 2]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=8, a2=28, a3=9, a4=95, angle=angle_18, mask=mask)

        # Dih19
        mask = np.zeros(len(abc))
        for k1 in [48, 59, 57, 58, 25, 85, 86, 87, 24, 84, 23, 7, 1, 21, 22, 83, 26, 89, 88, 8, 27, 90, 91, 92, 28, 2,
                   9, 94]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=28, a2=9, a3=32, a4=30, angle=angle_19, mask=mask)

        # Dih20
        mask = np.zeros(len(abc))
        for k1 in [29, 61, 62, 63, 30, 64, 31, 65, 67, 66]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=29, a2=30, a3=32, a4=68, angle=angle_20, mask=mask)

        # Dih21
        mask = np.zeros(len(abc))
        for k1 in [48, 59, 57, 58, 25, 85, 86, 87, 24, 84, 23, 7, 1, 21, 22, 83, 26, 89, 88, 8, 27, 90, 91, 92, 28, 2,
                   9, 94, 29, 61, 62, 63, 30, 64, 31, 65, 67, 66, 32, 68]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=9, a2=32, a3=33, a4=3, angle=angle_21, mask=mask)

        # Dih22
        mask = np.zeros(len(abc))
        for k1 in [48, 59, 57, 58, 25, 85, 86, 87, 24, 84, 23, 7, 1, 21, 22, 83, 26, 89, 88, 8, 27, 90, 91, 92, 28, 2,
                   9, 94, 29, 61, 62, 63, 30, 64, 31, 65, 67, 66, 32, 68, 33, 3]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=32, a2=33, a3=10, a4=95, angle=angle_22, mask=mask)

        # Dih23
        mask = np.zeros(len(abc))
        for k1 in [48, 59, 57, 58, 25, 85, 86, 87, 24, 84, 23, 7, 1, 21, 22, 83, 26, 89, 88, 8, 27, 90, 91, 92, 28, 2,
                   9, 94, 29, 61, 62, 63, 30, 64, 31, 65, 67, 66, 32, 68, 33, 3, 10, 95]:
            mask[k1] = 1
        abc.rotate_dihedral(a1=33, a2=10, a3=13, a4=76, angle=angle_23, mask=mask)

    #print angle_1, angle_2, angle_3, angle_4, angle_5
    return abc

# def conf_angles(angle,randomizer="gauss",molecule="cocaine"):
#     high_angle = angle
#     low_angle = -angle
#
#     if randomizer == "uniform":
#         angle_1 = (high_angle - low_angle) * random.random() + low_angle
#         angle_2 = (high_angle - low_angle) * random.random() + low_angle
#         angle_3 = (high_angle - low_angle) * random.random() + low_angle
#         angle_4 = (high_angle - low_angle) * random.random() + low_angle
#         angle_5 = (high_angle - low_angle) * random.random() + low_angle
#         angle_6 = (high_angle - low_angle) * random.random() + low_angle
#     else:
#         angle_1 = (high_angle - low_angle) * gauss_number() + low_angle
#         angle_2 = (high_angle - low_angle) * gauss_number() + low_angle
#         angle_3 = (high_angle - low_angle) * gauss_number() + low_angle
#         angle_4 = (high_angle - low_angle) * gauss_number() + low_angle
#         angle_5 = (high_angle - low_angle) * gauss_number() + low_angle
#         angle_6 = (high_angle - low_angle) * gauss_number() + low_angle
#     return [angle_1,angle_2,angle_3,angle_4,angle_5]
#
#
# def change_conformation(abc, angle_1, angle_2, angle_3, angle_4, angle_5):
#
#     mask = np.zeros(len(abc))
#     for k1 in [8, 9, 10, 11, 12, 13, 32, 33, 34, 35, 36]:
#         mask[k1] = 1
#     abc.rotate_dihedral(a1=13, a2=8, a3=7, a4=18, angle=angle_1, mask=mask)
#
#     mask = np.zeros(len(abc))
#     for k1 in [8, 9, 10, 11, 12, 13, 32, 33, 34, 35, 36, 19, 7]:
#         mask[k1] = 1
#     abc.rotate_dihedral(a1=19, a2=7, a3=18, a4=2, angle=angle_2, mask=mask)
#
#     mask = np.zeros(len(abc))
#     for k1 in [8, 9, 10, 11, 12, 13, 32, 33, 34, 35, 36, 19, 7, 18]:
#         mask[k1] = 1
#     abc.rotate_dihedral(a1=7, a2=18, a3=2, a4=3, angle=angle_3, mask=mask)
#
#     mask = np.zeros(len(abc))
#     for k1 in [20, 14, 21, 15, 37, 38, 39]:
#         mask[k1] = 1
#     abc.rotate_dihedral(a1=2, a2=1, a3=14, a4=20, angle=angle_4, mask=mask)
#
#     mask = np.zeros(len(abc))
#     for k1 in [21, 15, 37, 38, 39]:
#         mask[k1] = 1
#     abc.rotate_dihedral(a1=1, a2=14, a3=21, a4=15, angle=angle_5, mask=mask)
#
#     #print angle_1, angle_2, angle_3, angle_4, angle_5
#     return abc




# xyz=read('/Users/balodis/work/cocaine/structure_files/COCAIN10_H_relaxed_out_cell.pdb')
# angles = conf_angles(0)
# view(change_conformation(xyz,angles[0],angles[1],angles[2],angles[3],angles[4]))