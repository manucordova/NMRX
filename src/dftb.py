import os
import sys
import numpy as np
import ase
import ase.io
import subprocess as sp
import pprint
import shutil
import random

def compute_k_points(lattice, factor):
    k_points = []
    for i in range(3):
        k_points.append(max(1, int(1 / lattice[i] / factor)))
    return k_points


def compute_s_values(k_points):
    s_values = []
    for i in range(3):
        if k_points[i] // 2 == k_points[i] / 2:
            s_values.append(0.0)
        else:
            s_values.append(0.5)
    return s_values


def make_dftb_input(xyz, k_points_factor, skfdir, outdir, dispersion):
    pos = xyz.get_positions()
    nr = xyz.get_number_of_atoms()
    typ = xyz.get_chemical_symbols()
    elements = []
    ntyp = 0
    for atom in typ:
        if atom not in elements:
            elements.append(atom)
            ntyp += 1
    abc = xyz.get_cell_lengths_and_angles()
    cell = xyz.get_cell()

    pp = "Geometry = {\n"
    pp += " TypeNames = { "
    for elem in elements:
        pp += "\"" + elem + "\" "
    pp += "}\n TypesAndCoordinates [Angstrom] = {\n"
    for i, position in enumerate(pos):
        pp += "  " + str(elements.index(typ[i]) + 1) + " " + str(round(position[0], 6)) + " " + str(
            round(position[1], 6)) + " " + str(round(position[2], 6)) + "\n"
    pp += " }\n Periodic = Yes\n"

    pp += " LatticeVectors [Angstrom] = {\n"
    for k in range(3):
        for x in range(3):
            pp += "  " + '{:8.6f}'.format(cell[k][x])
        pp += '\n'
    pp += " }\n}\n"

    pp += "Hamiltonian = DFTB {\n"
    pp += " SCC = Yes\n"
    pp += " MaxAngularMomentum = {\n"
    for elem in elements:
        if elem == "H":
            pp += "  " + elem + " = \"s\"\n"
        elif elem in ["C", "N", "O","S"]:
            pp += "  " + elem + " = \"p\"\n"
    pp += " }\n"

    pp += " SlaterKosterFiles = {\n"
    for elem1 in elements:
        for elem2 in elements:
            pp += "  " + elem1 + "-" + elem2 + " = \"" + skfdir + elem1 + "-" + elem2 + ".skf\"\n"
    pp += " }\n"

    k_points = compute_k_points(abc[:3], k_points_factor)
    pp += " KPointsAndWeights = SupercellFolding {\n"
    pp += "  " + str(k_points[0]) + " 0 0\n"
    pp += "  0 " + str(k_points[1]) + " 0\n"
    pp += "  0 0 " + str(k_points[2]) + "\n"
    s_values = compute_s_values(k_points)
    pp += "  " + str(round(s_values[0], 1)) + " " + str(round(s_values[1], 1)) + " " + str(round(s_values[2], 1)) + "\n"
    pp += " }\n"

    if dispersion == "D3":
        pp += " Dispersion = DftD3 { Damping = BeckeJohnson {\n"
        pp += "  a1 = 0.5719\n"
        pp += "  a2 = 3.6017 }\n"
        pp += "  s6 = 1.0\n"
        pp += "  s8 = 0.5883\n"
        pp += " }\n"
    elif dispersion == "LJ":
        pp += " Dispersion = LennardJones {\n"
        pp += "  Parameters = UFFParameters {}\n"
        pp += " }\n"
    elif dispersion != "None":
        raise ValueError("Unknown dispersion")

    pp += "}\n"

    with open(outdir + "/dftb_in.hsd", "w") as f:
        f.write(pp)


def dftbplus_energy(directory, struct, dftb_path, dispersion="D3"):
    """
    DFTB+ full energy computation

    Options:    struct = ase.Atoms object. Crystal structure to compute the energy for
                fileformat = "cif", "xyz", ... (any format valid for ASE)
                dispersion = "D3", "LJ", "None"
    """
    number = str(random.random())
    if not os.path.isdir(directory+number+"tmp/"):
        os.mkdir(directory+number+"tmp/")
    initdir = os.getcwd()
    outdir = os.path.abspath(directory+number+"tmp/")
    skfdir = os.path.abspath("../src/dftb_sk_files/") + "/"

    k_points_factor = 0.06  # Length for k-point sampling in reciprocal space (A^-1)

    make_dftb_input(struct, k_points_factor, skfdir, outdir, dispersion)

    os.chdir(outdir)
    process = sp.Popen(dftb_path, stdout=sp.PIPE)

    os.chdir(initdir)
    output, error = process.communicate()
    outputStr = output.decode("utf-8").split("\n")
    out = [s for s in outputStr if 'Total Energy' in s]
    E = [float(s) for s in out[0].split(" ") if s.replace(".", "", 1).replace("-", "", 1).isdigit()]
    shutil.rmtree(outdir)

    return E[0]


# # E= dftbplus_energy("/Users/balodis/work/azd8329/structure_files/", "/Users/balodis/work/azd8329/structure_files/azd8329_csp.cif", dispersion="D3")*2625.50
# E= dftbplus_energy("/Users/balodis/work/ritonavir/structure_files/Ritonavir_reordered/", "/Users/balodis/work/ritonavir/structure_files/Ritonavir_reordered/Ritonavir_polymorph_2_DFTB_vc-relax.cif", dispersion="D3")*2625.50
#
#
# #E = dftbplus_energy("/Users/balodis/work/ritonavir/results/2020/ritonavir_conf_const_test/4000_loops_0.005_factor__H1_True_C13_False_test_1/","/Users/balodis/work/ritonavir/results/2020/ritonavir_conf_const_test/4000_loops_0.005_factor__H1_True_C13_False_test_1/1_0_opt_structure.cif", dispersion="D3")*2625.50
# print(E)

