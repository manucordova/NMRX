#####
# Last modified 29.06.2020
# Re-writing the script so that it is more general
# Authors: Manuel Cordova, Martins Balodis
#####



### Import libraries
import numpy as np
import os
import sys
import ase
import ase.io
import subprocess as sp
import shutil



def which(pgm):
    """
    Find the path to a program
    
    Inputs:     - pgm   Program name
    
    Outputs:    - p     path to program pgm
    """
    
    path = os.getenv('PATH')
    for p in path.split(os.path.pathsep):
        p = os.path.join(p, pgm)
        if os.path.exists(p) and os.access(p, os.X_OK):
        
            return p
    
    raise ValueError("Path not found.")
    
    return



def compute_k_points(lattice, factor):
    """
    Generate k-points from the distance in k-space (given by the k-point factor)
    
    Inputs:     - lattice   Unit cell lengths
                - factor    Maximum distance in k-space
    
    Outputs:    - k_points  Number of k-points in each dimension
    """
    k_points = []
    for i in range(3):
        k_points.append(max(1, int(1 / lattice[i] / factor)))
    return k_points



def compute_s_values(k_points):
    """
    Convert k-points in s values used by DFTB
    
    Inputs:     - k_points      List of k-points
    
    Outputs:    - s_values      List of s-values used by DFTB
    """
    s_values = []
    for i in range(3):
        if k_points[i] // 2 == k_points[i] / 2:
            s_values.append(0.0)
        else:
            s_values.append(0.5)
    return s_values



def make_dftb_input(xyz, k_points_factor, skfdir, outdir, dispersion):
    """
    Generate input file for DFTB
    
    Inputs:     - xyz               Structure of the crystal
                - k_points_factor   Factor for the k-point grid
                - skfdir            Directory for the Slater-Koster files
                - outdir            Directory where the input file should be written
                - dispersion        Type of dispersion correction to use
    """
    
    # Get atomic symbols and positions
    pos = xyz.get_positions()
    nr = xyz.get_number_of_atoms()
    typ = xyz.get_chemical_symbols()
    elements = []
    ntyp = 0
    for atom in typ:
        if atom not in elements:
            elements.append(atom)
            ntyp += 1
    # Get cell parameters
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
    
    # Write the generated input file
    with open(outdir + "/dftb_in.hsd", "w") as f:
        f.write(pp)
        
    return



def dftbplus_energy(directory, struct, dftb_path, dispersion="D3"):
    """
    DFTB+ single-point energy computation
    
    Inputs:     - directory     F
                - struct
                - dftb_path     Path for the DFTB program
                - dispersion    Type of dispersion correction to use
    
    Outputs:    - E     F
    Options:    struct = ase.Atoms object. Crystal structure to compute the energy for
                fileformat = "cif", "xyz", ... (any format valid for ASE)
                dispersion = "D3", "LJ", "None"
    """
    
    # Create temporary directory to store the DFTB files
    number = str(np.random.random())
    if not os.path.exists(directory+number+"tmp/"):
        os.mkdir(directory+number+"tmp/")
    initdir = os.getcwd()
    outdir = os.path.abspath(directory+number+"tmp/")
    skfdir = os.path.abspath("../src/dftb_sk_files/") + "/"

    k_points_factor = 0.05  # Length for k-point sampling in reciprocal space (A^-1)

    # Generate DFTB input file
    make_dftb_input(struct, k_points_factor, skfdir, outdir, dispersion)

    # Move to the input file directory
    os.chdir(outdir)
    # Run DFTB
    process = sp.Popen(dftb_path, stdout=sp.PIPE)

    # Come back to initial directory
    os.chdir(initdir)
    # Obtain energy
    output, error = process.communicate()
    outputStr = output.decode("utf-8").split("\n")
    out = [s for s in outputStr if 'Total Energy' in s]
    E = [float(s) for s in out[0].split(" ") if s.replace(".", "", 1).replace("-", "", 1).isdigit()]
    # Remove the temporary directory
    shutil.rmtree(outdir)

    return E[0]
