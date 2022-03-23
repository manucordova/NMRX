###################################################################################################
#####                                                                                         #####
#####                        Functions to run DFTB energy computations                        #####
#####                     Author: Manuel Cordova (manuel.cordova@epfl.ch)                     #####
#####                                Last modified: 23.03.2022                                #####
#####                                                                                         #####
###################################################################################################

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
        try:
            k_points.append(max(1, int(1 / lattice[i] / factor)))
        except:
            k_points.append(1)
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



def make_dftb_input(xyz, periodic, skfdir, outdir, dispersion, k_points_factor=None, comp_type="sp", driver="SD", relax_elems="all", max_steps=10000, SCC=True, DFTB3=False, showForces=False):
    """
    Generate DFTB+ input file

    Inputs:     - xyz                   Input structure
                - periodic              Whether the structure is periodic or not
                - skfdir                Directory for the DFTB parameter files
                - outdir                Output directory
                - dispersion            Type of dispersion correction to use
                - k_points_factor       Factor for the generation of k-points
                - comp_type             Type of computation to run
                - driver                Driver for relaxation
                - relax_elems           Elements to relax
                - SCC                   Use second-order DFTB
                - DFTB3                 Use third-order DFTB
    """

    pos = xyz.get_positions()
    typ = xyz.get_chemical_symbols()
    elements = []
    ntyp = 0
    for atom in typ:
        if atom not in elements:
            elements.append(atom)
            ntyp += 1

    if periodic:
        abc = xyz.get_cell_lengths_and_angles()
        cell = xyz.get_cell()

    pp = "Geometry = {\n"
    pp += " TypeNames = { "
    for elem in elements:
        pp += "\"" + elem + "\" "
    pp += "}\n TypesAndCoordinates [Angstrom] = {\n"
    for i, position in enumerate(pos):
        pp += "  " + str(elements.index(typ[i])+1) + " " + str(round(position[0],6)) + " " + str(round(position[1],6)) + " " + str(round(position[2],6)) + "\n"
    pp += " }\n"

    if periodic:
        pp += " Periodic = Yes\n"

        pp += " LatticeVectors [Angstrom] = {\n"
        for k in range(3):
            for x in range(3):
                pp += "  " + '{:8.6f}'.format(cell[k][x])
            pp +='\n'
        pp += " }\n"
    pp += "}\n"

    pp += "Hamiltonian = DFTB {\n"
    if SCC:
        pp += " SCC = Yes\n"
        pp += " MaxSCCIterations = 50\n"
    pp += " MaxAngularMomentum = {\n"
    for elem in elements:
        if elem == "H":
            pp += "  " + elem + " = \"s\"\n"
        elif elem in ["C", "N", "O"]:
            pp += "  " + elem + " = \"p\"\n"
        elif elem in ["S", "P"]:
            pp += "  " + elem + " = \"d\"\n"
        else:
            raise ValueError("No maximum angular momentum set for {}.".format(elem))
    pp += " }\n"

    pp += " SlaterKosterFiles = {\n"
    for elem1 in elements:
        for elem2 in elements:
            pp += "  " + elem1 + "-" + elem2 + " = \"" + skfdir + elem1 + "-" + elem2 + ".skf\"\n"
    pp += " }\n"

    if periodic:
        k_points = compute_k_points(abc[:3], k_points_factor)
        pp += " KPointsAndWeights = SupercellFolding {\n"
        pp += "  " + str(k_points[0]) + " 0 0\n"
        pp += "  0 " + str(k_points[1]) + " 0\n"
        pp += "  0 0 " + str(k_points[2]) + "\n"
        s_values = compute_s_values(k_points)
        pp += "  " + str(round(s_values[0],1)) + " " + str(round(s_values[1],1)) + " " + str(round(s_values[2],1)) + "\n"
        pp += " }\n"

    if DFTB3:
        for e in elements:
            if e not in ["H", "C", "N", "O", "S", "P"]:
                raise ValueError("No Hubbard derivative for element {}. Cannot use third-order correction.".format(e))
        pp += " ThirdOrder = Yes\n"
        pp += " HubbardDerivs {\n"
        if "H" in elements:
            pp += "  H = -0.1857\n"
        if "C" in elements:
            pp += "  C = -0.1492\n"
        if "N" in elements:
            pp += "  N = -0.1535\n"
        if "O" in elements:
            pp += "  O = -0.1575\n"
        if "S" in elements:
            pp += "  S = -0.11\n"
        if "P" in elements:
            pp += "  P = -0.14\n"
        pp += " }\n"

    if dispersion == "D3H5":
        pp += " HCorrection = H5 {}\n"
        pp += " Dispersion = DftD3 {\n"
        pp += "  Damping = ZeroDamping {\n"
        pp += "    sr6 = 1.25\n"
        pp += "    alpha6 = 29.61\n"
        pp += "  }\n"
        pp += "  s6 = 1.0\n"
        pp += "  s8 = 0.49\n"
        pp += "  HHRepulsion = Yes\n"
        pp += " }\n"
    elif dispersion == "LJ":
        pp += " Dispersion = LennardJones {\n"
        pp += "  Parameters = UFFParameters {}\n"
        pp += " }\n"
    elif dispersion != "None":
        raise ValueError("Unknown dispersion")

    pp += "}\n"

    if "relax" in comp_type:
        pp += "Driver {\n"
        if driver == "SD":
            pp += " SteepestDescent{\n"
        elif driver == "CG":
            pp += " ConjugateGradient{\n"
        elif driver == "gDIIS":
            pp += " gDIIS{\n"
        elif driver == "LBFGS":
            pp += " LBFGS{\n"
        else:
            raise ValueError("Unknown driver!")
        pp += "  MaxSteps = {}\n".format(max_steps)
        pp += "  OutputPrefix = relax\n"

        if isinstance(relax_elems, str):
            if relax_elems != "all":
                raise ValueError("Enter a list of elements to relax")
        else:
            pp += "  Constraints = {\n"
            for elem in relax_elems:
                if elem not in elements:
                    print("WARNING: Atomic type {} not present in the molecule {}!".format(elem, file))
            for i, atom in enumerate(typ):
                if atom not in relax_elems:
                    pp += "   {} 1.0 0.0 0.0\n".format(i+1)
                    pp += "   {} 0.0 1.0 0.0\n".format(i+1)
                    pp += "   {} 0.0 0.0 1.0\n".format(i+1)
            pp += "  }\n"

        if comp_type == "vc-relax":
            pp += "  LatticeOpt = Yes\n"
        elif comp_type != "relax":
            raise ValueError("Unknown computation type!")

        pp += " }\n"

        pp += "}\n"

    if showForces:
        pp += "Analysis {\n"
        pp += " CalculateForces = Yes\n"
        pp += "}\n"

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    with open(outdir + "/dftb_in.hsd", "w") as f:
        f.write(pp)
    return



def dftbplus_energy(directory, struct, dftb_path, dispersion="D3H5"):
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
    make_dftb_input(struct, True, skfdir, outdir, dispersion, k_points_factor=k_points_factor, comp_type="sp", SCC=True, DFTB3=True)

    # Move to the input file directory
    os.chdir(outdir)

    # Run DFTB and get output
    output = sp.run([dftb_path], capture_output=True)
    outputStr = output.stdout.decode("utf-8").split("\n")

    # Come back to initial directory
    os.chdir(initdir)
    # Obtain energy
    #output, error = process.communicate()
    out = [s for s in outputStr if 'Total Energy' in s]
    # Remove the temporary directory
    shutil.rmtree(outdir)
    try:
        E = [float(s) for s in out[0].split(" ") if s.replace(".", "", 1).replace("-", "", 1).isdigit()]
    except:
        print("Energy computation failed!")
        return 0

    return E[0]



def dftbplus_forces(directory, struct, dftb_path, dispersion="D3H5"):
    """
    DFTB+ single-point energy and forces computation

    Inputs:     - directory     F
                - struct
                - dftb_path     Path for the DFTB program
                - dispersion    Type of dispersion correction to use

    Outputs:    - E     F
    Options:    struct = ase.Atoms object. Crystal structure to compute the energy for
                fileformat = "cif", "xyz", ... (any format valid for ASE)
                dispersion = "D3", "LJ", "None"
    """

    # Initialize array of forces
    forces = np.zeros((len(struct), 3))
    # Initialize stress tensor
    stress = np.zeros((3,3))

    # Create temporary directory to store the DFTB files
    number = str(np.random.random())
    if not os.path.exists(directory+number+"tmp/"):
        os.mkdir(directory+number+"tmp/")
    initdir = os.getcwd()
    outdir = os.path.abspath(directory+number+"tmp/")
    skfdir = os.path.abspath("../src/dftb_sk_files/") + "/"

    k_points_factor = 0.05  # Length for k-point sampling in reciprocal space (A^-1)

    # Generate DFTB input file
    make_dftb_input(struct, True, skfdir, outdir, dispersion, k_points_factor=k_points_factor, comp_type="sp", SCC=True, DFTB3=True, showForces=True)

    # Move to the input file directory
    os.chdir(outdir)

    # Run DFTB and get output
    output = sp.run([dftb_path], capture_output=True)
    outputStr = output.stdout.decode("utf-8").split("\n")

    with open(outdir + "/detailed.out", "r") as F:
        lines = F.read().split("\n")

    for i, l in enumerate(lines):
        if "Total Forces" in l:
            for j in range(len(struct)):
                forces[j] = np.array([float(x) for x in lines[i+j+1].split()[1:]])

        if "Total stress tensor" in l:
            for j in range(3):
                stress[j] = np.array([float(x) for x in lines[i+j+1].split()])

    # Come back to initial directory
    os.chdir(initdir)
    # Obtain energy
    #output, error = process.communicate()
    out = [s for s in outputStr if 'Total Energy' in s]



    # Remove the temporary directory
    shutil.rmtree(outdir)
    try:
        E = [float(s) for s in out[0].split(" ") if s.replace(".", "", 1).replace("-", "", 1).isdigit()]
    except:
        print("Energy computation failed!")
        return 0

    return E[0], forces, stress



def dftb_relax(directory, struct, dftb_path, n_opt, elems=["H"], dispersion="D3H5"):
    """
    DFTB+ relaxation

    Inputs:     - directory     Root directory for the creation of the input and output files
                - struct        Crystal to relax
                - dftb_path     Path for the DFTB program
                - dispersion    Type of dispersion correction to use

    Outputs:    - out_struct    Relaxed structure
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
    make_dftb_input(struct, True, skfdir, outdir, dispersion, k_points_factor=k_points_factor, comp_type="relax", driver="gDIIS", relax_elems=elems, max_steps=n_opt, SCC=True, DFTB3=True)

    # Move to the input file directory
    os.chdir(outdir)

    # Run DFTB and get output
    output = sp.run([dftb_path], capture_output=True)

    # Get relaxed crystal
    out_struct = ase.io.read(outdir + "/relax.gen")

    # Come back to the initial directory
    os.chdir(initdir)
    # Remove the temporary directory
    shutil.rmtree(outdir)

    return out_struct



def compute_distance_constraints(struct, n_atoms, pairs, thresh=5., exponent=2., contact=False, c_type="avg"):
    """
    Compute the cost associated with the selected distance constraints

    Inputs:     - struct        Input structure
                - n_atoms       Number of atoms in a single molecule
                - pairs         Pairs of atoms to compute the constraints for (should refer to the atoms in the
                                    first molecule, although the constraints will be evaluated both for intramolecular
                                    and intermolecular contacts)
                - thresh        Threshold for distance constraints. Can be a single number or an array of numbers.
                - exponent      Exponent to raise the computed costs to.
                - contact       Whether the penalty should be incurred for larger (True) or smaller (False) distance
                                    than the threshold. Can be a single boolean or an array of booleans or indices.
                - c_type        Type of constraint (sum == sum, avg == average, rmsd == root-mean-square deviation
                                    [overrides the exponent variable])
    """

    # Get number of molecules in the unit cell
    symbs = struct.get_chemical_symbols()
    n_mol = int(len(symbs)/n_atoms)

    min_ds = []
    # For each pair of atoms set
    for p in pairs:
        ds = []
        # Get all possible distances (intra- and intermolecular)
        js = []
        for j in range(n_mol):
            js.append(p[1]+(j*n_atoms))
        ds.extend(struct.get_distances(p[0], js, mic=True))
        if len(ds) < 1:
            raise ValueError("Error when computing the distance for pair {}-{}".format(p[0], p[1]))

        # Get minimum distance
        min_ds.append(np.min(ds))

    # Retract the threshold from the minimum distance
    min_ds = np.array(min_ds)
    min_ds -= thresh

    # Invert the sign if a contact is expected
    if type(contact) == bool:
        if contact:
            min_ds *= -1.
    elif type(contact) == list:
        min_ds[contact] *= -1.
    else:
        raise ValueError("The variable contact should be a boolean or a list, not {}".format(type(contact)))

    # Set negative values to zero (constraint is fulfilled)
    min_ds[min_ds < 0] = 0

    # Return the corresponding cost
    if c_type == "sum":
        return np.sum(np.power(min_ds, exponent))
    elif c_type == "avg":
        return np.mean(np.power(min_ds, exponent))
    elif c_type == "rmsd":
        return np.sqrt(np.mean(np.square(min_ds)))
    else:
        raise ValueError("Unknown constraint type: {}".format(c_type))
