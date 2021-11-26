#####
# Last modified 01.07.2020
# Last update: Included smart cell generation
# Authors: Manuel Cordova, Martins Balodis
#####



### Import libraries
import numpy as np
import ase
import ase.data
import ase.spacegroup
import copy
import time



def check_valid_angles(alpha, beta, gamma):
    """
    Check if the three angles for the construction of the unit cell are valid

    Inputs:     - alpha     First cell angle (in degrees)
                - beta      Second cell angle (in degrees)
                - gamma     Third cell angle (in degrees)

    Output:     - valid     Whether the angles are valid or not
    """

    al = np.deg2rad(alpha)
    be = np.deg2rad(beta)
    ga = np.deg2rad(gamma)

    cx2 = np.square(np.cos(be))
    cy2 = np.square((np.cos(al) - np.cos(be)*np.cos(ga))/np.sin(ga))

    return 1 - cx2 - cy2 > 0.



def molecular_volume(symbs, n_mol):
    """
    Obtain the volume taken by all atoms in the molecule (sum of VDW spheres)

    Inputs:     - symbs     Symbols of the molecule
                - n_mol     Number of molecules in the unit cell

    Outputs:    - V_mol     Volume of all atoms in the molecule
    """

    V_mol = 0
    # Add VdW radii to estimate the volume of the molecule
    for s in symbs:
        V_mol += (ase.data.vdw_radii[ase.data.atomic_numbers[s]] ** 3)

    return V_mol * 4./3. * np.pi * n_mol



def write_gas_phase_conformer(struct, n_atoms, path):
    """
    Write the gas phase conformer of a molecule to XYZ format

    Inputs:     - struct    Structure to write
                - n_atoms   Number of atoms in the molecule
                - path      Path to write the file to

    Outputs:    None
    """

    # Get chemical symbols and atomic positions
    symbs = struct.get_chemical_symbols()
    pos = struct.get_positions()
    # Set the two first line
    pp = "{}\nSingle molecule\n".format(n_atoms)
    # Set the atomic types and positions
    for s, p in zip(symbs[:n_atoms], pos[:n_atoms]):
        pp += "{}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(s, p[0], p[1], p[2])
    # Write the XYZ file
    with open(path, "w") as f:
        f.write(pp)

    return



def distance(x0, x1, abc, pbc=False):
    """
    Obtain distances

    Inputs:     - x0    First coordinate
                - x1    Array of coordinates
                - abc   Periodic boundary conditions

    Outputs:    - d     Distances between x0 and each point in x1
    """

    delta = x0-x1
    if pbc:
        # Loop over all PBC vectors
        for a in abc:
            # Get the norm of the PBC vector
            norm = np.linalg.norm(a)

            for i in range(len(delta)):
                x = delta[i].T.dot(a)/norm
                while x > 0.5:
                    delta[i] -= a
                    x = delta[i].T.dot(a)/norm
                while x < -0.5:
                    delta[i] += a
                    x = delta[i].T.dot(a)/norm

    return np.sqrt((delta**2).sum(axis=-1))



def get_distances(struct, inds1, inds2, mic=False, vector=False):
    """
    Compute the distance matrix between two sets of atoms in a structure.

    Inputs: - struct    Input structure
            - inds1     First set of atom indices
            - inds2     Second set of atom indices
            - mic       Apply the minimal image convention (for periodic systems)
            - vector    Return the vectorial distance matrix

    Output: - D         Distance matrix
    """

    # Get atomic positions
    pos = struct.get_positions()

    # Compute distance matrix
    D = pos[np.newaxis, inds2, :] - pos[inds1, np.newaxis, :]

    # Apply MIC
    if mic:
        # Get PBC vectors
        abc = struct.get_cell()

        for a in abc[::-1]:

            x = D.dot(a) / a.T.dot(a)

            D[x > 0.5] -= a
            D[x <= -0.5] += a

    # Convert to scalar distances
    if not vector:
        D = np.linalg.norm(D, axis=-1)

    return D



def check_clash(structure, n_atoms, pbc=True, clash_type="intra", factor=0.85):
    """
    Check whether clashes are found in the molecule or crystal

    Inputs:     - structure     Structure
                - n_atoms       Number of atoms to consider
                - pbc           Whether to take periodicity into account or not
                - clash_type    Whether check clashes for atoms in the same molecule or
                                    in different molecules
                - factor        If atoms are less than factor * (sum of covalent radii) apart,
                                    a clash is detected

    Outputs:    - True/False    True if a clash is detected, false otherwise


    """

    if clash_type not in ["intra", "inter"]:
        raise ValueError("Unknown clash type: {}".format(clash_type))

    # Get atomic species and positions, and unit cell vectors
    symbs = structure.get_chemical_symbols()
    pos = structure.get_positions()
    abc = structure.get_cell()
    # Set reference length for every pair of elements
    elems = np.unique(symbs)
    contacts = {}
    max_contact = 0.
    for e1 in elems:
        for e2 in elems:
            contacts["{}-{}".format(e1, e2)] = (ase.data.covalent_radii[ase.data.atomic_numbers[e1]]+ase.data.covalent_radii[ase.data.atomic_numbers[e2]]) * factor
            if contacts["{}-{}".format(e1, e2)] > max_contact:
                max_contact = contacts["{}-{}".format(e1, e2)]

    if clash_type == "intra":
        D = get_distances(structure, range(n_atoms), range(n_atoms), mic=pbc)

        for i in range(n_atoms-1):
            for j in np.where(D[i, i+1:] < max_contact)[0]:
                if D[i,i+j+1] < contacts[f"{symbs[i]}-{symbs[i+j+1]}"]:
                    return True

    if clash_type == "inter":
        D = get_distances(structure, range(n_atoms), range(n_atoms, len(structure)))

        for i in range(n_atoms):
            for j in np.where(D[i] < max_contact)[0]:
                if D[i, j] < contacts[f"{symbs[i]}-{symbs[n_atoms+j]}"]:
                    return True

    return False



def constraint_violations(struct, n_atoms, pairs, thresh=5., contact=True, pbc=False):
    """
    Check a structure for distance constraint violation

    Inputs: - struct    Input structure
            - n_atoms   Number of atoms in a single molecule
            - pairs     Pairs of atoms to check for constraint violation
            - thresh    Distance threshold for contact
            - contact   Whether a violation should be identified for larger (True) or smaller (False) distance
                            than the threshold. Can be a single boolean or an array of booleans or indices.
            - pbc       Whether periodic boundary conditions should be taken into account

    Output: - n         Number of violated constraints
    """

    # Initialize the number of violated constraints
    n = 0

    # Get number of molecules in the unit cell
    symbs = struct.get_chemical_symbols()
    n_mol = int(len(symbs)/n_atoms)

    for p in pairs:
        if not pbc:
            d = struct.get_distance(p[0], p[1], mic=False)
            if type(contact) == bool:
                if contact:
                    if d > thresh:
                        n += 1
                elif d < thresh:
                    n += 1
            else:
                print("Array of contact booleans not implemented yet.")
        else:
            print("Constraint violations with pbc not implemented yet.")

    return n



def generate_conformer(starting_structure, n_atoms, n_conf, conf_params, constraints=None, verbose=False):
    """
    Generate a conformaer

    Inputs:     - starting_structure    Initial structure
                - n_atoms               Number of atoms in the molecule
                - n_conf                Number of dihedral angles to vary
                - conf_params           Atoms and masks of the dihedral angles
                - constraints           Dictionary of constraint parameters for conformer selection
                - verbose               Verbosity level

    Outputs:    - conf_angles           Array of conformational angles
    """

    if verbose:
        print("Generating conformer...")
        start = time.time()

    # Check that atoms in the dihedral angles and masks are valid
    for i in range(n_conf):
        if conf_params["a1"][i] not in conf_params["mask"][i] and conf_params["a2"][i] not in conf_params["mask"][i] and conf_params["a3"][i] not in conf_params["mask"][i] and conf_params["a4"][i] not in conf_params["mask"][i]:
            raise ValueError("No atom from the diheral angle {} in the mask.".format(i+1))

    clash = True
    conf_angles = [0 for _ in range(n_conf)]
    # As long as clashes between atoms are found, generate a new conformation
    N = 0
    while clash:
        N += 1
        xyz = copy.deepcopy(starting_structure)
        for i in range(n_conf):
            # Generate random dihedral angle
            conf_angles[i] = np.random.random()*360.
            mask = np.array([1 if j in conf_params["mask"][i] else 0 for j in range(len(xyz))])
            # Change conformation according to the generated angle
            xyz.set_dihedral(a1=conf_params["a1"][i], a2=conf_params["a2"][i], a3=conf_params["a3"][i], a4=conf_params["a4"][i], angle=conf_angles[i], mask=mask)
        # Check clashes for the conformer (single molecule)
        clash = check_clash(xyz, n_atoms, pbc=False, clash_type="intra", factor=0.85)

        if constraints is not None:
            n_viol = constraint_violations(xyz, n_atoms, constraints["pairs"], constraints["thresh"], constraints["contact"], pbc=False)
            if n_viol > constraints["n_tol"]:
                print("  The generated conformer violated set distance constraints! {}/{}".format(n_viol, constraints["n_tol"]))
                clash = True

    if verbose:
        stop = time.time()
        dt = stop-start
        print("Conformer successfully generated after {} tries! ({:.2f} s elapsed)".format(N, dt))

    return conf_angles



def rotation_matrix(rot):
    """
    Generate the rotation matrix from a vector containing the direction and the angle of the rotation.

    Inputs:     - rot       Rotation vector: vector of length 4 containing the direction and
                                angle (in degrees) of the rotation.

    Outputs:    - R         Corresponding rotation matrix
    """

    # Generate components of the rotation vector and angles
    vx = rot[0]
    vxx = rot[0]**2
    vxy = rot[0]*rot[1]
    vxz = rot[0]*rot[2]
    vy = rot[1]
    vyy = rot[1]**2
    vyz = rot[1]*rot[2]
    vz = rot[2]
    vzz = rot[2]**2
    c = np.cos(rot[3]/180*np.pi)
    s = np.sin(rot[3]/180*np.pi)

    # Generate the rotation matrix
    R = np.zeros((3,3))
    R[0,0] = c+vxx*(1-c)
    R[0,1] = vxy*(1-c)-vz*s
    R[0,2] = vxz*(1-c)+vy*s
    R[1,0] = vxy*(1-c)+vz*s
    R[1,1] = c+vyy*(1-c)
    R[1,2] = vyz*(1-c)-vx*s
    R[2,0] = vxz*(1-c)-vy*s
    R[2,1] = vyz*(1-c)+vx*s
    R[2,2] = c+vzz*(1-c)

    return R



def rotate_molecule(struct, R, center, n_atoms, centred=False):
    """
    Rotate the molecule with the given center and rotation matrix.

    Inputs:     - struct    Initial structure
                - R         Rotation matrix
                - center    Center around which the rotation should be done
                - n_atoms   Number of atoms to rotate
                - centred   Set to True if the molecule should end up centred around the origin

    Outputs:    - struct    rotated structure
    """

    # Extract input atomic positions
    xyz = struct.get_positions()
    # Center the position, apply rotation, and move positions back (if centred is False)
    for i in range(n_atoms):
        xyz[i] -= center
        xyz[i] = R.dot(xyz[i])
        if not centred:
            xyz[i] += center

    # Update the crystal structure
    struct.set_positions(xyz)

    return struct



def get_unit_cell_vectors(lat):
    """
    Obtain unit cell vectors from unit cell lengths and angles

    Inputs:     - lat       Unit cell parameters

    Outputs:    - a         First unit cell vector
                - b         Second unit cell vector
                - c         Third unit cell vector
    """

    tmp = ase.Atoms()
    tmp.set_cell(lat)
    abc = tmp.get_cell()

    return abc



def get_projections(struct, lat, n_atoms, R):
    """
    Get projections of the molecule along the unit cell vectors

    Inputs:     - struct    Structure
                - lat       Lattice parameters (cell lengths and angles)
                - n_atoms   Number of atoms in the molecule
                - R         Rotation matrix

    Outputs:    - la        Length along unit cell vector a
                - lb        Length along unit cell vector b
                - lc        Length along unit cell vector c
    """

    # Get unit cell vectors
    tmp = ase.Atoms()
    tmp.set_cell(lat)
    abc = tmp.get_cell()

    # Normalize unit cell vectors
    unit_a = abc[0]/np.linalg.norm(abc[0])
    unit_b = abc[1]/np.linalg.norm(abc[1])
    unit_c = abc[2]/np.linalg.norm(abc[2])

    # Rotate molecule
    s = copy.deepcopy(struct)
    center = s[:n_atoms].get_center_of_mass()
    s = rotate_molecule(s, R, center, n_atoms, centred=True)
    xyz = s.get_positions()[:n_atoms]
    symbs = s.get_chemical_symbols()[:n_atoms]

    # Initialize projections vectors
    proj_a = np.zeros(n_atoms)
    proj_b = np.zeros(n_atoms)
    proj_c = np.zeros(n_atoms)

    # Get projections of each atom
    for i, pos in enumerate(xyz):
        proj_a[i] = pos.T.dot(unit_a)
        proj_b[i] = pos.T.dot(unit_b)
        proj_c[i] = pos.T.dot(unit_c)

    # Get the max distances along the projections
    la = np.max(proj_a)-np.min(proj_a)
    lb = np.max(proj_b)-np.min(proj_b)
    lc = np.max(proj_c)-np.min(proj_c)

    # Add covalent radii to projections
    la += ase.data.covalent_radii[ase.data.atomic_numbers[symbs[np.argmax(proj_a)]]]
    la += ase.data.covalent_radii[ase.data.atomic_numbers[symbs[np.argmin(proj_a)]]]
    lb += ase.data.covalent_radii[ase.data.atomic_numbers[symbs[np.argmax(proj_b)]]]
    lb += ase.data.covalent_radii[ase.data.atomic_numbers[symbs[np.argmin(proj_b)]]]
    lc += ase.data.covalent_radii[ase.data.atomic_numbers[symbs[np.argmax(proj_c)]]]
    lc += ase.data.covalent_radii[ase.data.atomic_numbers[symbs[np.argmin(proj_c)]]]

    return [la, lb, lc]



def get_cell_volume(lat):
    """
    Compute the volume of a unit cell given its lengths and angles

    Inputs:     - lat       Lattice

    Outputs:    - V         Volume of the unit cell
    """

    cell = ase.Atoms(cell=lat)

    return cell.get_volume()



def get_third_length_vol(a, b, alpha, beta, gamma, Vmol):
    """
    Determine the third length of a cell to correspond to the molecular volume

    Inputs:     - a         First cell length
                - b         Second cell length
                - alpha     First cell angle
                - beta      Second cell angle
                - gamma     Third cell angle
                - Vmol      Volume of the molecule

    Outputs:    - c         Third cell length
    """
    sqroot = 1+(2*np.cos(np.deg2rad(alpha))*np.cos(np.deg2rad(beta))*np.cos(np.deg2rad(gamma)))
    sqroot += -np.cos(np.deg2rad(alpha))**2-np.cos(np.deg2rad(beta))**2-np.cos(np.deg2rad(gamma))**2
    sqroot = np.sqrt(sqroot)
    return Vmol / (a*b*sqroot)



def translate_molecule(struct, trans, n_atoms, lattice=[]):
    """
    Translate a molecule by the vector trans. The vector can be expressed in XYZ, or in scaled coordinates of the lattice vectors.

    Inputs:     - struct        Structure to translate
                - trans         Vector of XZY or scaled coordinates to translate by
                - n_atoms       Number of atoms to translate
                - lattice       Either None, translation vector is thus XYZ, or corresponds to
                                    cell lengths and angles and translation vector is given
                                    in scaled coordinates.

    Outputs:    - struct        Translated structure
    """

    # Get atomic positions
    xyz = struct.get_positions()
    if len(lattice) == 0:
        for i in range(n_atoms):
            xyz[i] += trans
    else:
        abc = get_unit_cell_vectors(lattice)
        for t, v in zip(trans, abc):
            for i in range(n_atoms):
                xyz[i] += t*v
    struct.set_positions(xyz)
    return struct



def generate_random_unit_vector():
    v = np.zeros(3)
    v[0] = np.random.normal(loc=0., scale=1.)
    v[1] = np.random.normal(loc=0., scale=1.)
    v[2] = np.random.normal(loc=0., scale=1.)
    return v / np.linalg.norm(v)



def generate_random_rot(A):
    """
    Generate a random rotation and output the corresponding rotation matrix

    Inputs:     - A     Amplitude of the rotation in degrees

    Outputs:    - R     Rotation matrix
    """
    rot = np.zeros(4)
    # Select a random direction
    rot[:3] = generate_random_unit_vector()
    rot[:3] /= np.linalg.norm(rot[:3])
    # Select a random angle
    rot[3] = (np.random.random()-0.5)*A
    return rotation_matrix(rot)



def reorder_atoms(struct, n_atoms):
    """
    Reorder the atoms in a crystal to sequential

    Inputs:     - struct    Input structure (from the ase cell generation)
                - n_atoms   Number of atoms in the molecule

    Outputs:    - struct    Reordered structure
    """

    # Get atomic symbols and positions
    symbs = struct.get_chemical_symbols()
    pos = struct.get_positions()

    # Get number of molecules in the unit cell
    n_mol = int(len(symbs)/n_atoms)
    if n_mol != len(symbs)/n_atoms:
        raise ValueError("Number of molecules is not an integer! (:.2f)".format(len(symbs)/n_atoms))

    # Reorder the atoms
    new_symbs = []
    new_pos = []
    for i in range(n_mol):
        for j in range(n_atoms):
            ind = i + n_mol*j
            new_symbs.append(symbs[ind])
            new_pos.append(pos[ind])

    struct.set_chemical_symbols(new_symbs)
    struct.set_positions(new_pos)

    return struct



def create_crystal(starting_structure, lat, trans, R, conf_angles, sg, n_atoms, conf_params=None):
    """
    Create a crystal structure from a list of parameters and an initial structure

    Inputs:     - trial_structure   Initial structure
                - conf_angles       Conformational angles to set
                - trans             Translation to apply
                - R                 Rotation matrix to rotate the molecule
                - lat               Lattice parameters
                - sg                Space group
                - n_atoms           Number of atoms in the molecule
                - conf_params       Atoms and masks for changing the conformation

    Outputs:    - trial_crystal     Crystal structure generated
    """

    # Copy crystal structure to modify it
    trial_structure = copy.deepcopy(starting_structure)

    # If the angles are not valid
    if not check_valid_angles(lat[3], lat[4], lat[5]):
        return None, True

    # Set conformation
    for i, a in enumerate(conf_angles):
        # Set mask for the conformational change
        mask = np.array([1 if j in conf_params["mask"][i] else 0 for j in range(len(trial_structure))])
        # Change conformation according to the generated angle
        trial_structure.set_dihedral(a1=conf_params["a1"][i], a2=conf_params["a2"][i], a3=conf_params["a3"][i], a4=conf_params["a4"][i], angle=a, mask=mask)

    # Rotate molecule and center it to the origin
    center = trial_structure[:n_atoms].get_center_of_mass()
    centred = True
    if np.linalg.norm(trans) == 0.:
        centred = False
    trial_structure = rotate_molecule(trial_structure, R, center, n_atoms, centred=centred)

    # Translate molecule
    trial_structure = translate_molecule(trial_structure, trans, n_atoms, lattice=lat)

    # Get atomic symbols and positions
    symbs = trial_structure.get_chemical_symbols()[:n_atoms]
    trial_structure.set_cell(lat, scale_atoms=False)
    scaled_pos = trial_structure.get_scaled_positions(wrap=False)[:n_atoms]

    # Generate the crystal structure
    clash = False
    try:
        trial_crystal = ase.spacegroup.crystal(symbols=symbs, basis=scaled_pos, spacegroup=sg, cellpar=lat, onduplicates="error", symprec=0.001, pbc=True)
        # Reorder the atoms in the crystal
        trial_crystal = reorder_atoms(trial_crystal, n_atoms)
    except:
        clash = True
        trial_crystal = ase.spacegroup.crystal(symbols=symbs, basis=scaled_pos, spacegroup=sg, cellpar=lat, onduplicates="keep", symprec=0.001, pbc=True)

    return trial_crystal, clash



def generate_crystal(starting_structure, n_atoms, n_mol, sg, parameter_set, cell_params_lims, n_conf=0, conf_params=None, smart_cell=True, max_V_factor=None, constraints=None, verbose=False):
    """
    Generate an initial crystal structure

    Inputs:     - starting_structure    Initial structure
                - parameter_set         Set of parameters to change
                - n_atoms               Number of atoms in the molecule
                - n_mol                 Number of molecules in the unit cell
                - sg                    Space group of the crystal
                - n_conf                Number of dihedral angles to vary
                - conf_params           Atoms and masks of the dihedral angles
                - constraints           Dictionary of constraint parameters for conformer selection
                - verbose               Verbosity level

    Outputs:    - TODO
    """

    # Get molecular volume
    symbs = starting_structure.get_chemical_symbols()[:n_atoms]
    V_mol = molecular_volume(symbs, n_mol)

    n_failed = 0

    # Generate conformer. Trial conformers are generated until no intramolecular clash is detected
    if "conf" in parameter_set:
        if n_conf == 0 or not conf_params:
            raise ValueError("No dihedral angle set.")
        conf_angles = generate_conformer(starting_structure, n_atoms, n_conf, conf_params, constraints=constraints, verbose=verbose)
    else:
        conf_angles = []

    # Generate all other parameters until no intermolecular clash is detected
    if verbose:
        start = time.time()
    clash = True
    N = 0
    while clash:
        N += 1

        # Rotate molecule randomly
        if "rot" in parameter_set:
            # Generate random rotation matrix
            R = generate_random_rot(360.)
        else:
            rot = np.array([1., 0., 0., 0.])
            R = rotation_matrix(rot)

        # Translate molecule
        trans = np.zeros(3)
        if "trans" in parameter_set:
            for i in range(3):
                # Translation vector expressed in fractions of the unit cell vectors
                trans[i] = np.random.random()

        # Get cell from initial structure
        lat = starting_structure.get_cell_lengths_and_angles()

        # Generate new angles if they should be varied, generate new angles until the cell can be constructed without issue
        valid_angles = False
        while not valid_angles:
            if "alpha" in parameter_set:
                lat[3] = cell_params_lims[2] + (cell_params_lims[3] - cell_params_lims[2]) * np.random.random()
            if "beta" in parameter_set:
                lat[4] = cell_params_lims[2] + (cell_params_lims[3] - cell_params_lims[2]) * np.random.random()
            if "gamma" in parameter_set:
                lat[5] = cell_params_lims[2] + (cell_params_lims[3] - cell_params_lims[2]) * np.random.random()

            # Check that the angles generated are valid
            valid_angles = check_valid_angles(lat[3], lat[4], lat[5])

        # Generate cell lengths
        if smart_cell:
            if "a" not in parameter_set or "b" not in parameter_set or "c" not in parameter_set:
                raise ValueError("Smart cell generation can only be done when varying all cell lengths!")
            if not max_V_factor:
                raise ValueError("No maximum volume factor set!")
            # Set cell lengths to an impossible value
            for i in range(3):
                lat[i] = -1.
            # Get projctions of the molecule onto the unit cell axes
            low_lengths = get_projections(starting_structure, lat, n_atoms, R)
            # Set the two (randomly selected) first lengths, randomly between the projection and the projection times the number of molecules in the structure
            first_two_lengths = []
            for i in np.random.permutation(range(3))[:2]:
                lat[i] = low_lengths[i] * (n_mol - 1) * np.random.random() + low_lengths[i]
                first_two_lengths.append(lat[i])
            # Set the third length depending on the two first lengths and on the maximum volume factor
            for i in range(3):
                if lat[i] < 0:
                    # Get third cell length boundaries to correspond to the volume and volume factor
                    low_length = get_third_length_vol(first_two_lengths[0], first_two_lengths[1], lat[3], lat[4], lat[5], V_mol)
                    high_length = low_length * max_V_factor
                    # Set the third length
                    lat[i] = (high_length-low_length)*np.random.random()+low_length

        else:
            # Set cell lengths randomly between the boundaries
            if "a" in parameter_set:
                lat[0] = cell_params_lims[0] + (cell_params_lims[1] - cell_params_lims[0]) * np.random.random()
            if "b" in parameter_set:
                lat[1] = cell_params_lims[0] + (cell_params_lims[1] - cell_params_lims[0]) * np.random.random()
            if "c" in parameter_set:
                lat[2] = cell_params_lims[0] + (cell_params_lims[1] - cell_params_lims[0]) * np.random.random()

        if verbose:
            V = get_cell_volume(lat)
            print("Volume ratio: {:.2f}".format(V/V_mol))

        # Generate the trial structure
        trial_crystal, clash = create_crystal(starting_structure, lat, trans, R, conf_angles, sg, n_atoms, conf_params=conf_params)

        if not clash:
            # Check the structure for clashes
            clash = check_clash(trial_crystal, n_atoms, pbc=True, clash_type="intra", factor=0.85)
            if not clash:
                clash = check_clash(trial_crystal, n_atoms, pbc=True, clash_type="inter", factor=1.2)

    if verbose:
        stop = time.time()
        dt = stop-start
        print("Initial crystal structure successfully generated after {} tries! ({:.2f} s elapsed)".format(N, dt))

    return trial_crystal, lat, trans, R, conf_angles



def reconstruct_molecule_from_crystal(cryst, n_atoms, thresh=2., cutoff_factor=None):
    """
    Reconstruct a molecule from a crystal

    Inputs: - cryst             Input crystal structure
            - n_atoms           Number of atoms in the molecule
            - thresh            Distance threshold (should be smaller than half of the smallest unit cell length)
            - cutoff_factor     Factor for the tolerance for bond recognition. If set, bypasses the thresh variable

    Output: - mol_pos           Position of the atoms in the reconstructed molecule
    """

    # If we set a cutoff factor, get the distances for each pair of elements in the structure
    if cutoff_factor is not None:
        cutoffs = {}
        symbs = cryst.get_chemical_symbols()
        elems = []
        # Get all elements in the structure
        for s in symbs:
            if s not in elems:
                elems.append(s)

        # For all pair of elements, get the sum of covalent radii and multiply by the factor
        for e1 in elems:
            for e2 in elems:
                cutoffs[(e1, e2)] = (ase.data.covalent_radii[ase.data.atomic_numbers[e1]]+ase.data.covalent_radii[ase.data.atomic_numbers[e2]])*cutoff_factor

    # Initialize positions of the reconstructed molecule
    mol_pos = np.zeros((n_atoms, 3))
    # Set the first position to [0, 0, 0] as reference
    used_indices = [0]
    # While all atoms are not placed:
    while len(used_indices) < n_atoms:
        # Loop over all pairs of atoms in the molecule
        for i in range(n_atoms):
            for j in range(n_atoms):
                # If one atom in the pair is already set and not the other, get the vector between these two atoms
                if i in used_indices and j not in used_indices:
                    v = cryst.get_distance(i, j, mic=True, vector=True)
                    # If the distance is smaller than the threshold, add the position of atom j to the reconstructed structure
                    if cutoff_factor is not None:
                        if np.linalg.norm(v) < thresh:
                            mol_pos[j] = mol_pos[i] + v
                            used_indices.append(j)
                    else:
                        if np.linalg.norm(v) < cutoffs[(symbs[i], symbs[j])]:
                            mol_pos[j] = mol_pos[i] + v
                            used_indices.append(j)

    return mol_pos



def optimize_rot_rmsd(pos, ref_pos, align=None):
    """
    Get the positional RMSD between two sets of atomic positions, by optimizing the rotation between them. Makes use of the Kabsch algorithm.

    Inputs:     - pos       Atomic positions to obtain the RMSD for
                - ref_pos   Reference atomic positions
                - align     Indices of the subset of atoms to align

    Outputs:    - rmsd      RMSD between the two structures, after rotation
                - rot_pos   Atomic positions rotated to match ref_pos
    """

    # If a subset of atoms to align is set, construct the P and Q matrices only for them
    if align is not None:
        P = pos[align]
        Q = ref_pos[align]
    # Otherwise, construct the matrices for all positions
    else:
        P = np.copy(pos)
        Q = np.copy(ref_pos)

    # Get covariance matrix, and perform SVD decomposition
    H = P.T.dot(Q)
    U, S, V = np.linalg.svd(H)

    # Get rotation matrix
    d = np.linalg.det(V.T.dot(U.T))
    d /= np.abs(d)
    I = np.eye(3)
    I[2,2] = d
    R = V.T.dot(I.dot(U.T))

    # Rotate the atomic coordinates
    rot_pos = np.zeros_like(pos)
    for i in range(len(pos)):
        rot_pos[i] = R.dot(pos[i])

    # Get the positional RMSD
    rmsd = np.sqrt(np.mean(np.square(np.linalg.norm(rot_pos-ref_pos, axis=1))))

    return rmsd, rot_pos



def get_opt_rot(pos, ref_pos):
    """
    Get the rotation matrix that minimizes the RMSD between two sets of atomic positions. Makes use of the Kabsch algorithm.

    Inputs: - pos       Atomic positions to align
            - ref_pos   Reference atomic positions

    Output: - R         Rotation matrix
    """

    # Construct the P and Q matrices
    P = pos
    Q = ref_pos

    # Get covariance matrix, and perform SVD decomposition
    H = P.T.dot(Q)
    U, S, V = np.linalg.svd(H)

    # Get rotation matrix
    d = np.linalg.det(V.T.dot(U.T))
    d /= np.abs(d)
    I = np.eye(3)
    I[2,2] = d
    R = V.T.dot(I.dot(U.T))

    return R



def pos_rmsd(struct, ref, inds, align=None):
    """
    Obtain the positional RMSD between a structure and a reference.

    Inputs:     - struct    Input structure
                - ref       Reference structure
                - inds      Indices of the atoms for which the RMSD should be computed
                - align     Atoms to align between the input and reference structures

    Outputs:    - rmsd      RMSD of atomic positions between the structure and the reference
                - al_pos    Aligned positions of the input structure
    """

    # Extract positions of the input and reference structures
    pos = struct.get_positions()[inds]
    ref_pos = ref.get_positions()[inds]


    # If alignment should be performed on a subset of the atoms only
    if align is not None:
        # Get the atomic positions to align
        tmp_pos = struct.get_positions()[align]
        tmp_ref_pos = ref.get_positions()[align]

        # Center the atoms to align around the origin (center of mass)
        c = struct[align].get_center_of_mass()
        c_ref = ref[align].get_center_of_mass()

    else:
        # Get the atomic positions to align
        tmp_pos = struct.get_positions()[inds]
        tmp_ref_pos = ref.get_positions()[inds]

        # Center the atoms to align around the origin (center of mass)
        c = struct[inds].get_center_of_mass()
        c_ref = ref[inds].get_center_of_mass()

    c_pos = pos - c
    c_pos_ref = ref_pos - c_ref
    c_tmp_pos = tmp_pos - c
    c_tmp_pos_ref = tmp_ref_pos - c_ref

    R = get_opt_rot(c_tmp_pos, c_tmp_pos_ref)


    # Optimize rotation and get the RMSD
    rot_pos = []
    for p in c_pos:
        rot_pos.append(R.dot(p))
    rot_pos = np.array(rot_pos)

    rmsd = np.sqrt(np.mean(np.square(np.linalg.norm(rot_pos-c_pos_ref, axis=1))))

    # Match the atomic positions of the input structure to the reference
    al_pos = rot_pos + c_ref

    return rmsd, al_pos



def opt_conf(struct, ref, a1, a2, a3, a4, inds_mask, ha_inds, thresh):
    """
    Optimize the angle of one conformer to match a reference.

    Inputs: - struct        Input structure
            - ref           Reference structure
            - a1            First atom in the dihedral
            - a2            Second atom in the dihedral
            - a3            Third atom in the dihedral
            - a4            Fourth atom in the dihedral
            - inds_mask     Indices of the atoms to move when changing the dihedral angle
            - ha_inds       Indices of heavy atoms (to compute the RMSD for)
            - thresh        Threshold for convergence of the step size

    Output: - a             Dihedral angle that best matches the reference structure
    """

    # Generate array of initial dihedral angles
    init_angles = np.linspace(0., 360., 37)[:-1]

    # Get the indices of the atoms to align
    inds_al = [a for a in [a1, a2, a3, a4] if a not in inds_mask]

    # Get the indices of the atoms to compute the RMSD for
    inds_rmsd = [i for i in [a1, a2, a3, a4] if i in ha_inds]
    inds_rmsd.extend([i for i in inds_mask if i not in inds_rmsd and i in ha_inds])

    # Generate the mask for dihedral rotation
    mask = np.array([1 if j in inds_mask else 0 for j in range(len(struct))])

    init_rmsds = []
    # Loop over all the initial angles
    for a in init_angles:
        # Copy the input structure
        tmp_struct = copy.deepcopy(struct)
        # Set the dihedral to the selected angle
        tmp_struct.set_dihedral(a1=a1, a2=a2, a3=a3, a4=a4, angle=a, mask=mask)
        # Compute the positional RMSD with the given angle
        rmsd, _ = pos_rmsd(tmp_struct, ref, inds_rmsd, align=inds_al)
        init_rmsds.append(rmsd)

    # Initialize step size for optimizing the angle
    step = 1.
    # Get the best initial angle
    a = init_angles[np.argmin(init_rmsds)]
    # Loop until the step size is small enough
    while step > thresh:
        change = True
        while change:
            # Take a step in the positive direction
            a_p = a + step
            # Copy the input structure
            tmp_struct = copy.deepcopy(struct)
            # Set the dihedral to the selected angle
            tmp_struct.set_dihedral(a1=a1, a2=a2, a3=a3, a4=a4, angle=a_p, mask=mask)
            # Compute the positional RMSD with the given angle
            rmsd_p, _ = pos_rmsd(tmp_struct, ref, inds_rmsd, align=inds_al)

            # Take a step in the negative direction
            a_m = a - step
            # Copy the input structure
            tmp_struct = copy.deepcopy(struct)
            # Set the dihedral to the selected angle
            tmp_struct.set_dihedral(a1=a1, a2=a2, a3=a3, a4=a4, angle=a_m, mask=mask)
            # Compute the positional RMSD with the given angle
            rmsd_m, _ = pos_rmsd(tmp_struct, ref, inds_rmsd, align=inds_al)

            # If the initial RMSD is the lowest, reduce the step size
            if rmsd < rmsd_p and rmsd < rmsd_m:
                change = False
            # Otherwise, get the new optimal angle and perform the search for a minimum again.
            else:
                if rmsd_m < rmsd_p:
                    rmsd = rmsd_m
                    a = a_m
                else:
                    rmsd = rmsd_p
                    a = a_p

        # Reduce the step size
        step /= 2.

    return a



def align_molecules(struct, ref, n_conf=0, conf_params=None, thresh=1e-4):
    """
    Align the heavy atoms of a molecule to those of a reference, changing the conformer angles to obtain the best alignment.

    Inputs:     - struct        Input structure
                - ref           Reference structure
                - n_conf        Number of conformer angles
                - conf_params   Parameters of the conformer angles
                - thresh        Threshold for angle convergence

    Outputs:    - tmp_struct    Aligned structure
                - rmsd          RMSD of heavy atom positions compared to the reference structure
    """

    # Get the elements in the molecule
    symbs = ref.get_chemical_symbols()
    # Get the indices of the heavy atoms in the molecule
    ha_inds = [i for i, s in enumerate(symbs) if s != "H"]

    tmp_struct = copy.deepcopy(struct)

    # Initialize array of optimal conformer angles
    conf_angles = np.zeros(n_conf)

    # If conformers are variable, get the indices of the conformers in order of increasing number of atoms involved in the mask
    if conf_params is not None:
        ls = []
        for m in conf_params["mask"]:
            ls.append(len(m))
        inds = np.argsort(ls)
    else:
        inds = []


    for i in inds:
        # Generate the mask for the selected conformer
        mask = np.array([1 if j in conf_params["mask"][i] else 0 for j in range(len(tmp_struct))])

        # Optimize the angle of conformer i
        a = opt_conf(tmp_struct, ref, conf_params["a1"][i], conf_params["a2"][i], conf_params["a3"][i], conf_params["a4"][i], conf_params["mask"][i], ha_inds, thresh)

        # Update the structure
        tmp_struct.set_dihedral(a1=conf_params["a1"][i], a2=conf_params["a2"][i], a3=conf_params["a3"][i], a4=conf_params["a4"][i], angle=a, mask=mask)

    # Get heavy atoms positional RMSD
    rmsd, _ = pos_rmsd(tmp_struct, ref, ha_inds, align=ha_inds)
    # Get aligned positions (including hydrogen atoms)
    _, al_pos = pos_rmsd(tmp_struct, ref, list(range(len(symbs))), align=ha_inds)

    # Set the aligned structure
    tmp_struct.set_positions(al_pos)

    return tmp_struct, rmsd




def retrieve_initial_structure(cryst, n_atoms, ref_struct, n_conf=0, conf_params=None, thresh=1e-4):
    """
    Obtain the initial conformer from a crystal structure

    Inputs: - cryst             Input crystal
            - n_atoms           Number of atoms in the molecule
            - ref_struct        Reference conformer
            - n_conf            Number of conformer angles
            - conf_params       Parameters of the conformer angles
            - thresh            Threshold for angle step size and heavy atom RMSD

    Output: - aligned_struct    Initial conformer of the crystal structure, aligned to best match the reference conformer
    """

    # Get atomic symbols and positions
    symbs = cryst.get_chemical_symbols()[:n_atoms]
    pos = cryst.get_positions()[:n_atoms]

    # Retrieve the gas-phase conformer corresponding to the crystal structure
    max_length = np.min(cryst.get_cell_lengths_and_angles()[:3])/2. - 0.5
    mol_pos = reconstruct_molecule_from_crystal(cryst, n_atoms, cutoff_factor=1.1)

    # Generate the gas-phase conformer
    new_struct = ase.Atoms(symbols=symbs, positions=mol_pos, pbc=False)

    # Align the structure to the reference, optimizing the conformer angles
    aligned_struct, rmsd = align_molecules(new_struct, ref_struct, n_conf=n_conf, conf_params=conf_params)

    # If the RMSD of atomic positions of heavy atoms is small enough, update the input conformer
    if rmsd < thresh:
        return aligned_struct
    # Otherwise, return the reference conformer
    else:
        return ref_struct
