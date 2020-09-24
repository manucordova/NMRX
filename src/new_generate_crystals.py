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
    
    delta = np.abs(x0-x1)
    if pbc:
        for a in abc:
            delta = np.where(delta > 0.5*a, delta-a, delta)
    
    return np.sqrt((delta**2).sum(axis=-1))



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
    
    try:
        # Check clashes
        if clash_type == "intra":
            for i in range(n_atoms-1):
                ds = distance(pos[i], pos[i+1:n_atoms], abc, pbc=pbc)
                if np.min(ds) < max_contact:
                    inds = np.where(ds < max_contact)[0]
                    for j in inds:
                        if ds[j] < contacts["{}-{}".format(symbs[i], symbs[i+1+j])]:
                            return True
        if clash_type == "inter":
            for i in range(n_atoms):
                ds = distance(pos[i], pos[n_atoms:], abc, pbc=pbc)
                if np.min(ds) < max_contact:
                    inds = np.where(ds < max_contact)[0]
                    for j in inds:
                        if ds[j] < contacts["{}-{}".format(symbs[i], symbs[n_atoms+j])]:
                            return True
    except:
        # If there is an issue in the clash detection, assume that there is a clash
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
    
    # Set conformation
    for i, a in enumerate(conf_angles):
        # Set mask for the conformational change
        mask = np.array([1 if j in conf_params["mask"][i] else 0 for j in range(len(trial_structure))])
        # Change conformation according to the generated angle
        trial_structure.set_dihedral(a1=conf_params["a1"][i], a2=conf_params["a2"][i], a3=conf_params["a3"][i], a4=conf_params["a4"][i], angle=a, mask=mask)
    
    # Rotate molecule and center it to the origin
    center = trial_structure[:n_atoms].get_center_of_mass()
    centred = True
    if np.linalg.norm(trans) == 0:
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
        # Generate new angles if they should be varied
        if "alpha" in parameter_set:
            lat[3] = cell_params_lims[2] + (cell_params_lims[3] - cell_params_lims[2]) * np.random.random()
        if "beta" in parameter_set:
            lat[4] = cell_params_lims[2] + (cell_params_lims[3] - cell_params_lims[2]) * np.random.random()
        if "gamma" in parameter_set:
            lat[5] = cell_params_lims[2] + (cell_params_lims[3] - cell_params_lims[2]) * np.random.random()
        
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
