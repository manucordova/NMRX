#####
# Last modified 29.06.2020
# Re-writing the script so that it is more general
# Authors: Manuel Cordova, Martins Balodis
#####



### Import libraries
import numpy as np
import sys
import copy
import scipy.optimize as op
import ase
#import ase.io
#import ase.spacegroup
#import fnmatch

### Import custom libraries
sys.path.insert(0, "../src")
# Import source files
import new_ml_functions as ml
import new_energy_functions as en
import new_generate_crystals as cr

# Available nuclei for ShiftML
symb2z = {"H":1, "C":6, "N":7, "O":8}



def compute_T(vs, ms):
    """
    """
    # Boltzmann constant in Ha/K
    k = 3.16681534524639e-6
    
    # Return the temperature
    return compute_KE(vs, ms)*2/(3*len(ms)*k)



def compute_KE(vs, ms):
    """
    """
    # Boltzmann constant in Ha/K
    k = 3.16681534524639e-6

    # Get the kinetic energy of the system
    K = 0.
    for i in range(len(ms)):
        K += ms[i]*vs[i].dot(vs[i])

    # Return the kinetic energy
    return K/2

    
    
def compute_T_SI(vs, ms):
    """
    """
    k = 1.38064852e-23
    
    v_conv = 2187691.26379
    m_conv = 9.10938356e-31

    # Get the kinetic energy of the system
    K = 0
    for i in range(len(ms)):
        K += ms[i]*vs[i].dot(vs[i])/(2)*m_conv*v_conv*v_conv

    # Return the temperature
    return K*2/(3*len(ms)*k)



def initialize_velocities(struct, T_ext, verbose=True):
    """
    """
    # Boltzmann constant in Ha/K
    k = 3.16681534524639e-6
    # Mass conversion from u to kg
    m_conv = 1822.8884849294
    
    # Get number of atoms
    N = len(struct)
    # Get array of masses in kg
    ms = struct.get_masses()*m_conv
    M = np.sum(ms)
    
    # Initialize array of momenta
    vs = np.zeros((len(struct), 3))
    # Initialize COM velocity
    v_COM = np.zeros(3)
    
    # Sample the momenta according to the Maxwell-Boltzmann distribution of velocities
    for i in range(N):
        # Variance of Gaussian distribution
        sig = np.sqrt(k*T_ext/ms[i])
        for j in range(3):
            vs[i,j] = sig*np.random.normal(0,1)
        v_COM += ms[i]*vs[i]/M
    
    # Remove the COM motion
    for v in vs:
        v -= v_COM
    
    # Compute the actual temperature corresponding to the velocity distribution
    Temp = compute_T(vs, ms)
    
    # Scale the velocities such that the actual temperature corresponds to the target
    vs /= np.sqrt(Temp/T_ext)
    
    # Recompute the actual temperature (sanity check)
    Temp = compute_T(vs, ms)
    
    if verbose:
        print("Initial momenta generated. Initial temperature: {:.2f} K (target: {:.2f} K)".format(Temp, T_ext))
    
    return vs, ms
    

    
def compute_forces_part(struct, cost_type, cost_options):
    """
    """
    
    if cost_type == "E":
        E, forces, stress = en.dftbplus_forces(cost_options["directory"], struct, cost_options["dftb_path"])
    
    elif cost_type in symb2z.keys():
        raise ValueError("Not implemented yet.")
    
    elif cost_type == "D":
        raise ValueError("Not implemented yet.")
    
    else:
        raise ValueError("Unknown cost function type: {}".format(cost_type))

    return E, forces, stress



def compute_forces(struct, cost_function, cost_factors, cost_options):
    """
    """
    
    forces = {}
    E = {}
    stress = {}
    
    for c in cost_function:
        E[c], forces[c], stress[c] = compute_forces_part(struct, c, cost_options[c])
    
    E["Tot"] = 0
    forces["Tot"] = np.zeros((len(struct), 3))
    stress["Tot"] = np.zeros((3,3))
    
    for c in cost_function:
        E["Tot"] += E[c]
        forces["Tot"] += forces[c]
        stress["Tot"] += stress[c]
        
    return E, forces, stress



def npt_evolve(struct, init_dt, nu_T, nu_P, T_ext, P_ext, ms, U, fs, sig, zeta_half_prev, eta_half_prev, rs_half_prev, vs_prev, vs_half_prev, fs_half_prev, all_zetas, n_mols=0, variable_shape=True):

    # Boltzmann constant in Ha/K
    k = 3.16681534524639e-6
    # Time conversion from second to atomic units
    t_conv = 1/(2.4188843265857e-17)
    # Length conversion from angstrom to atomic units
    l_conv = 1.88973
    
    # Extract atomic positions and cell from the crystal
    rs = struct.get_positions() * l_conv
    abc = struct.get_cell() * l_conv
    V = struct.get_volume() * (l_conv**3)
    
    # Number of atoms in the system
    N = len(ms)
    
    # Get time constants for thermostat and barostat
    tau_T = 1e-12/nu_T * t_conv
    tau_P = 1e-12/nu_P * t_conv
    
    # Convert time step from fs to atomic units
    dt = init_dt * 1e-15 * t_conv
    
    # Predict current velocities
    vs = dt * np.divide(fs_half_prev, ms[:, None])
    vs += vs_prev
    
    # Predict positions after half step
    rs_half_next = rs_half_prev + (dt * vs)
    
    # Get current temperature
    Temp = compute_T(vs, ms)
    
    # Compute zeta after half step
    zeta_half_next = (Temp/T_ext) - 1
    zeta_half_next *= dt/(tau_T**2)
    zeta_half_next += zeta_half_prev
    
    # Compute current zeta
    zeta = (zeta_half_prev + zeta_half_next)/2
    
    # Compute eta after half step
    if variable_shape:
        eta_half_next = np.triu(sig) - (P_ext * np.eye(3))
    else:
        eta_half_next = sig - P_ext
    
    eta_half_next *= dt * V / (N * k * T_ext * (tau_P**2))
    eta_half_next += eta_half_prev
    
    # Compute current eta
    eta = (eta_half_prev + eta_half_next)/2
    
    # Compute velocities after half step
    vs_half_next = np.divide(fs, ms[:, None])
    
    # Compute molecule velocities
    if n_mols > 0:
        n_at = int(len(struct)/n_mols)
        Vs = np.zeros((n_mols, 3))
        Ms = np.zeros(n_mols)
        for i in range(n_mols):
            Vs[i] = np.sum(np.multiply(vs[i*n_at:(i+1)*n_at], ms[i*n_at:(i+1)*n_at, None]))/np.sum(ms[i*n_at:(i+1)*n_at])
            Ms[i] = np.sum(ms[i*n_at:(i+1)*n_at])
    
    if variable_shape:
        if n_mols > 0:
            vs_half_next -= zeta * vs
            for i, v in enumerate(vs):
                I = int(i/n_at)
                vs_half_next[i] -= eta.dot(Vs[I])
                
        else:
            M = (zeta * np.eye(3)) + eta
            for i, v in enumerate(vs):
                vs_half_next[i] -= M.dot(vs[i])
                
    else:
        if n_mols > 0:
            vs_half_next -= zeta * vs
            for i in range(n_mols):
                vs_half_next[i*n_at:(i+1)*n_at] -= eta * Vs[i]
            
        else:
            vs_half_next -= (zeta + eta) * vs
    
    vs_half_next *= dt
    vs_half_next += vs_half_prev
    
    # Compute current velocities
    vs = (vs_half_prev + vs_half_next)/2
    
    if n_mols > 0:
        # Get COM position after half step
        Rs_half_next = np.zeros((n_mols, 3))
        for i in range(n_mols):
            Rs_half_next[i] = np.sum(np.multiply(rs[i*n_at:(i+1)*n_at], ms[i*n_at:(i+1)*n_at, None]))
            Rs_half_next[i] /= np.sum(ms[i*n_at:(i+1)*n_at])
    
    else:
        r_COM_half_next = np.sum(np.multiply(rs, ms[:, None])) / np.sum(ms)
    
    # Compute positions after full step
    if n_mols > 0:
        if variable_shape:
            rs_next = np.zeros_like(rs)
            for i, r in enumerate(rs):
                I = int(i/n_at)
                rs_next[i] = eta_half_next.dot(r - Rs_half_next[I])
                
        else:
            for i, r in enumerate(rs):
                I = int(i/n_at)
                rs_next[i] = eta_half_next * (r - Rs_half_next[I])
                
    else:
        if variable_shape:
            rs_next = np.zeros_like(rs)
            for i, r in enumerate(rs):
                rs_next[i] = eta_half_next.dot(r - r_COM_half_next)
                
        else:
            rs_next = eta_half_next * (rs - r_COM_half_next)
    
    rs_next += vs_half_next
    rs_next *= dt
    rs_next += rs
    
    # Compute positions after half step
    rs_half_next = (rs + rs_next)/2
    
    # Compute cell parameters after full step
    if variable_shape:
        abc_next = eta_half_next.dot(abc)
    else:
        abc_next = eta_half_next * abc
    abc_next *= dt
    abc_next += abc
    abc_next = np.tril(abc_next)
    
    # Update the crystal object
    struct.set_cell(abc_next / l_conv)
    struct.set_positions(rs_next / l_conv)
    struct.wrap()
    
    # Get conserved quantity
    all_zetas.append(zeta)
    
    conserved = U + compute_KE(vs, ms)
    conserved += 3/2 * N * k * T_ext * (zeta**2) * (tau_T**2)
    conserved += 3 * N * k * T_ext * (np.sum(all_zetas)*dt)
    conserved += P_ext * V
    
    return struct, rs_half_next, vs, vs_half_next, zeta_half_next, eta_half_next, conserved, all_zetas
    
    
    
def compute_constraints(struct, constr, exponent):
    """
    """
    # Length conversion from angstrom to atomic units
    l_conv = 1.88973
    
    cs = []
    for c in constr.keys():
        cs.append(((struct.get_distance(c[0], c[1], mic=True)-constr[c]) * l_conv)**exponent)
    
    return np.array(cs)



def compute_constraint_gradient(struct, i, j, d, exponent, step=1e-3):
    """
    """
    # Length conversion from angstrom to atomic units
    l_conv = 1.88973
    
    g = np.zeros(3)
    
    v = struct.get_distance(i, j, mic=True, vector=True)
    
    for k in range(3):
    
        v_p = np.copy(v)
        v_m = np.copy(v)
        
        v_p[k] -= step
        v_m[k] += step
        
        f_p = ((np.linalg.norm(v_p)**exponent - d**exponent) * l_conv**exponent)
        f_m = ((np.linalg.norm(v_m)**exponent - d**exponent) * l_conv**exponent)
        
        g[k] = (f_p - f_m)/(2 * step * l_conv)
    
    
    return g



def SHAKE(init_dt, cryst, cryst_next, vs_half_prev, vs_half_next, fs, ms, constr, thresh=0.1, exponent=2.):
    """
    """
    # Time conversion from second to atomic units
    t_conv = 1/(2.4188843265857e-17)
    # Length conversion from angstrom to atomic units
    l_conv = 1.88973
    
    # Convert time step to atomic units
    dt = init_dt * 1e-15 * t_conv
    
    N = len(cryst)
    
    while not np.all(compute_constraints(cryst_next, constr, exponent) < thresh):

        print(np.max(compute_constraints(cryst_next, constr, exponent)), thresh)

        # Compute the constraints after the time step
        cs_next = compute_constraints(cryst_next, constr, exponent)
        
        # Compute Lagrange multipliers
        ls = []
        for k, c in enumerate(constr.keys()):
            l = 1/(dt**2)
            l *= cs_next[k]
            
            tmp = 0.
            for i in range(N):
                if c[0] == i:
                    g = compute_constraint_gradient(cryst, c[0], c[1], constr[c], exponent)
                    g_next = compute_constraint_gradient(cryst_next, c[0], c[1], constr[c], exponent)
                    
                    tmp += g.T.dot(g_next)/ms[i]
                    
                elif c[1] == i:
                    g = compute_constraint_gradient(cryst, c[1], c[0], constr[c], exponent)
                    g_next = compute_constraint_gradient(cryst_next, c[1], c[0], constr[c], exponent)
                    
                    tmp += g.T.dot(g_next)/ms[i]
            
            l /= tmp
            ls.append(l)
        
        # Update velocities
        for i in range(N):
            tmp = np.copy(vs_half_prev[i])
            
            tmp += fs[i]/ms[i] * dt
            
            for k, (l, c) in enumerate(zip(ls, constr.keys())):
                if c[0] == i:
                    g = compute_constraint_gradient(cryst, c[0], c[1], constr[c], exponent)
                    
                    tmp -= l/ms[i] * g * dt
                elif c[1] == i:
                    g = compute_constraint_gradient(cryst, c[1], c[0], constr[c], exponent)
                    
                    tmp -= l/ms[i] * g * dt
            
            vs_half_next[i] = np.copy(tmp)
        
        rs_next = cryst.get_positions() * l_conv
        rs_next += vs_half_next * dt
        rs_next /= l_conv
        
        cryst_next.set_positions(rs_next)
        cryst_next.wrap()
    
    return cryst_next, vs_half_next



def generate_bond_constraints(struct, n_mol, thresh=1.1):
    """
    """
    
    pos = struct.get_positions()
    symbs = struct.get_chemical_symbols()
    
    constr = {}
    for i, (p1, s1) in enumerate(zip(pos, symbs)):
        for j, (p2, s2) in enumerate(zip(pos, symbs)):
            d = struct.get_distance(i, j)
            if i != j and d < thresh * (ase.data.covalent_radii[ase.data.atomic_numbers[s1]] + ase.data.covalent_radii[ase.data.atomic_numbers[s2]]):
                for k in range(n_mol):
                    constr[(i + (k * len(struct)), j + (k * len(struct)))] = d
    
    
    return constr
