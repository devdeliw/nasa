import numpy as np
from scipy.linalg import solve_sylvester

# taken from matlab portion at end of Colins 

def dirac_delta(m, n):
    return 1 if m == n else 0

def pauli(spin, xyz):
    """

    Create a Pauli spin matrix for a given spin along axis:
      xyz == 1: σx, 
      xyz == 2: σy, 
      xyz == 3: σz.

    """
    dim = int(2 * spin + 1)
    mvals = np.linspace(-spin, spin, dim)
    Out = np.zeros((dim, dim), dtype=complex)
    
    if xyz == 1:
        # x-component
        for j in range(dim):
            for k in range(dim):
                val = 0.5 * np.sqrt(spin*(spin+1) - mvals[j]*mvals[k]) * \
                      (dirac_delta(j, k+1) + dirac_delta(j+1, k))
                Out[j, k] = val
    elif xyz == 2:
        # y-component
        for j in range(dim):
            for k in range(dim):
                val = 0.5j * np.sqrt(spin*(spin+1) - mvals[j]*mvals[k]) * \
                      (dirac_delta(j, k+1) - dirac_delta(j+1, k))
                Out[j, k] = val
    elif xyz == 3:
        # z-component
        for j in range(dim):
            for k in range(dim):
                val = -dirac_delta(j, k) * mvals[j]
                Out[j, k] = val
    return Out

def spin_system(spins):
    """

    Create full spin operators (sx, sy, sz) in the tensor-product basis 
    for a list of spins.
    
    Params : spins: list of spin values (e.g. [0.5, 0.5, 0.5, 0.5]).
    Returns: three lists [sx_list, sy_list, sz_list] corresponding to the 
             full operator for each spin.

    """
    n_spins = len(spins)
    sx_list = [None] * n_spins
    sy_list = [None] * n_spins
    sz_list = [None] * n_spins

    for i in range(n_spins):
        Px = pauli(spins[i], 1)
        Py = pauli(spins[i], 2)
        Pz = pauli(spins[i], 3)

        # full operator by taking the Kronecker (Tensor) product with 
        # identities for other spins
        for j in range(n_spins):
            d_j = int(2 * spins[j] + 1)
            if j == 0:
                if j == i:
                    op_x = Px
                    op_y = Py
                    op_z = Pz
                else:
                    op_x = np.eye(d_j, dtype=complex)
                    op_y = np.eye(d_j, dtype=complex)
                    op_z = np.eye(d_j, dtype=complex)
            else:
                if j == i:
                    op_x = np.kron(op_x, Px)
                    op_y = np.kron(op_y, Py)
                    op_z = np.kron(op_z, Pz)
                else:
                    op_x = np.kron(op_x, np.eye(d_j, dtype=complex))
                    op_y = np.kron(op_y, np.eye(d_j, dtype=complex))
                    op_z = np.kron(op_z, np.eye(d_j, dtype=complex))
        sx_list[i] = op_x
        sy_list[i] = op_y
        sz_list[i] = op_z
    return sx_list, sy_list, sz_list

def spectra_producer(Field, g, HF1, HF2, r_ab, r_ae, r_ea):
    """ 

    Parameters:
      Field : 1D numpy array of magnetic field values (in mT)
      g     : g-factor
      HF1   : Hyperfine interaction strength at Site 1 (mT)
      HF2   : Hyperfine interaction strength at Site 2 (mT)
      r_ab  : Parameter r_ab (GHz)
      r_ae  : Parameter r_ae (GHz)
      r_ea  : Parameter r_ea (GHz)
    
    Returns:
      dI    : The derivative of I1 with respect to Field (the simulated NZFMR spectrum)

    """
    HF1  = abs(HF1)
    HF2  = abs(HF2)
    ks1  = abs(r_ab)
    kd1  = abs(r_ae)
    Gen1 = abs(r_ea)

    # two electrons and two nuclei
    spins = [0.5, 0.5, 0.5, 0.5]
    sx, sy, sz = spin_system(spins)

    # Spin operators
    S_tot = sz[0] + sz[1]   # Total electron spin (z component)
    S1S2  = sx[0] @ sx[1] + sy[0] @ sy[1] + sz[0] @ sz[1]    # Electron-electron product
    IS1   = sx[2] @ sx[0] + sy[2] @ sy[0] + sz[2] @ sz[0]     # Electron-nuclear product (site 1)
    IS2   = sx[3] @ sx[1] + sy[3] @ sy[1] + sz[3] @ sz[1]     # Electron-nuclear product (site 2)

    # Projection operators (singlet and triplet)
    dim_full = S1S2.shape[0]
    Ident = np.eye(dim_full, dtype=complex)
    Ps = 0.25 * (Ident - 4.0 * S1S2)
    Pt = 0.25 * (3.0 * Ident + 4.0 * S1S2)

    # Constants (units: eV/mT and eV*ns)
    uB = 5.788e-8
    h_bar = 6.582e-7

    # Hyperfine Hamiltonian
    Hhf1 = g * uB * HF1 * IS1 + g * uB * HF2 * IS2
    I1 = np.zeros(len(Field), dtype=float)
    G1 = (1.0/16.0) * Gen1 * Ident

    # Loop over each field value
    for i, B0 in enumerate(Field):
        Bshift = B0 - 0.037851
        H01 = g * uB * Bshift * S_tot
        H_NZFMR1 = H01 + Hhf1

        # Define operators A and B 
        A = -((1j/h_bar) * H_NZFMR1 + (ks1/2) * Ps + (kd1/2) * Pt)
        B =  ((1j/h_bar) * H_NZFMR1 - (ks1/2) * Ps - (kd1/2) * Pt)
        
        # sylvester equation: A*P + P*B = -G1
        P1 = solve_sylvester(A, B, -G1)
        I1[i] = (kd1/ks1) * np.real(np.trace(Ps @ P1))
        
    I1 = I1 - I1[0]
    dI = np.gradient(I1, Field)
    return dI

