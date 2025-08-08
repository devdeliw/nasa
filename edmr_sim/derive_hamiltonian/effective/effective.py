# This code was written by Deval Deliwala 
# NASA Glenn Research Center

"""
This code adds the Rotating Wave Approximation (RWA) 
to the static Spin Hamiltonian. 

However, this is also done in the ~/nasa/solving_SLE code. 
Hence, the Hamiltonian built here is not used. 

"""

import sympy as sp 
import numpy as np
import pickle

from pathlib import Path

g_e, mu_B, B0       = sp.symbols("g_e mu_B B0")
g_n1, g_n2, mu_N    = sp.symbols("g_n1 g_n2 mu_N")
h, nu, omega1       = sp.symbols("h nu omega1")
Aa1, Aa2, Ab1, Ab2  = sp.symbols("Aa1 Aa2 Ab1 Ab2")
J, D1, D2           = sp.symbols("J D1 D2")

# coupled electron basis 
S_plus = sp.zeros(4)
idx = {(1,+1):0,(1,0):1,(0,0):2,(1,-1):3}
for (s,m), i in idx.items():
    if m < s:
        j = idx[(s,m+1)]
        S_plus[j,i] = sp.sqrt(s*(s+1) - m*(m+1))
S_minus = S_plus.T
Sx_e = (S_plus + S_minus)/2
Sz_e = sp.diag(+1, 0, 0, -1)           

# lift to electron-nuclear 16x16
I4 = sp.eye(4)  
Sx_tot = sp.kronecker_product(Sx_e, I4)
Sz_tot = sp.kronecker_product(Sz_e, I4)

# static spin hamiltonian
pkl_dir = Path(__file__).parent.parent / "pickle"
H_static = pickle.load((pkl_dir / "spin_hamiltonian.pickle").open("rb"))
if not (isinstance(H_static, sp.MatrixBase) and H_static.shape == (16,16)):
    raise ValueError("spin_hamiltonian.pickle must hold a 16x16 SymPy Matrix")

# RWA terms 
H_drive = 0.5 * h * omega1 * Sx_tot
H_shift = -h * nu    * Sz_tot

# full effective hamiltonian 
H_eff = sp.expand(H_static + H_drive + H_shift)

# numeric hermicity check 
subs = {
    B0:0.01, g_e:2.002319, mu_B:5.788e-9,
    g_n1:-1.110104, g_n2:1.404738, mu_N:3.152e-12,
    h:4.13566766e-15, nu:2.00098e8, omega1:2.0e6,
    Aa1:0, Aa2:0, Ab1:0, Ab2:0, J:0, D1:0, D2:0
}
H_num = np.array(H_eff.subs(subs).evalf().tolist(), dtype=complex)
assert np.allclose(H_num, H_num.conj().T, atol=1e-10)

out_path = pkl_dir / "effective.pickle"
pickle.dump(H_eff, out_path.open("wb"))
print("Effective Hamiltonian written to", out_path)
