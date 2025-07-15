import sympy as sp
import textwrap
import pickle 
import numpy as np 

from pathlib import Path

g_e, mu_B, B0, g_n1, g_n2, mu_N = sp.symbols('g_e mu_B B0 g_n1 g_n2 mu_N')

# build basis in same ordering as hyperfine
electron_states = [
    (1, +1),
    (1,  0),
    (0,  0),
    (1, -1),
]
nuclear_pairs = [
    ( sp.Rational(+1,2), sp.Rational(+1,2) ),
    ( sp.Rational(+1,2), sp.Rational(-1,2) ),
    ( sp.Rational(-1,2), sp.Rational(+1,2) ),
    ( sp.Rational(-1,2), sp.Rational(-1,2) ),
]

# zeeman frequencies
omega_e   = g_e  * mu_B * -abs(B0)
omega_n1  = g_n1 * mu_N * -abs(B0)
omega_n2  = g_n2 * mu_N * -abs(B0)

# build the diagonal entries
entries = []
for s, m in electron_states: 
    for mI1, mI2 in nuclear_pairs: 
        entries.append(
            m*omega_e     +      # electron zeeman
            mI1*omega_n1  +      # nucleus 1 zeeman
            mI2*omega_n2         # nucleus 2 zeeman
        )
H_Z = sp.diag(*entries) # Hamiltonian 

# Verify Numerical Hermicity 
H_np = np.array(
    H_Z.subs({B0:0.01, g_e:2.002319, g_n1:-1.110104, g_n2:1.404738,
                 mu_B:5.7883818e-9, mu_N:3.1524513e-12})
    .evalf()
    .tolist(),
    dtype=complex
)
assert np.allclose(H_np, H_np.conj().T, atol=1e-12), "Zeeman Hamiltonian is not Hermitian!"


def write_matrix_tex(matrix, filename, matrix_name="H_Z"):
    latex_matrix = sp.latex(matrix)
    doc = textwrap.dedent(rf"""
    \documentclass[a4paper,landscape]{{article}}
    \usepackage{{graphicx}}
    % -- math
    \usepackage{{amssymb}}
    \usepackage{{amsmath}}
    \usepackage{{esint}}
    % -- noindent
    \setlength\parindent{{0pt}}
    \begin{{document}}
    \small
    \[\scalebox{{0.4}}{{%
        $
        {matrix_name} = {latex_matrix}
        $
    }}\]
    \end{{document}}
    """).strip()
    with open(f"./{filename}", "w") as f:
        f.write(doc)
    return f"./{filename}"

home = Path.home() 
target = home / "nasa" / "hamiltonian"
fname = target / "pickle"/ "zeeman.pickle"

with open(fname, "wb") as f: 
    pickle.dump(H_Z, f)
    print(f"Zeeman Hamiltonian saved to {fname}.")

# ordering: |1,+1>, |1,0>, |0,0>, |1,−1>
S_plus = sp.zeros(4)
idx = {(1,+1):0,(1,0):1,(0,0):2,(1,-1):3}
for (s,m), i in idx.items():
    if m < s:
        j = idx[(s,m+1)]
        S_plus[j,i] = sp.sqrt(s*(s+1) - m*(m+1))
S_minus = S_plus.T
Sx_e = (S_plus + S_minus)/2
Sz_e = sp.diag(+1, 0, 0, -1)           # eigen-m values

# lift to 16×16 (electron ⊗ nuclear identity)
I4 = sp.eye(4)
Sx_tot = sp.kronecker_product(Sx_e, I4)
Sz_tot = sp.kronecker_product(Sz_e, I4)

c_B = sp.simplify(sp.trace(H_Z*Sz_tot) / sp.trace(Sz_tot*Sz_tot))
print("Detuning term c(B0) =", c_B)   


 
