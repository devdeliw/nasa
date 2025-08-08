# This code was written by Deval Deliwala 
# NASA Glenn Research Center 

import textwrap
import pickle 
import numpy as np 
import sympy as sp

from pathlib import Path

HAMILTONIAN_DIRECTORY = Path(__file__).parent.parent.parent / "pickle"
g_e, mu_B, B0, g_n1, g_n2, mu_N = sp.symbols('g_e mu_B B0 g_n1 g_n2 mu_N')

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

# diagonal entries 
entries = []
for s, m in electron_states: 
    for mI1, mI2 in nuclear_pairs: 
        entries.append(
            m*omega_e    +  # electron zeeman
            mI1*omega_n1 +  # nucleus 1 zeeman
            mI2*omega_n2    # nucleus 2 zeeman
        )

# Zeeman Hamiltonian 
H_Z = sp.diag(*entries) 

def verify_hermicity(zeeman_hamiltonian: sp.Matrix) -> bool: 
    H_np = np.array(
        zeeman_hamiltonian.subs(
            { 
                B0: 0.01, 
                g_e: 2.0023, 
                g_n1: -1.11, 
                g_n2: 1.404, 
                mu_B: 5.788e-9,
                mu_N: 3.152e-12, 
            }
        ).evalf().tolist(), dtype=complex
    )
    return np.allclose(H_np, H_np.conj().T, atol=1e-12)
assert verify_hermicity(H_Z), "Zeeman Hamiltonian not Hermitian"

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

fname = HAMILTONIAN_DIRECTORY / "zeeman.pickle"
with open(fname, "wb") as f: 
    pickle.dump(H_Z, f)
    print(f"Zeeman Hamiltonian saved to {fname}.")
