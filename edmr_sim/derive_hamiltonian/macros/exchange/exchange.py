# This code was written by Deval Deliwala 
# NASA Glenn Research Center 

import sympy as sp
import textwrap
import pickle 
import numpy as np 

from pathlib import Path

HAMILTONIAN_DIRECTORY = Path(__file__).parent.parent.parent / "pickle"

J = sp.symbols('J')
electron_states = [(1, 1), (1, 0), (0, 0), (1, -1)]

# 4x4 electron Hamiltonian 
H_ex_elec = sp.zeros(4)
for i, (s, m) in enumerate(electron_states):
    H_ex_elec[i, i] = -J * (s * (s + 1) - 1.5) / 2

# 4x4 nuclear identity 
I_nuc = sp.eye(4)

# Full 16x16 exchange Hamiltonian 
H_EX = sp.kronecker_product(I_nuc, H_ex_elec)

def verify_hermicity(exchange_hamiltonian: sp.Matrix) -> bool: 
    H_np = np.array(
        exchange_hamiltonian.subs(
            {
                J: 1.0e-9
            }
        ).evalf().tolist(), dtype=complex
    )
    return np.allclose(H_np, H_np.conj().T, atol=1e-12)
assert verify_hermicity(H_EX), "Exchange Hamiltonian not Hermitian!"

def write_matrix_tex(matrix, filename, matrix_name="H_{ex}"):
    latex_mat = sp.latex(matrix)
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
        {matrix_name} = {latex_mat}
        $
    }}\]

    \end{{document}}
    """).strip()
    path = f"./{filename}.tex"
    with open(path, "w") as f:
        f.write(doc)
    return path

fname = HAMILTONIAN_DIRECTORY / "exchange.pickle"
with open(fname, "wb") as f: 
    pickle.dump(H_EX, f) 
    print(f"Exchange Hamiltonian written to {fname}.")
