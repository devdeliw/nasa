# This code was written by Deval Deliwala 
# NASA Glenn Research Center 

import sympy as sp
import textwrap
import pickle
import numpy as np

from pathlib import Path

HAMILTONIAN_DIRECTORY = Path(__file__).parent.parent.parent / "pickle" 

D1, D2 = sp.symbols('D1 D2')
electron_states = [(1, 1), (1, 0), (0, 0), (1, -1)]

# 4x4 electronic zfs hamiltonian 
H_elec = sp.zeros(4)
for i, (s, m) in enumerate(electron_states):

    # diagonal term: D1 * m^2 - (D1/3) * s(s+1)
    H_elec[i, i] = D1*m**2 - (D1/3)*s*(s+1)

    # off-diagonal term coupling m -> m+/-2
    for dm, expr in [
        (2, D2/2*(s*(s+1) - m*(m+1))),
        (-2, D2/2*(s*(s+1) - m*(m-1)))
    ]:
        newm = m + dm
        if (s, newm) in electron_states:
            j = electron_states.index((s, newm))
            H_elec[i, j] = expr
            H_elec[j, i] = expr

# 4x4 nuclear identity 
I_nuc = sp.eye(4)

# Final 16x16 ZFS Hamiltonian 
H_ZFS = sp.kronecker_product(H_elec, I_nuc)

def verify_hermicity(zfs_hamiltonian: sp.Matrix) -> bool: 
    H_np = np.array( 
        zfs_hamiltonian.subs(
            {
                D1: 1.65e-7, 
                D2: 2.07e-10
            }
        ).evalf().tolist(), dtype=complex
    )
    return np.allclose(H_np, H_np.conj().T, atol=1e-12)
assert verify_hermicity(H_ZFS), "ZFS Hamiltonian not Hermitian!"

def write_matrix_tex(matrix, filename, matrix_name="H_{ZFS}"):
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
    return f"./{filename}.tex"

fname = HAMILTONIAN_DIRECTORY / "zfs.pickle"
with open(fname, "wb") as f: 
    pickle.dump(H_ZFS, f) 
    print(f"ZFS Hamiltonian saved to {fname}.")
 
