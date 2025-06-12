import sympy as sp
import textwrap
import pickle

# ZFS parameters
D1, D2 = sp.symbols('D1 D2')

electron_states = [(1, 1), (1, 0), (0, 0), (1, -1)]

# build 4x4 electron ZFS Hamiltonian 
H_elec = sp.zeros(4)
for i, (s, m) in enumerate(electron_states):
    # Diagonal term: D1 * m^2 - (D1/3) * s(s+1)
    H_elec[i, i] = D1*m**2 - (D1/3)*s*(s+1)
    # Off-diagonal terms coupling m → m±2
    for dm, expr in [(2, D2/2*(s*(s+1) - m*(m+1))),
                     (-2, D2/2*(s*(s+1) - m*(m-1)))]:
        newm = m + dm
        if (s, newm) in electron_states:
            j = electron_states.index((s, newm))
            H_elec[i, j] = expr

# 4x4 identity nuclear ZFS Hamiltonian 
I_nuc = sp.eye(4)

# Final ZFS Hamiltonian 
H_ZFS = sp.kronecker_product(I_nuc, H_elec)

def write_matrix_tex(matrix, filename, matrix_name="H_{DD,ee}"):
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


fname = "/Users/devaldeliwala/nasa/Hamiltonian/zfs.pickle" 
with open(fname, "wb") as f: 
    pickle.dump(H_ZFS, f) 
    print(f"ZFS Hamiltonian saved to {fname}.")
