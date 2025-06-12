import sympy as sp
import textwrap
import pickle 

J = sp.symbols('J')

# same electron basis ordering
electron_states = [(1, 1), (1, 0), (0, 0), (1, -1)]

# build 4x4 electron exchange Hamiltonian 
H_ex_elec = sp.zeros(4)
for i, (s, m) in enumerate(electron_states):
    H_ex_elec[i, i] = -J * (s * (s + 1) - sp.Rational(3, 2)) / 2

# 4x4 nuclear identity 
I_nuc = sp.eye(4)

# Exchange Hamiltonian 
H_EX = sp.kronecker_product(I_nuc, H_ex_elec)

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


fname = "/Users/devaldeliwala/nasa/Hamiltonian/exchange.pickle" 
with open(fname, "wb") as f: 
    pickle.dump(H_EX, f) 
    print(f"Exchange Hamiltonian written to {fname}.")
