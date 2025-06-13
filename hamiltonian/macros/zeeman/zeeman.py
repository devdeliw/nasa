import sympy as sp
import textwrap
import pickle 

g_e, mu_B, B0, g_n1, g_n2, mu_N = sp.symbols('g_e mu_B B0 g_n1 g_n2 mu_N')

# zeeman frequencies
Ωe   = g_e  * mu_B * B0
Ωn1  = g_n1 * mu_N * B0
Ωn2  = g_n2 * mu_N * B0

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

# build the diagonal entries
entries = []
for mI1, mI2 in nuclear_pairs:
    for s, m in electron_states:
        entries.append(
            m*Ωe     +      # electron zeeman
            mI1*Ωn1  +      # nucleus 1 zeeman
            mI2*Ωn2         # nucleus 2 zeeman
        )
H_Z = sp.diag(*entries) # Hamiltonian 

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

fname = "/Users/devaldeliwala/nasa/Hamiltonian/zeeman.pickle"
with open(fname, "wb") as f: 
    pickle.dump(H_Z, f)
    print(f"Zeeman Hamiltonian saved to {fname}.")


