# Run from ~/nasa/hamiltonian/
import pickle
import sympy as sp 
import textwrap 

def load_matrix(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_matrix(matrix, filename):
    with open(filename, 'wb') as f:
        pickle.dump(matrix, f)

def combine_spin_hamiltonians(
        zeeman_file, 
        hyperfine_file, 
        zfs_file, 
        exchange_file, 
        output_file
):
    H_zeeman    = load_matrix(zeeman_file)
    H_hyperfine = load_matrix(hyperfine_file)
    H_zfs       = load_matrix(zfs_file)
    H_exchange  = load_matrix(exchange_file)

    # shape mismatch
    shapes = {mat.shape for mat in (H_zeeman, H_hyperfine, H_zfs, H_exchange)}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent shapes: {shapes}")

    # Spin Hamiltonian
    H_SPIN = H_zeeman + H_hyperfine + H_zfs + H_exchange

    save_matrix(H_SPIN, output_file)
    print(f"Spin Hamiltonian saved to {output_file}.")
    return H_SPIN 

def write_latex(matrix, filename, matrix_name="H_total"):
    """Write a LaTeX document containing the matrix."""
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
    with open(filename, 'w') as f:
        f.write(doc)


if __name__ == "__main__":
    H_SPIN = combine_spin_hamiltonians(
        zeeman_file="zeeman.pickle",
        hyperfine_file="hyperfine.pickle",
        zfs_file="zfs.pickle",
        exchange_file="exchange.pickle",
        output_file="spin_hamiltonian.pickle"
    )

    write_latex(H_SPIN, "spin_hamiltonian.tex", matrix_name="H_0")

