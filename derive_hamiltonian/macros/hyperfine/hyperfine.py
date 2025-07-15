import sympy as sp
import textwrap

class HyperfineHamiltonian:
    """
    Builds a 2 electron x 2 s=1/2 nuclei Hyperfine Hamiltonian in the Zeeman 
    and coupled triplet-singlet basis via a Clebsch-Gordan transformation.

    """

    def __init__(self):
        # Define symbols for isotropic hyperfine constants
        # Eigenvalues of spin-1/2 operators S_z, I_z are ±1/2, hbar=1 natural units.
        self.A = {}  # Will contain A tensor labels
        for e in ('a', 'b'):
            for n in ('1', '2'):
                for comp in ('x', 'y', 'z'):
                    key = f'A{e}{n}{comp}'
                    self.A[key.upper()] = sp.symbols(key)

        # Zeeman Basis
        self.basis_ze = self._generate_zeeman_basis()               
        self.index_ze = {ket: i for i, ket in enumerate(self.basis_ze)}
        self.H_ze = self._build_zeeman_matrix()                     # 16x16

        # Clebsch-Gordan Unitary Transformation
        self.W = self._build_cg_unitary()                           # 16x16

        # Coupled Basis Hyperfine Hamiltonian
        self.H_coup = sp.simplify(self.W * self.H_ze * self.W.H)    # type: ignore # 16x16

    def _generate_zeeman_basis(self):
        # Builds the Zeeman |m_a, m_b, m_I1, m_I2> basis with m = ±1/2.

        half = sp.Rational(1, 2)
        return [
            (m_a, m_b, m_I1, m_I2)
            for m_a in (half, -half)
            for m_b in (half, -half)
            for m_I1 in (half, -half)
            for m_I2 in (half, -half)
        ]

    def _delta_parallel(self, m_e, m_I):
        return m_e == m_I

    def _delta_antiparallel(self, m_e, m_I):
        return m_e == -m_I

    def _action_on_ket(self, ket):
        # Calculates H_HF |ket> in Zeeman basis, returns dict {new_ket: coeff}.
        
        m_a, m_b, m_I1, m_I2 = ket
        psi = {}

        # diagonal S_z I_z term: A * m_e * m_I
        diag = (
            self.A['AA1Z'] * m_a * m_I1 +
            self.A['AB1Z'] * m_b * m_I1 +
            self.A['AA2Z'] * m_a * m_I2 +
            self.A['AB2Z'] * m_b * m_I2
        )
        psi[ket] = diag

        # off-diagonal flip-flop S_x I_x + S_y I_y -> (A_x - A_y)/4 etc.
        def add_flip(m_e, m_I, x_key, y_key, new_ket):
            if self._delta_parallel(m_e, m_I):
                coeff = (self.A[x_key] - self.A[y_key]) / 4
                psi[new_ket] = coeff
            elif self._delta_antiparallel(m_e, m_I):
                coeff = (self.A[x_key] + self.A[y_key]) / 4
                psi[new_ket] = coeff

        add_flip(m_a, m_I1, 'AA1X', 'AA1Y', (-m_a,  m_b, -m_I1,  m_I2))
        add_flip(m_a, m_I2, 'AA2X', 'AA2Y', (-m_a,  m_b,  m_I1, -m_I2))
        add_flip(m_b, m_I1, 'AB1X', 'AB1Y', ( m_a, -m_b, -m_I1,  m_I2))
        add_flip(m_b, m_I2, 'AB2X', 'AB2Y', ( m_a, -m_b,  m_I1, -m_I2))

        return psi

    def _build_zeeman_matrix(self):
        # Return the full 16x16 hyperfine template in the Zeeman basis.
        
        size = len(self.basis_ze)
        H = sp.MutableDenseMatrix(size, size, lambda *_: 0)
        for j, ket in enumerate(self.basis_ze):
            action = self._action_on_ket(ket)
            for new_ket, coeff in action.items():
                i = self.index_ze[new_ket]
                H[i, j] = coeff
        return H.as_immutable()

    def _build_cg_unitary(self):
        # Builds the unitary for two-electron Clebsch-Gordan transform otimes identity on nuclei.

        half = sp.sqrt(sp.Rational(1, 2))
        U = sp.Matrix([
            [1,    0,     0,    0],
            [0,  half,  half,   0],
            [0,  half, -half,   0],
            [0,    0,     0,    1]
        ])
        I4 = sp.eye(4)
        return sp.kronecker_product(U, I4)

    def coupled_matrix(self):
        # Return the 16x16 hyperfine matrix in coupled basis.

        return self.H_coup

# Helper to convert full anisotropic to isotropic form
A_a1x, A_a1y, A_a1z, A_a2x, A_a2y, A_a2z, \
A_b1x, A_b1y, A_b1z, A_b2x, A_b2y, A_b2z = sp.symbols(
    'Aa1x Aa1y Aa1z Aa2x Aa2y Aa2z Ab1x Ab1y Ab1z Ab2x Ab2y Ab2z'
)
Aa1, Aa2, Ab1, Ab2 = sp.symbols('Aa1 Aa2 Ab1 Ab2')

def convert_to_isotropic(H_sym: sp.Matrix) -> sp.Matrix:
    import numpy as np 
    # Substitute anisotropic A_xyz -> isotropic Aa, Ab values.

    subs_map = {
        A_a1x: Aa1, A_a1y: Aa1, A_a1z: Aa1,
        A_a2x: Aa2, A_a2y: Aa2, A_a2z: Aa2,
        A_b1x: Ab1, A_b1y: Ab1, A_b1z: Ab1,
        A_b2x: Ab2, A_b2y: Ab2, A_b2z: Ab2,
    }
    H_iso = sp.simplify(H_sym.subs(subs_map))

    # Verify Numerical Hermicity 
    H_np = np.array(
        H_iso.subs({
            Aa1: 1.46e-7, Ab1: 1.46e-7,
            Aa2: 2.81e-7, Ab2: 2.81e-7
        }).evalf().tolist(),
        dtype=complex
    )
    assert np.allclose(H_np, H_np.conj().T, atol=1e-12), "Hyperfine Hamiltonian is not Hermitian!"
    return H_iso

def convert_to_secular(H_sym: sp.Matrix) -> sp.Matrix:
    zero_transverse = {
        A_a1x: 0, A_a1y: 0,
        A_a2x: 0, A_a2y: 0,
        A_b1x: 0, A_b1y: 0,
        A_b2x: 0, A_b2y: 0,
    }
    H_sec_aniso = H_sym.subs(zero_transverse)
    return convert_to_isotropic(H_sec_aniso)

# Export to LaTeX
def write_matrix_tex(matrix, filename, matrix_name="H"):
    latex_matrix = sp.latex(matrix)
    doc = textwrap.dedent(rf"""
    \documentclass[preview]{{standalone}}
    \usepackage{{amsmath}}
    \begin{{document}}
    \[
    {matrix_name} = {latex_matrix}
    \]
    \end{{document}}
    """).strip()
    with open(filename, 'w') as f:
        f.write(doc)
    return filename

if __name__ == '__main__':
    import pickle
    from pathlib import Path

    hf = HyperfineHamiltonian()
    H_iso = convert_to_isotropic(hf.H_coup)
    H_secular   = convert_to_secular(hf.H_coup)

    folder = Path.home() / "nasa" / "hamiltonian" / "pickle"
    folder.mkdir(parents=True, exist_ok=True)
    fname = folder / "hyperfine.pickle"
    
    with open(fname, "wb") as f:
        pickle.dump(H_iso, f)
        print(f"Hyperfine Hamiltonian saved to {fname}.")

    fname = folder / "hyperfine_sec.pickle" 
    with open(fname, "wb") as f: 
        pickle.dump(H_secular, f) 
        print(f"Secular Hyperfine Hamiltonian saved to {fname}")
