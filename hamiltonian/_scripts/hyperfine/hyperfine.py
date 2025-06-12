import sympy as sp
import textwrap

class HyperfineHamiltonian:
    """
    Builds a 2 electron x 2 s=1/2 nuclei Hyperfine hamiltonian in the Zeeman 
    and coupled triplet-singlet basis via a Clebsch-Gordan transformation. 

    """

    def __init__(self):
        self.h2 = sp.symbols('hbar2')
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

        # Clebsch Gordan Unitary Transformation
        self.W = self._build_cg_unitary()                           # 16x16

        # Coupled Basis
        self.H_coup = sp.simplify(self.W * self.H_ze * self.W.T)    # 16x16

    def _generate_zeeman_basis(self):
        """
        Builds the zeeman |m_a, m_b, m_I1, m_I2> basis. 
        All spins are 1/2 so each basis state just has m=+/-1/2 labels. 

        """
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
        """
        Calculates H_HF (zeeman) |ket>. Stores resulting |kets> and 
        their  coefficients via `psi = {new_ket: coefficient}` 

        """
        m_a, m_b, m_I1, m_I2 = ket
        psi = {}

        # diagonal S_z I_z contribution
        diag = (
            self.A['AA1Z'] * m_a * m_I1 +
            self.A['AB1Z'] * m_b * m_I1 +
            self.A['AA2Z'] * m_a * m_I2 +
            self.A['AB2Z'] * m_b * m_I2
        )
        psi[ket] = self.h2 * diag

        # off-diagonal helper 
        def add_flip(m_e, m_I, x_key, y_key, new_ket):
            if self._delta_parallel(m_e, m_I):
                coeff = self.h2 / 4 * (self.A[x_key] - self.A[y_key])
                psi[new_ket] = coeff
            elif self._delta_antiparallel(m_e, m_I):
                coeff = self.h2 / 4 * (self.A[x_key] + self.A[y_key])
                psi[new_ket] = coeff

        add_flip(m_a, m_I1, 'AA1X', 'AA1Y', (-m_a,  m_b, -m_I1,  m_I2))
        add_flip(m_a, m_I2, 'AA2X', 'AA2Y', (-m_a,  m_b,  m_I1, -m_I2))
        add_flip(m_b, m_I1, 'AB1X', 'AB1Y', ( m_a, -m_b, -m_I1,  m_I2))
        add_flip(m_b, m_I2, 'AB2X', 'AB2Y', ( m_a, -m_b,  m_I1, -m_I2))

        return psi

    # building matrix
    def _build_zeeman_matrix(self):
        """
        Return the full 16x16 matrix in the Zeeman basis.

        """
        size = len(self.basis_ze) # 16 
        H = sp.MutableDenseMatrix(size, size, lambda *_: 0)
        for j, ket in enumerate(self.basis_ze):
            action = self._action_on_ket(ket)  # dict {new_ket: coeff}
            for new_ket, coeff in action.items():
                i = self.index_ze[new_ket]
                H[i, j] = coeff
        return H.as_immutable()

    def _build_cg_unitary(self):
        """
        Builds the Unitary Transformation to convert the Zeeman-Basis 
        Hamiltonian to the electron-coupled |s, m>|m_I1>|m_I2> Hamiltonian. 

        Kronecker product U otimes I4.
        (two-electron CG × two-nucleus identity).

        """
        half = sp.sqrt(sp.Rational(1, 2))
        U = sp.Matrix([
            [1,  0,    0,   0],
            [0,  half, half, 0],
            [0,  half, -half,0],
            [0,  0,    0,   1]
        ])
        I4 = sp.eye(4)
        return sp.kronecker_product(U, I4)

    def zeeman_matrix(self):
        return self.H_ze

    def coupled_matrix(self):
        return self.H_coup

    def pretty_print(self, coupled=False):
        M = self.H_coup if coupled else self.H_ze
        sp.pretty_print(M)

A_a1x, A_a1y, A_a1z, A_a2x, A_a2y, A_a2z, \
A_b1x, A_b1y, A_b1z, A_b2x, A_b2y, A_b2z = sp.symbols(
    'Aa1x Aa1y Aa1z Aa2x Aa2y Aa2z '
    'Ab1x Ab1y Ab1z Ab2x Ab2y Ab2z'
)
Aa1, Aa2, Ab1, Ab2 = sp.symbols('Aa1 Aa2 Ab1 Ab2')

def convert_to_isotropic(H_sym: sp.Matrix) -> sp.Matrix:
    """
    Given H_sym(A_a1x, A_a1y, A_a1z, ..., A_b2z),
    substitute:
      A_a1x, A_a1y, A_a1z -> Aa1
      A_a2x, A_a2y, A_a2z -> Aa2
      A_b1x, A_b1y, A_b1z -> Ab1
      A_b2x, A_b2y, A_b2z -> Ab2
    and returns the simplified matrix H_iso(Aa1,Aa2,Ab1,Ab2).
    """
    subs_map = {
        A_a1x: Aa1, A_a1y: Aa1, A_a1z: Aa1,
        A_a2x: Aa2, A_a2y: Aa2, A_a2z: Aa2,
        A_b1x: Ab1, A_b1y: Ab1, A_b1z: Ab1,
        A_b2x: Ab2, A_b2y: Ab2, A_b2z: Ab2,
    }
    return sp.simplify(H_sym.subs(subs_map))


# Writes matrices out to .tex  
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
    path = f"./{filename}"

    with open(path, "w") as f:
        f.write(doc)
    return path


if __name__ == '__main__':
    import pickle 

    hf = HyperfineHamiltonian()
    H_iso = convert_to_isotropic(hf.H_coup)

    fname = "/Users/devaldeliwala/nasa/Hamiltonian/hyperfine.pickle" 
    with open(fname, "wb") as f: 
        pickle.dump(H_iso, f)
        print(f"Hyperfine Hamiltonian saved to {fname}.")

