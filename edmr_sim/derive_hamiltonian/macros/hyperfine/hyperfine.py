# This code was written by Deval Deliwala 
# NASA Glenn Research Center 

import sympy as sp
import textwrap
import pickle

from pathlib import Path

HAMILTONIAN_DIRECTORY = Path(__file__).parent.parent.parent / "pickle"

class HyperfineHamiltonian:
    """
    Builds the two-electronic + two-nuclear Hyperfine Hamiltonian 
    in the Zeeman Basis. 

    Then converts to the coupled-electronic + nuclear-zeeman basis
    via Clebsch-Gordan transformation. 

    """

    def __init__(self):
        # symbols for isotropic hyperfine constants
        self.A = {}  
        for e in ('a', 'b'):
            for n in ('1', '2'):
                for comp in ('x', 'y', 'z'):
                    key = f'A{e}{n}{comp}'
                    self.A[key.upper()] = sp.symbols(key)

        # Zeeman Basis
        self.basis_ze = self._generate_zeeman_basis()               
        self.index_ze = {ket: i for i, ket in enumerate(self.basis_ze)}
        self.H_ze     = self._build_zeeman_matrix()              # 16x16

        # Clebsch-Gordan Unitary Transformation
        self.W = self._build_cg_unitary()                        # 16x16

        # Coupled Basis Hyperfine Hamiltonian
        self.H_coup = sp.simplify(self.W * self.H_ze * self.W.H) # type: ignore

    def _generate_zeeman_basis(self):
        """ 
        Builds the len16 Zeeman Basis. 

        The Zeeman Basis for the two-electronic + two-nuclear system as 
        {|m_a, m_b>|m_I1, m_I2>}. 

        a, b    are the two electrons 
        I1, I2  are the two nuclei

        m is either {+1/2, -1/2}.

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
        Calculates the action of H_HF|m_a, m_b>|m_I1, m_I2> 
        Returns dict[new_ket, hyperfine_coefficient]. 

        """
        
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

        # off-diagonal flip-flop S_x I_x + S_y I_y -> (A_x - A_y)/4 
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
        """ 
        Builds the full 16x16 Hyperfine Hamiltonian 
        in the ZEEMAN BASIS. 

        """
        
        size = len(self.basis_ze)
        H = sp.MutableDenseMatrix(size, size, lambda *_: 0)
        for j, ket in enumerate(self.basis_ze):
            action = self._action_on_ket(ket)
            for new_ket, coeff in action.items():
                i = self.index_ze[new_ket]
                H[i, j] = coeff
        return H.as_immutable()

    def _build_cg_unitary(self):
        """ 
        Builds the unitary two-electronic Clebsch-Gordan transformation. 

        This will transform the zeeman electron-part of the Hamiltonian into 
        the coupled |s, m> basis while leaving the nuclei terms in the zeeman basis. 

        """
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
        """ 
        Returns the full 16x16 Hyperfine Hamiltonian in the correct 
        coupled electronic - nuclear zeeman basis |s, m>|m_I1, m_I2>. 

        """
        return self.H_coup

A_a1x, A_a1y, A_a1z, A_a2x, A_a2y, A_a2z, \
A_b1x, A_b1y, A_b1z, A_b2x, A_b2y, A_b2z = sp.symbols(
    'Aa1x Aa1y Aa1z Aa2x Aa2y Aa2z Ab1x Ab1y Ab1z Ab2x Ab2y Ab2z'
)
Aa1, Aa2, Ab1, Ab2 = sp.symbols('Aa1 Aa2 Ab1 Ab2')

def convert_to_isotropic(H_sym: sp.Matrix) -> sp.Matrix:
    import numpy as np 
    """ 
    Converts Hamiltonian to being isotropic for each nuclei. 

    Aa1-2x-y-z -> Aa1-2 
    Ab1-2x-y-z -> Ab1-2 
    """

    subs_map = {
        A_a1x: Aa1, A_a1y: Aa1, A_a1z: Aa1,
        A_a2x: Aa2, A_a2y: Aa2, A_a2z: Aa2,
        A_b1x: Ab1, A_b1y: Ab1, A_b1z: Ab1,
        A_b2x: Ab2, A_b2y: Ab2, A_b2z: Ab2,
    }
    H_iso = sp.simplify(H_sym.subs(subs_map))

    # verify numerical hermicity 
    H_np = np.array(
        H_iso.subs({
            Aa1: 1.46e-7, Ab1: 1.46e-7,
            Aa2: 2.81e-7, Ab2: 2.81e-7
        }).evalf().tolist(),
        dtype=complex
    )
    assert (np.allclose(H_np, H_np.conj().T, atol=1e-12)), (
            "Hyperfine Hamiltonian is not Hermitian!"
    )
    return H_iso

def convert_to_secular(H_sym: sp.Matrix) -> sp.Matrix:
    """ 
    Converts Hamiltonian to being secular (only include z terms)    
    """
    zero_transverse = {
        A_a1x: 0, A_a1y: 0,
        A_a2x: 0, A_a2y: 0,
        A_b1x: 0, A_b1y: 0,
        A_b2x: 0, A_b2y: 0,
    }
    H_sec_aniso = H_sym.subs(zero_transverse)
    return convert_to_isotropic(H_sec_aniso)

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
    hf = HyperfineHamiltonian()
    H_iso       = convert_to_isotropic(hf.H_coup)
    H_secular   = convert_to_secular(hf.H_coup)

    fname = HAMILTONIAN_DIRECTORY / "hyperfine.pickle"
    
    with open(fname, "wb") as f:
        pickle.dump(H_iso, f)
        print(f"Hyperfine Hamiltonian saved to {fname}.")

    fname = HAMILTONIAN_DIRECTORY / "hyperfine_sec.pickle" 
    with open(fname, "wb") as f: 
        pickle.dump(H_secular, f) 
        print(f"Secular Hyperfine Hamiltonian saved to {fname}")
