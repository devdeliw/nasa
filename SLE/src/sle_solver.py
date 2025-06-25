import sympy as sp
from sympy.physics.quantum import Dagger

__all__ = ["SLE"]

class SLE():
    """
    Builds the singlet–triplet Liouvillian for a 2‑electron/2‑nucleus (I = 1/2)
    spin system and solves the steady‑state stochastic Liouville equation (SLE). 

    I use a modified model for Spin Dependent Recombination (SDR) from Hansen 
    and Pedersen. 

    frac{d rho}{dt} = -frac{i}{hbar}[H, rho] - frac{1}{2}(ks + kd)
    {Lambda_s, rho} - frac{kd}{2}{Lambda_T, rho} + frac{p}{16}Gamma
    = 0 (steady state) 
    
    # unicode 
    0 = -(i/ħ)[H,ρ]
        - ½ (k_S + k_D) {Λ_S, ρ}
        - ½ k_D         {Λ_T, ρ}
        + (p/16) Γ.

    Lambda_S and Lambda_T are the singlet and triplet projection operators 
    acting on the 2-electron 4x4 subspace and the identity on the nuclear subspace. 
    Gamma is the 16x16 identity. 

    """

    def __init__(self, H_sym: sp.Matrix):
        """
        Args: 
            * H_sym : sympy.Matrix (16 x 16)
                * Symbolic spin Hamiltonian containing the symbols
                * B0, g_e, mu_B, g_n1, g_n2, mu_N, 
                * Aa1, Aa2, Ab1, Ab2, D1, D2, J.
        """

        if H_sym.shape != (16, 16):
            raise ValueError("H_sym must be 16 x 16")
        self.H = H_sym

        # symbolic constants in SLE 
        self.k_S, self.k_D, self.p, self.hbar = sp.symbols("k_S k_D p hbar", positive=True, real=True)

        # projection operators 
        self._Lambda_S, self._Lambda_T = self._build_projectors()
        self._Gamma = sp.eye(16)

        # build the Liouvillian superoperator L (256 x 256) and RHS vector b.
        self._L_super, self._b_vec = self._build_liouvillian()

    def L(self):
        """
        Return Liouvillian superoperator (256 x 256) symbolically.

        """
        return self._L_super

    def rhs(self):
        """
        Return RHS vector (-p/16 vec Gamma).

        """
        return self._b_vec

    def solve_symbolic(self, normalize=True):
        """
        Solve for vec(rho_ss) symbolically.

        Args: 
        * normalize : bool, default True
            * If True, divides rho_ss by Tr(rho_ss) so that the density matrix is
            * trace=1.
        """

        r_vec = self._L_super.LUsolve(self._b_vec)  # vec(rho)
        rho = sp.Matrix(r_vec).reshape(16, 16)
        if normalize:
            rho /= sp.trace(rho)
        return rho

    def lambdify(self, free_symbols=None, modules="numpy"):
        """
        Return a fast numerical callable for rho_ss after giving numeric values.

        Example: 
            >>> L = SLE(H_sym)
            >>> f = L.lambdify()
            >>> rho_numeric = f(B0=0.345, g_e=2.0023, ... , k_S=1e5, k_D=1e6, p=1e3)

        """

        rho_sym = self.solve_symbolic()
        if free_symbols is None:
            free_symbols = sorted(rho_sym.free_symbols, key=lambda s: s.name)
        return sp.lambdify(free_symbols, rho_sym, modules=modules)

    @staticmethod
    def _singlet_triplet_projectors_electron():
        """
        Return (Lambda_S, Lamda_T) in electron‑spin 4 x 4 space.

        """

        # computational basis |↑↑>, |↑↓>, |↓↑>, |↓↓>
        up, dn = sp.Matrix([1, 0]), sp.Matrix([0, 1])
        basis_e = [sp.kronecker_product(s1, s2) for s1 in (up, dn) for s2 in (up, dn)]  

        # basis states 
        S  = (basis_e[1] - basis_e[2]) / sp.sqrt(2)                  # (|↑↓> - |↓↑>)/√2
        Tp = basis_e[0]                                              # |↑↑>
        T0 = (basis_e[1] + basis_e[2]) / sp.sqrt(2)                  # (|↑↓> + |↓↑>)/√2
        Tm = basis_e[3]                                              # |↓↓>

        Lambda_S = S * Dagger(S)                                          # projector |S><S|
        Lambda_T = Tp * Dagger(Tp) + T0 * Dagger(T0) + Tm * Dagger(Tm)    # sum over triplet subspace
        return Lambda_S, Lambda_T

    def _build_projectors(self):
        """
        Embed electron projectors into full 16 × 16 space (otimes I_nuclei).

        """

        Lambda_S_e, Lambda_T_e = self._singlet_triplet_projectors_electron()   # 4 x 4 each
        I_nuc = sp.eye(2)                                            # each nucleus is spin-1/2
        I_full_nuc = sp.kronecker_product(I_nuc, I_nuc)              # 4 x 4 nuclear identity
        Lambda_S = sp.kronecker_product(Lambda_S_e, I_full_nuc)                # 4 x 4 otimes 4 x 4 -> 16 x 16
        Lambda_T = sp.kronecker_product(Lambda_T_e, I_full_nuc)
        return Lambda_S, Lambda_T

    def _build_liouvillian(self):
        """
        Return (L_super, b_vec) where L_super is 256 x 256 and b_vec 256 x 1.

        """

        n = 16
        I_n = sp.eye(n)
        H  = self.H

        kron = sp.kronecker_product

        # (i/hbar)[H, rho] superoperator
        L_H = (-sp.I / self.hbar) * (kron(I_n, H) - kron(H.T, I_n))

        # anticommutator 
        Lambda_S, Lambda_T = self._Lambda_S, self._Lambda_T
        LambdaS2 = Lambda_S # P^2 = P
        LambdaT2 = Lambda_T # P^2 = P

        L_S = kron(Lambda_S.T, Lambda_S) - sp.Rational(1, 2) * (kron(I_n, LambdaS2.T) + kron(LambdaS2, I_n))
        L_T = kron(Lambda_T.T, Lambda_youT) - sp.Rational(1, 2) * (kron(I_n, LambdaT2.T) + kron(LambdaT2, I_n))

        # final Liouvillian
        L_super = (
            L_H
            - sp.Rational(1, 2) * (self.k_S + self.k_D) * L_S
            - sp.Rational(1, 2) * self.k_D * L_T
        )

        # p/16 * vec(Gamma)
        Gamma = self._Gamma
        b_vec = -(self.p / 16) * sp.Matrix(Gamma).reshape(n**2, 1)

        return L_super, b_vec

