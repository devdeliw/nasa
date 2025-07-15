from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import sympy as sp
import numpy as np
from numpy import ndarray
from numpy.linalg import eigvals, eigvalsh, solve
from scipy.linalg import kron  

def _to_numeric(M: sp.Matrix | ndarray, subs: Mapping[sp.Symbol, float]) -> ndarray:
    """
    Converts a sympy.Matrix into a numerical numpy.ndarray matrix by 
    substituting relevant parameters. 

    Used in building the numerical sylvester equation. 

    """
    if isinstance(M, sp.Matrix):
        return np.array(M.subs(subs).evalf(), dtype=complex)
    return np.array(M, dtype=complex)

@dataclass
class SteadyStateSLESolver:
    """
    Build the singlet-triplet Liouvillian for a 2-electron + 2-nucleus (I=1/2)
    spin system numerically. Returns a callable

        * rho_numeric(**params) -> 16x16 numpy array (trace=1)

    I use a modified model for Spin Dependent Recombination (SDR) from Hansen 
    and Pedersen. I solve
 
    # unicode 
    0 = -(i/ħ)[H,ρ]
        - ½ (k_S + k_D) {Λ_S,ρ}
        - ½ k_D         {Λ_T,ρ}
        + (p/16) Γ.

    Args: 
        * H_sym: sp.Matrix                              (16, 16)
            * Symbolic Hamiltonian to solve 
            * Will be either static or static + RWA 
        * Lambda_S, Lambda_T: sp.Matrix | np.ndarray    (16, 16)
            * Singlet and Triplet projection Operators 
        * Gamma: sp.Matrix | np.ndarray                 (16, 16)
            * 16x16 identity for generation term
        * N: int 
            * Dimension-size of Hilbert Space 
            * 2-electron + 2-nuclei -> 16 Dimensional Space. 
    """
    
    H_sym:      sp.Matrix
    Lambda_S:   sp.Matrix | np.ndarray
    Lambda_T:   sp.Matrix | np.ndarray
    Gamma:      sp.Matrix | np.ndarray
    N:          int = field(default=16, init=False)
    _free_params: Sequence[sp.Symbol] = field(init=False, repr=False)

    def __post_init__(self):
        self._check_dimensions()
        self._free_params = tuple(sorted(self.H_sym.free_symbols, key=lambda s: s.name))

        # precompute static numeric Hamiltonians 
        self.LS    = np.array(_to_numeric(self.Lambda_S, {}), dtype=complex)
        self.LT    = np.array(_to_numeric(self.Lambda_T, {}), dtype=complex)

    def rho(self, *, params: Mapping[str, float] | None = None) -> ndarray:
        """
        Returns steady-state numerical density matrix (16x16, complex Hermitian)
        
            * All parameters are specified through `params`. 
            * Generation and Dissociation rates are specified through 
              params['p'], params['k_S'], params['k_d']. 

        Args: 
            * params: dict[str, float]
                * Numeric substitutions for every parameter. 
                * see utils.load_params::load_params() 

        """

        if params is None:
            params = {}
        subs = {sp.Symbol(k): v for k, v in params.items()}
        self._check_param_completeness(subs)

        # build sylvester operator
        A = self._build_A(
            k_S=params["k_S"], 
            k_D=params["k_D"], 
            hbar=params["hbar"], 
            subs=subs
        )
        self._spectral_check(A) # no eigenvalues ~ 0
        B = A.conj().T  

        # generation term 
        g = self._build_g(p=params['p'], subs=subs)

        I = np.eye(self.N, dtype=complex)
        M = kron(I, A) + kron(B.T, I)
        self._structure_checks(M, A, B)

        # solve vectorized density matrix 
        # fortran column ordering
        vec_rho = solve(M, -g.reshape(-1, 1, order="F"))
        rho = vec_rho.reshape(self.N, self.N, order="F")
        rho = self._postprocess(rho)
        return rho

    def _check_dimensions(self):
        for name, M in (
            ("H", self.H_sym),
            ("Lambda_S", self.Lambda_S),
            ("Lambda_T", self.Lambda_T),
            ("Gamma", self.Gamma),
        ):
            if M.shape != (self.N, self.N):
                raise ValueError(f"{name} must be {self.N}x{self.N}; got {M.shape}.")

    def _check_param_completeness(self, subs: Mapping[sp.Symbol, float]):
        missing = set(self._free_params) - set(subs)
        if missing:
            raise ValueError(
                "Numeric values missing for: " + ", ".join(s.name for s in missing)
            )

    # build numerical A operator in sylvester eq. 
    def _build_A(
            self,
            *,
            k_S: float,
            k_D: float,
            hbar: float,
            subs: Mapping[sp.Symbol, float]
    ) -> ndarray:
        H  = np.array(self.H_sym.subs(subs).evalf(), dtype=complex)
        return (
            -(1j / hbar) * H
            - 0.5 * (k_S + k_D) * self.LS
            - 0.5 * k_D * self.LT
        )

    # generation term 
    def _build_g(self, *, p: float, subs: Mapping[sp.Symbol, float]) -> ndarray:
        # This is static, but Gamma doesn't necessarily *have* 
        # to be the identity like it is here. 
        Gamma = np.array(_to_numeric(self.Gamma, subs), dtype=complex)
        return (p / 16.0) * Gamma

    def _spectral_check(self, A: ndarray):
        lam = eigvals(A)
        if np.any(np.abs(np.real(lam)) < 1e-10):
            raise ValueError(
                "steady state solution must be unique. some eigenvalues are near 0."
            )
        return lam

    def _structure_checks(self, M: ndarray, A: ndarray, B: ndarray):
        N2 = self.N * self.N
        if M.shape != (N2, N2):
            raise RuntimeError("kronecker sum matrix M has wrong dimensions.")
        if np.linalg.matrix_rank(M) < N2:
            raise RuntimeError("M is singular; steady state not unique or absent.")
        if not np.allclose(B, A.conj().T):
            raise RuntimeError("B \\neq A^dagger")

    def _postprocess(self, rho: ndarray) -> ndarray:
        # enforce hermiticity & normalization 
        rho = 0.5 * (rho + rho.conj().T)
        tr = np.trace(rho)
        if not np.isclose(tr, 1.0, atol=1e-8):
            rho /= tr
        # positivity, with some small eigenwrinkles allowed
        evals = eigvalsh(rho)
        if np.any(evals < -1e-8):
            raise RuntimeError("rho not positive semi-definite (min eigenvalue < 0).")
        return rho


