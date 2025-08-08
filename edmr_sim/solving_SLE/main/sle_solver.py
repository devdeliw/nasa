# This code was written by Deval Deliwala 
# NASA Glenn Research Center 

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping

import sympy as sp
import numpy as np
from numpy import ndarray
from numpy.linalg import eigvals, eigvalsh
from scipy.linalg import kron, solve_sylvester

def _to_numeric(
        M: sp.Matrix | ndarray, 
        subs: Mapping[sp.Symbol, float]
) -> ndarray:
    """
    Converts a sympy.Matrix into a numerical numpy.ndarray matrix by 
    substituting relevant parameters. 

    Used in building the numerical sylvester equation. 

    """
    if isinstance(M, sp.Matrix):
        return np.array(
            M.subs(subs).evalf(), 
            dtype=complex
        )
    return M if M.dtype == np.complex128 else M.astype(np.complex128)

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
    parameters: dict[str, float] 
    init_check: bool = True
    N:          int = field(default=16, init=False)

    def __post_init__(self):
        self._check_dimensions()

        free_symbols = self.H_sym.free_symbols
        sym_map: dict[str, sp.Symbol] = {s.name: s for s in free_symbols}

        # Verify parameters are complete
        missing = sym_map.keys() - self.parameters.keys()
        if missing == {"B0"}: 
            # test value for B0 
            # only used for tests, gets overwritten in SLE solves. 
            self.parameters["B0"] = 1.0  
        elif missing: 
            raise ValueError(f"Missing parameter values for {missing}")

        # We pre-lambidfy the Hamiltonian with every parameter but B0
        # B0 is the only parameter that changes
        B0 = sym_map.pop("B0") 
        const_subs = {
            sym: float(self.parameters[name]) 
            for name, sym in sym_map.items()
        }
        H_reduced = self.H_sym.subs(const_subs) 
        self._H_func = sp.lambdify((B0, ), H_reduced, modules="numpy") 

        # static matrices
        self.LS    = np.array(_to_numeric(self.Lambda_S, {}), dtype=complex)
        self.LT    = np.array(_to_numeric(self.Lambda_T, {}), dtype=complex)
        self.g     = -(self.parameters["p"]/16) * _to_numeric(self.Gamma, {})

        # preliminary checks 
        if self.init_check: self.preliminary_checks()

    def preliminary_checks(self): 
        """ 
        Performs import matrix checks  
            * positivity, 
            * structure, 
            * nonsingularity 

        This would be costly to run everytime a new rho is calculated. 
        So we just calculate it once in self.__post_init__() 

        Only thing that changes afterward is the value of B0, which 
        won't affect any check. 

        """
        # Perform checks from input params. 
        # Only B changes after -- which won't affect anything. 
        _A0 = self._build_A(
            k_S=self.parameters["k_S"], 
            k_D=self.parameters["k_D"], 
            hbar=self.parameters["hbar"], 
            B=1.0, # test
        )
        self._spectral_check(_A0)
        _B0 = _A0.conj().T
        _M0 = kron(np.eye(self.N), _A0) + kron(_B0.T, np.eye(self.N))
        self._structure_checks(M=_M0, A=_A0, B=_B0)

        # positivitity 
        rho0 = self.rho(parameters=self.parameters)
        evals = eigvalsh(rho0)
        if np.any(evals < -1e-8):
            raise ValueError(
                "rho not positive semi-definite (min eigenvalue < 0)."
            )

    def rho(self, *, parameters: Mapping[str, float] | None = None) -> ndarray:
        """
        Returns steady-state numerical density matrix 
        (16x16, complex Hermitian)
        
        * All parameters are specified through `params`. 
        * Generation and Dissociation rates are specified through 
            params['p'], params['k_S'], params['k_d']. 

        Args: 
            * params: dict[str, float]
                * Numeric substitutions for every parameter. 
                * see utils.load_params::load_params() 

        """

        if parameters is None:
            parameters = {}

        # build sylvester operator
        A = self._build_A(
            k_S  = parameters["k_S"], 
            k_D  = parameters["k_D"], 
            hbar = parameters["hbar"], 
            B    = parameters["B0"],
        )
        B = A.conj().T  

        # solve vectorized density matrix 
        rho = solve_sylvester(A, B, self.g) 

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
                raise ValueError(
                    f"{name} must be {self.N}x{self.N}; "
                    f"got {M.shape}."
                )

    # sylvester operator 
    def _build_A(
            self,
            *,
            k_S: float,
            k_D: float,
            hbar: float,
            B: float,
    ) -> ndarray:

        H_num = self._H_func(B)
        return (
            -(1j / hbar) * H_num
            - 0.5 * (k_S + k_D) * self.LS
            - 0.5 * k_D * self.LT
        )

    # generation term 
    def _build_g(self, *, p: float) -> ndarray:
        Gamma = _to_numeric(self.Gamma, {})
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
        return rho


