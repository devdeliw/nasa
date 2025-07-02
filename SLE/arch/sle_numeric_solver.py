import sympy as sp
import numpy as np
import logging, time
from sympy.physics.quantum import Dagger

logging.basicConfig(level=logging.INFO)

__all__ = ["SLE_NUMERIC"]


def _assert_hermitian(mat, name, tol=1e-12):
    """
    Raise if *mat* is not Hermitian.
    Works for either a SymPy or a NumPy matrix.

    """
    if isinstance(mat, sp.MatrixBase):
        diff = (mat - Dagger(mat)).simplify()
        for i in range(diff.rows):
            for j in range(diff.cols):
                if not diff[i, j].equals(0):
                    raise ValueError(
                        f"{name} symbolic check failed at ({i},{j}): {diff[i, j]}"
                    )
        return

    diff = mat - mat.conj().T
    if np.amax(np.abs(diff)) > tol:
        raise ValueError(
            f"{name} not Hermitian – max|X−X†|={np.amax(np.abs(diff)):.3e}"
        )


def _assert_super_maps_hermitian(L, n, name, trials=6, tol=1e-12):
    """
    Randomly test that the super-operator L maps every Hermitian rho to a
    Hermitian output in vectorized form.

    """
    for i in range(trials):
        rnd = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        rho = rnd + rnd.conj().T
        out = L @ rho.reshape(n * n, order="F")
        rout = out.reshape(n, n, order="F")
        if np.amax(np.abs(rout - rout.conj().T)) > tol:
            raise ValueError(
                f"{name} breaks Hermiticity on trial {i} – "
                f"max|rho_out−rho_out†|={np.amax(np.abs(rout - rout.conj().T)):.3e}"
            )


class SLE_NUMERIC:
    """
    Build the singlet-triplet Liouvillian for a 2-electron + 2-nucleus (I=1/2)
    spin system numerically. Returns a callable

        rho_numeric(**params) -> 16x16 numpy array (trace=1)

    I use a modified model for Spin Dependent Recombination (SDR) from Hansen 
    and Pedersen. I solve

    frac{d rho}{dt} = -frac{i}{hbar}[H, rho] - frac{1}{2}(ks + kd)
    {Lambda_s, rho} - frac{kd}{2}{Lambda_T, rho} + frac{p}{16}Gamma
    = 0 (steady state) 
    
    # unicode 
    0 = -(i/ħ)[H,ρ]
        - ½ (k_S + k_D) {Λ_S, ρ}
        - ½ k_D         {Λ_T, ρ}
        + (p/16) Γ.

    Lambda_S and Lambda_T are the singlet and triplet projection operators 
    acting on the 2-electron 4 x 4 subspace and the identity on the nuclear subspace. 
    Gamma is the 16x16 identity. 

    """

    def __init__(self, H_sym: sp.Matrix, verbose: bool = False) -> None:
        self.verbose = verbose
        self.log = logging.getLogger(__name__)
        if self.verbose:
            self.log.info(" Initializing numerical SLE solver")

        if H_sym.shape != (16, 16):
            raise ValueError("H_sym must be 16 x 16")

        # master parameter list
        self.ham_names = [
            "B0", "g_e", "mu_B", "g_n1", "g_n2", "mu_N",
            "Aa1", "Aa2", "Ab1", "Ab2", "D1", "D2", "J",
        ]
        self.sle_names = ["k_S", "k_D", "p", "hbar"]

        # single Symbol object for each name
        self.sym_list = [sp.Symbol(n, real=True) for n in (self.ham_names + self.sle_names)]
        name_map = {s.name: s for s in self.sym_list}

        self.k_S  = name_map["k_S"]
        self.k_D  = name_map["k_D"]
        self.p    = name_map["p"]
        self.hbar = name_map["hbar"]

        # harmonize symbols in H_sym
        self.H_sym = H_sym.xreplace({old: name_map.get(old.name, old)
                                     for old in H_sym.free_symbols})

        # projectors
        self.Lambda_S, self.Lambda_T = self._build_projectors()
        _assert_hermitian(self.Lambda_S, "Lambda_S")
        _assert_hermitian(self.Lambda_T, "Lambda_T")

        # Liouvillian and inhomogeneous term
        self.Gamma = sp.eye(16)
        self.L_super_sym, self.b_vec_sym = self._build_liouvillian()

        # lambdify to NumPy
        if self.verbose:
            self.log.info("     * Lambdifying operators to NumPy")
        t0 = time.time()
        self.L_func = sp.lambdify(self.sym_list, self.L_super_sym, modules="numpy")
        self.b_func = sp.lambdify(self.sym_list, self.b_vec_sym, modules="numpy")
        if self.verbose:
            self.log.info(f"        * done in {time.time() - t0:.2f}s")

        # quick Hermiticity sanity check
        baseline = [1 if s.name == "hbar" else 0 for s in self.sym_list]
        L_num = np.asarray(self.L_func(*baseline), dtype=np.complex128)
        _assert_super_maps_hermitian(L_num, 16, "L_super (numeric)")

    def make_density_func(self):
        """
        Return a function rho_numeric(**params) that computes
        the 16 x 16 steady-state density matrix.

        """

        Lf, bf = self.L_func, self.b_func

        def rho_numeric(**params):
            try:
                argvals = [params[n] for n in (self.ham_names + self.sle_names)]
            except KeyError as err:
                raise KeyError(f"parameter '{err.args[0]}' not supplied") from None

            A = np.asarray(Lf(*argvals), dtype=np.complex128)
            b = np.asarray(bf(*argvals), dtype=np.complex128).ravel()

            rho_vec = np.linalg.solve(A, b)
            rho = rho_vec.reshape(16, 16, order="F")
            rho = 0.5 * (rho + rho.conj().T)
            rho /= np.trace(rho)
            return rho

        return rho_numeric

    # helpers
    @staticmethod
    def _singlet_triplet_projectors_electron():
        up, dn = sp.Matrix([1, 0]), sp.Matrix([0, 1])
        basis = [sp.kronecker_product(s1, s2) for s1 in (up, dn) for s2 in (up, dn)]
        S  = (basis[1] - basis[2]) / sp.sqrt(2)                             # type: ignore
        Tp = basis[0]
        T0 = (basis[1] + basis[2]) / sp.sqrt(2)                             # type: ignore
        Tm = basis[3]
        Lambda_S_e = S  * Dagger(S)
        Lambda_T_e = Tp * Dagger(Tp) + T0 * Dagger(T0) + Tm * Dagger(Tm)    # type: ignore
        return Lambda_S_e, Lambda_T_e

    def _build_projectors(self):
        LS_e, LT_e = self._singlet_triplet_projectors_electron()
        I_nuc  = sp.eye(2)
        I_full = sp.kronecker_product(I_nuc, I_nuc)
        LS = sp.kronecker_product(LS_e, I_full)
        LT = sp.kronecker_product(LT_e, I_full)
        return LS, LT

    def _build_liouvillian(self):
        n = 16
        I_n  = sp.eye(n)
        kron = sp.kronecker_product

        if self.verbose: 
            self.log.info("     * Building Liouvillian symbolically")
        L_H = (-sp.I / self.hbar) * (
            kron(I_n, self.H_sym) - kron(Dagger(self.H_sym), I_n)       # type: ignore
        )

        A_S, A_T = self.Lambda_S, self.Lambda_T
        L_S = kron(A_S.T, A_S) - sp.Rational(1, 2) * (                  # type: ignore
            kron(I_n, A_S) + kron(A_S.T, I_n)                           # type: ignore
        )
        L_T = kron(A_T.T, A_T) - sp.Rational(1, 2) * (                  # type: ignore
            kron(I_n, A_T) + kron(A_T.T, I_n)                           # type: ignore
        )

        L_super = (
            L_H
            - sp.Rational(1, 2) * (self.k_S + self.k_D) * L_S           # type: ignore
            - sp.Rational(1, 2) * self.k_D * L_T                        # type: ignore
        )
        b_vec = -(self.p / 16) * sp.Matrix(self.Gamma).reshape(n ** 2, 1)   # type: ignore
        return L_super, b_vec


if __name__ == "__main__":
    from utils._load_hamiltonian import _load_spin
    import pathlib, cloudpickle as pickle

    H = _load_spin()
    sle = SLE_NUMERIC(H, verbose=True)
    rho_fn = sle.make_density_func()

    outdir = pathlib.Path.home() / "nasa/SLE/pickle"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "density_func.pickle", "wb") as fd:
        pickle.dump(rho_fn, fd)
        print("saved to", outdir / "density_func.pickle")
