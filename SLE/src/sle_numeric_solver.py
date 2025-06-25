import sympy as sp
import numpy as np
import logging, time

logging.basicConfig(level=logging.INFO)

from sympy.physics.quantum import Dagger

__all__ = ["SLE"]

class SLE:
    """
    Build the singlet-triplet Liouvillian for a 2-electron + 2-nucleus (I=1/2)
    spin system **numerically**.  The class now returns a ready-to-use callable

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

    def __init__(self, H_sym: sp.Matrix, verbose: bool = True) -> None:
        self.verbose = verbose
        self.log = logging.getLogger(__name__)
        if self.verbose: self.log.info(" Initializing numeric‑first SLE solver …")

        if H_sym.shape != (16, 16):
            raise ValueError("H_sym must be 16 x 16")
        
        self.H_sym = H_sym
        if self.verbose: self.log.info(f"     * Hamiltonian shape: {H_sym.shape}")

        self.ham_names = [
            "B0","g_e","mu_B","g_n1","g_n2","mu_N",
            "Aa1","Aa2","Ab1","Ab2","D1","D2","J",
        ]
        self.sle_names = ["k_S","k_D","p","hbar"]
        self.sym_list  = [sp.Symbol(n) for n in (self.ham_names + self.sle_names)]
        ( self.k_S, self.k_D, self.p, self.hbar ) = [sp.Symbol(n) for n in self.sle_names]

        # build static symbolic operators
        self.Lambda_S, self.Lambda_T = self._build_projectors()
        self.Gamma = sp.eye(16)
        self.L_super_sym, self.b_vec_sym = self._build_liouvillian()

        # lambdify
        if self.verbose: self.log.info("     * Lambdifying L_super and b_vec …")
        t0 = time.time()
        self.L_func = sp.lambdify(self.sym_list, self.L_super_sym, modules="numpy")
        self.b_func = sp.lambdify(self.sym_list, self.b_vec_sym, modules="numpy")
        if self.verbose: self.log.info(f"        * done in {time.time()-t0:.2f}s")

    def make_density_func(self):
        """
        Return rho_numeric(**param_dict) -> density matrix (numpy 16x16).
        
        """

        name_order = self.sym_list  
        Lf, bf = self.L_func, self.b_func

        def rho_numeric(**params):
            # build ordered arg list
            try:
                argvals = [params[n] for n in (self.ham_names + self.sle_names)]
            except KeyError as err:
                missing = err.args[0]
                raise KeyError(f"parameter '{missing}' not supplied")

            # numeric matrices
            A = np.asarray(Lf(*argvals), dtype=np.complex128)
            b = np.asarray(bf(*argvals), dtype=np.complex128).ravel()

            t0 = time.time()
            self.log.info(f"     * Numerically solving Rho {b.shape}...")
            rho_vec = np.linalg.solve(A, b)
            self.log.info(f"        * done in {time.time() - t0:.4f}s")

            # 256x1 -> 16x16
            rho = rho_vec.reshape(16, 16)

            # normalize
            rho /= np.trace(rho)
            return rho

        return rho_numeric

    @staticmethod
    def _singlet_triplet_projectors_electron():
        up = sp.Matrix([1, 0]); dn = sp.Matrix([0, 1])
        basis = [sp.kronecker_product(s1, s2) for s1 in (up, dn) for s2 in (up, dn)]
        S  = (basis[1] - basis[2]) / sp.sqrt(2)                                                             # type: ignore
        Tp = basis[0]       
        T0 = (basis[1] + basis[2]) / sp.sqrt(2)                                                             # type: ignore
        Tm = basis[3]
        Lambda_S_e = S  * Dagger(S)
        Lambda_T_e = Tp * Dagger(Tp) + T0 * Dagger(T0) + Tm * Dagger(Tm)                                    # type: ignore
        return Lambda_S_e, Lambda_T_e

    def _build_projectors(self):
        LS_e, LT_e = self._singlet_triplet_projectors_electron()
        I_nuc  = sp.eye(2)
        I_full = sp.kronecker_product(I_nuc, I_nuc)
        LS = sp.kronecker_product(LS_e, I_full)
        LT = sp.kronecker_product(LT_e, I_full)
        return LS, LT

    def _build_liouvillian(self):
        if self.verbose: self.log.info("     * Building Liouvillian symbolically …")
        n = 16; I_n = sp.eye(n); kron = sp.kronecker_product

        L_H = (-sp.I / self.hbar) * (kron(I_n, self.H_sym) - kron(self.H_sym.T, I_n))                       # type: ignore
        LS2, LT2 = self.Lambda_S, self.Lambda_T
        L_S = kron(self.Lambda_S.T, self.Lambda_S) - sp.Rational(1,2)*(kron(I_n, LS2.T) + kron(LS2, I_n))   # type: ignore
        L_T = kron(self.Lambda_T.T, self.Lambda_T) - sp.Rational(1,2)*(kron(I_n, LT2.T) + kron(LT2, I_n))   # type: ignore

        L_super = L_H - sp.Rational(1,2)*(self.k_S + self.k_D)*L_S - sp.Rational(1,2)*self.k_D*L_T          # type: ignore
        b_vec   = -(self.p/16) * sp.Matrix(self.Gamma).reshape(n**2, 1)                                     # type: ignore
        return L_super, b_vec


if __name__ == "__main__":
    from utils._load_hamiltonian import _load_spin
    import pathlib, time 
    import cloudpickle as pickle 

    H = _load_spin()                      
    sle = SLE(H, verbose=True)
    rho_fn = sle.make_density_func()

    # quick test solve
    params = {
        "B0":0.3, "g_e":2.0023, "mu_B":5.788e-5,
        "g_n1":-1.11, "g_n2":1.40, "mu_N":3.15e-8,
        "Aa1":4.14e-8, "Aa2":4.14e-8, "Ab1":4.14e-8, "Ab2":4.14e-8,
        "D1":1e-9, "D2":1e-9, "J":4.1e-8,
        "k_S":1e5, "k_D":1e6, "p":1e3, "hbar":1.0545718e-34,
    }

    t0 = time.time()
    rho_num = rho_fn(**params)
    dt = time.time()-t0

    print(f"Numeric solve: {dt:4f}s. Trace={np.trace(rho_num).round(2)}")

    # store functional      
    outdir = pathlib.Path.home()/"nasa/SLE/pickle"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir/"density_func.pickle", "wb") as fd:
        pickle.dump(rho_fn, fd)
