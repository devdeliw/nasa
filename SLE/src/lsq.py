import numpy as np, pickle
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from run_solver import P_ORDER
from pathlib import Path

from utils.bounds import lower, upper
from sle_model import edmr_spectra
from run_solver import load_params, make_fullpvec


import logging, psutil
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EDMRLSQ:
    """
    Fit EDMR spectra data via least-squares optimization.

    Args:
        * B (np.ndarray):
            * Magnetic field sweep data.
        * I (np.ndarray):
            * Experimental current data.
        * p0 (np.ndarray):
            * Initial parameter guess (19-vector).
        * lower (np.ndarray):
            * Lower bounds for parameters.
        * upper (np.ndarray):
            * Upper bounds for parameters.
        *edmr (callable):
            * Function edmr_spectra(B_array, pvec, modulate=True).
        * n_points_per (int):
            * The number of steps to simulate EDMR spectra per lsq call.
            * Defaults to 400 ~1 min/call if parallelized
        * n_jobs (int):
            * The number of threads to parallelize edmr_func across.
            * Defaults to cpu_count(logical=False) (# of physical cores)

    * result (OptimizeResult):
        *  The result of least_squares after fitting.
    """

    def __init__(self, B_array, I_array, p0, lower, upper, edmr_func, n_points_per=None, n_jobs=None):
        self._p0_full = np.asarray(p0, dtype=float)
        self._lower_full = np.asarray(lower, dtype=float)
        self._upper_full = np.asarray(upper, dtype=float)

        self._p0_nl = self._p0_full[2:]
        self._lower_nl = self._lower_full[2:]
        self._upper_nl = self._upper_full[2:]

        self._log_idx_nl = np.array([12, 13, 14, 15])

        self._p0_nl = self._p0_nl.astype(float)
        self._lower_nl = self._lower_nl.astype(float)
        self._upper_nl = self._upper_nl.astype(float)

        self._p0_nl[self._log_idx_nl] = np.log10(self._p0_nl[self._log_idx_nl])
        self._lower_nl[self._log_idx_nl] = np.log10(self._lower_nl[self._log_idx_nl])
        self._upper_nl[self._log_idx_nl] = np.log10(self._upper_nl[self._log_idx_nl])

        self.B = np.asarray(B_array, dtype=float)
        self.I = np.asarray(I_array, dtype=float)
        self.edmr = edmr_func

        self.n_points_per = n_points_per if n_points_per else 85
        self.n_jobs = n_jobs if n_jobs else psutil.cpu_count(logical=False)
        self.result = None

        self.param_names = ["A", "I0"] + P_ORDER + ["k_S", "k_D", "p", "B_mod"]

        self._nfev = 0
        self._calls_per_iter = 1 + self._p0_nl.size
        self._pkl_file = Path(__file__).resolve().parent / "utils/latest_param.pickle"

        assert len(self._p0_full) == 19, "expected 19 parameters."

    @staticmethod
    def _solve_linear_params(S, I):
        """
        Given model trace S and current I return (A, I0).
        Analytic solution.

        """
        X = np.column_stack((S, np.ones_like(S)))
        coeff, *_ = np.linalg.lstsq(X, I, rcond=None)
        return coeff

    def _build_full_pvec(self, p_nl, A, I0):
        full = np.empty_like(self._p0_full)
        full[0] = A
        full[1] = I0
        full[2:] = p_nl
        return full

    def _from_log(self, p):
        p = p.copy()
        p[self._log_idx_nl] = 10 ** p[self._log_idx_nl]
        return p

    def _residuals(self, p_nl, alpha=10):
        self._nfev += 1
        iter_no = (self._nfev - 1) // self._calls_per_iter + 1

        if len(self.B) > self.n_points_per:
            B_coarse = np.linspace(self.B[0], self.B[-1], self.n_points_per)
        else:
            B_coarse = self.B

        p_lin = self._from_log(p_nl)
        dummy_full = self._build_full_pvec(p_lin, 1.0, 0.0)
        S_coarse = self.edmr(B_coarse, dummy_full, modulate=True, n_jobs=self.n_jobs)

        spline = interp1d(B_coarse, S_coarse, kind="cubic", assume_sorted=True)
        S_pred = spline(self.B)

        A_opt, I0_opt = self._solve_linear_params(S_pred, self.I)
        I_pred = A_opt * S_pred + I0_opt

        w = np.abs(np.gradient(self.I, self.B))
        w = (w / w.max()) ** alpha

        if (self._nfev - 1) % self._calls_per_iter == 0:
            with open(self._pkl_file, "wb") as f:
                pickle.dump(p_lin, f)
                logger.info(" Saved pickle file. ")
            latest_full = self._build_full_pvec(p_lin, A_opt, I0_opt)
            print("")
            logger.info(" Iteration %d ", iter_no)
            for name, val in zip(self.param_names, latest_full):
                logging.info(f" * {name:<6} = {val:.3e}")
            print("")

        return w * (I_pred - self.I)

    def fit(self, ftol=1e-14, xtol=3e-16, gtol=1e-14, verbose=True):
    
        print("\nInitializing Least Squares Fitting Routine")
        print("==========================================\n")
        print(f"LSQ Arguments: ") 
        print(f"-------------- ")
        print(f"xtol      = {xtol:<6}")
        print(f"ftol      = {ftol:<6}")
        print(f"gtol      = {gtol:<6}")
        print(f"# points  = {self.n_points_per:<6}")
        print(f"# threads = {self.n_jobs:<6}\n")
        logger.info(
            " Rendering parallelized singlet spectrum - %d points in [%.1f, %.1f] G",
            min(len(self.B), self.n_points_per), min(self.B), max(self.B)
        )

        from utils.bounds import x_scale as _x_scale_full
        x_scale = _x_scale_full[2:].copy()
        x_scale[self._log_idx_nl] = 1.0

        self.result = least_squares(
            fun=self._residuals,
            x0=self._p0_nl,
            bounds=(self._lower_nl, self._upper_nl),
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            verbose=2 if verbose else 0,
            jac='2-point',
            method="trf",
            x_scale=x_scale # type: ignore
        )
        return self._from_log(self.result.x)

    @property
    def fitted_params(self):
        if self.result is None:
            raise RuntimeError("call fit() before accessing fitted_params.")
        return self._from_log(self.result.x)

    def predict(self, B_array=None, pvec=None):
        B_eval = np.asarray(B_array, dtype=float) if B_array is not None else self.B
        p_nl = np.asarray(pvec, dtype=float) if pvec is not None else self.fitted_params

        dummy_full = self._build_full_pvec(p_nl, 1.0, 0.0)
        S = self.edmr(B_eval, dummy_full, modulate=True, n_jobs=self.n_jobs)
        A_opt, I0_opt = self._solve_linear_params(
            S,
            self.I if B_array is None else np.interp(B_eval, self.B, self.I)
        )
        return A_opt * S + I0_opt

    def plot_best_fit(self, B_array=None, pvec=None, n_points: int = 500, save=True):
        import matplotlib.pyplot as plt
        from pathlib import Path

        B = B_array if B_array is not None else self.B

        if n_points is not None and n_points < len(B):
            B = np.linspace(min(B), max(B), n_points)
        if not pvec:
            with open(self._pkl_file, "rb") as f:
                pvec = pickle.load(f)

        logger.info(f" Rendering Fitting Plot with {n_points} steps.")
        pvec_nl = pvec if pvec is not None else self.fitted_params
        dI = self.predict(B_array=B, pvec=pvec_nl)

        spline = interp1d(B, dI, kind="cubic", assume_sorted=True)
        dI = spline(self.B)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.B, dI, c='r', ls=":", lw=1, label="fit")
        ax.plot(self.B, self.I, c='k', lw=2, label="raw")
        ax.axvline(0, ls=":", color="grey", alpha=0.6)
        ax.axhline(0, ls=":", color="grey", alpha=0.6)
        ax.set(
            xlabel="B [G]",
            ylabel=r"$\partial I/\partial B$ [nA]",
            title=f"EDMRLSQ Best-Fit",
        )
        ax.legend()

        outdir = Path(__file__).resolve().parent.parent / "media"
        if save:
            outdir.mkdir(parents=True, exist_ok=True)
            fname = outdir / "EDMRLSQ.png"
            fig.savefig(fname, dpi=300)
            logger.info(" fig::%s \n", fname)
        return fig

    def _print_pkl_params(self): 
        with open(self._pkl_file, "rb") as f: 
            params = pickle.load(f) 
        for name, val in zip(fitter.param_names[2:], params):   
            logger.info(f" {name:<6} = {val:.3e}") 
        print("")
        return params

# helper functions 
def _load_full_params(): 
    A, I0, hamiltonian_params, ks, kd, pgen, B_mod = load_params() 
    p0_full = make_fullpvec(
        k_s=ks,
        k_d=kd,
        p=pgen,
        params=hamiltonian_params,
        B_mod=B_mod,
        A=A,
        I0=I0,
    )
    return p0_full 

def _load_fitter(
    data_path: Path = Path.home() / "nasa/spectra/src/data/raw/[EDMR]_2G_3V_200MHz.pkl", 
    n_points_per: int = 100, 
    default_params: bool = True, 
    n_jobs=None,
    B_range=None, # (bmin, bmax) if not None, else entire data
    custom_params=None
):
    if not default_params and not custom_params: 
        logger.error(
            "custom_params must be provided \
             if default_params=False."
        ) 
        raise ValueError("custom_params not provided.")

    if default_params: 
        p0_full = _load_full_params() 
    else: 
        p0_full = custom_params

    with open(data_path, "rb") as f:
        data = pickle.load(f) 
    B = data["B (Gauss)"] 
    I = data["I (nA)"] 

    if B_range: 
        B = np.array(B) 
        I = np.array(I) 
    
        mask = (B >= B_range[0]) & (B <= B_range[1]) 
        B = B[mask] 
        I = I[mask]

    return EDMRLSQ( 
        B_array=B, 
        I_array=I, 
        p0=p0_full, 
        lower=lower, 
        upper=upper, 
        n_points_per=n_points_per, 
        edmr_func=edmr_spectra,
        n_jobs=n_jobs if n_jobs else None
    )

def _update_param_yaml(
    from_pkl: bool = True, 
    new_vals: np.ndarray = np.array([]),
    fitter: EDMRLSQ = _load_fitter(),
):
    """
    Updates the parameter file the fitter uses with custom parameters
    or the latest stored pickle file from a previous run. 

    Args: 
        * from_pkl: bool 
            * if True, just use update the param_file with the latest pickle. 
            * Defaults to True.
        * new_vals: np.ndarray
            * if not `from_pkl`, this is the 17-vector custom param. 
        * fitter: EDMRLSQ 
            * just an instance of the class to access what pickle file it uses. 
            * Defaults to generic instance from `_load_fitter()`.

    """


    from ruamel.yaml import YAML 
    PARAM_FILE = Path(__file__).resolve().parent / "utils/params.yaml"

    if not from_pkl: 
        assert len(new_vals) == 17, "new_vals length must be 17 if not from_pkl."
    else: 
        with open(fitter._pkl_file, "rb") as f: 
            new_vals = pickle.load(f)

    yaml = YAML() 
    yaml.indent(mapping=2, sequence=4, offset=2)
    with open(PARAM_FILE) as fp: 
        cfg = yaml.load(fp) 

    cfg['exchange']['J']               = float(new_vals[0])
    cfg['hyperfine']['Aa1']            = float(new_vals[1])
    cfg['hyperfine']['Ab1']            = float(new_vals[2])
    cfg['hyperfine']['Aa2']            = float(new_vals[3])
    cfg['hyperfine']['Ab2']            = float(new_vals[4])
    cfg['zfs']['D1']                   = float(new_vals[5])
    cfg['zfs']['D2']                   = float(new_vals[6])
    cfg['zeeman']['B0']                = float(new_vals[7])
    cfg['zeeman']['g_e']               = float(new_vals[8])
    cfg['zeeman']['g_n1']              = float(new_vals[9])
    cfg['zeeman']['g_n2']              = float(new_vals[10])
    cfg['microwave']['nu']             = float(new_vals[11])
    cfg['microwave']['omega1']         = float(new_vals[12])
    cfg['sle']['k_S']                  = float(new_vals[13])
    cfg['sle']['k_D']                  = float(new_vals[14])
    cfg['sle']['p']                    = float(new_vals[15])
    cfg['lockin']['B_mod']             = float(new_vals[16])

    with open(PARAM_FILE, 'w') as fp:
        yaml.dump(cfg, fp)
        logger.info(" params.yaml overwritten.")
        print("")
    

if __name__ == "__main__":

    _update_param_yaml()

    fitter = _load_fitter(
        data_path       = Path.home()/"nasa/spectra/src/data/raw/[EDMR]_2G_3V_200MHz.pkl", 
        n_points_per    = 50, 
        default_params  = True, 
        custom_params   = False, 
        n_jobs          = 5, 
        B_range         = (-50, 50)
    )

    #fitter.fit()
    fitter._print_pkl_params() 
    fitter.plot_best_fit()


