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

    """

    def __init__(self, B_array, I_array, p0, lower, upper, edmr_func, n_points_per=None, n_jobs=None, show_progress=True, verbose=True):
        self._p0_full   = np.asarray(p0,    dtype=float)
        self._lower_full = np.asarray(lower, dtype=float)
        self._upper_full = np.asarray(upper, dtype=float)

        self._p0_nl    = self._p0_full.copy()
        self._lower_nl = self._lower_full.copy()
        self._upper_nl = self._upper_full.copy()
        self.B = np.asarray(B_array, dtype=float)
        self.I = np.asarray(I_array, dtype=float)
        self.edmr = edmr_func
        self._show_progress = show_progress
        self._verbose = verbose

        self.n_points_per = n_points_per if n_points_per else 85
        self.n_jobs = n_jobs if n_jobs else psutil.cpu_count(logical=False)
        self.result = None

        self.param_names = ["A", "I0"] + P_ORDER + ["k_S", "k_D", "p", "B_mod"]

        self._nfev = 0
        self._calls_per_iter = 1 + self._p0_nl.size
        self._pkl_file = Path(__file__).resolve().parent / "utils/latest_param.pickle"

        assert len(self._p0_full) == 19, "expected 19 parameters."

    def _from_log(self, p):
        raise NotImplementedError("_from_log is disabled")

    def _residuals(self, p_nl, alpha=1, save_pickle=True):
        self._nfev += 1
        iter_no = (self._nfev - 1) // self._calls_per_iter + 1

        if len(self.B) > self.n_points_per:
            B_coarse = np.linspace(self.B[0], self.B[-1], self.n_points_per)
        else:
            B_coarse = self.B

        p_lin = p_nl
        S_coarse = self.edmr(B_coarse, p_lin, modulate=True, n_jobs=self.n_jobs, show_progress=self._show_progress)

        spline = interp1d(B_coarse, S_coarse, kind="cubic", assume_sorted=True)
        S_pred = spline(self.B)

        A_opt, I0_opt = p_lin[0], p_lin[1]
        I_pred = A_opt * S_pred + I0_opt

        w = np.abs(np.gradient(self.I, self.B))
        w = (w / w.max()) ** alpha

        if (self._nfev - 1) % self._calls_per_iter == 0:
            if save_pickle: 
                with open(self._pkl_file, "wb") as f:
                    pickle.dump(p_lin, f)
                    logger.info(" Saved pickle file. ")
            if self._verbose:
                print("")
                logger.info(" Iteration %d ", iter_no)
                for name, val in zip(self.param_names, p_lin):
                    logging.info(f" * {name:<6} = {val:.3e}")
                print("")

        return w * (I_pred - self.I)

    def fit(self, ftol=1e-14, xtol=3e-16, gtol=1e-14):

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
        scales_nl = _x_scale_full.copy() 

        x0_nl_norm    = self._p0_nl    / scales_nl
        lower_nl_norm = self._lower_nl / scales_nl
        upper_nl_norm = self._upper_nl / scales_nl

        def residuals_norm(x_norm): 
            p_nl_real = x_norm * scales_nl 
            return self._residuals(p_nl_real)

        self.result = least_squares(
            fun=residuals_norm,
            x0=x0_nl_norm,
            bounds=(lower_nl_norm, upper_nl_norm),
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            verbose=2 if self._verbose else 0,
            jac='2-point',
            method="trf",
        )

        x_nl_opt_norm = self.result.x 
        p_nl_opt_real = x_nl_opt_norm * scales_nl 
        p_nl_opt      = self._from_log(p_nl_opt_real)
        self.result.x = p_nl_opt_real 

        return p_nl_opt

    @property
    def fitted_params(self):
        if self.result is None:
            raise RuntimeError("call fit() before accessing fitted_params.")
        return self.result.x

    def predict(self, B_array=None, pvec=None):
        B_eval = np.asarray(B_array, dtype=float) if B_array is not None else self.B
        p_lin = self.fitted_params if pvec is None else np.asarray(pvec, float)
        S = self.edmr(B_eval, p_lin, modulate=True, n_jobs=self.n_jobs)
        A_opt, I0_opt = p_lin[0], p_lin[1]
        return A_opt * S + I0_opt

    def plot_best_fit(self, B_array=None, pvec=None, n_points: int = 500, save=True):
        import matplotlib.pyplot as plt
        from pathlib import Path

        B = B_array if B_array is not None else self.B

        if n_points is not None and n_points < len(B):
            B = np.linspace(min(B), max(B), n_points)

        p_lin = self._p0_full if pvec is None else np.asarray(pvec, float)
        dI = self.predict(B_array=B, pvec=p_lin)
        logger.info(f" Rendering Fitting Plot with {n_points} steps.")

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
        params = self._p0_nl
        for name, val in zip(self.param_names, params):   
            logger.info(f" {name:<6} = {val:.3e}") 
        print("")
        return params

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
    B_range=None, 
    custom_params=np.array([]), 
    show_progress=True,
    verbose=True,
):
    if not default_params and len(custom_params) == 0: 
        logger.error(
            "custom_params must be provided              if default_params=False."
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
        n_jobs=n_jobs if n_jobs else None,
        show_progress=show_progress,
        verbose=verbose,
    )

def _update_param_yaml(
    from_pkl: bool = True, 
    new_vals: np.ndarray = np.array([]),
    fitter: EDMRLSQ = _load_fitter(),
):
    from ruamel.yaml import YAML 
    PARAM_FILE = Path(__file__).resolve().parent / "utils/params.yaml"

    if not from_pkl: 
        assert len(new_vals) == 19, "new_vals length must be 19 if not from_pkl."
    else: 
        with open(fitter._pkl_file, "rb") as f: 
            new_vals = pickle.load(f)

    yaml = YAML() 
    yaml.indent(mapping=2, sequence=4, offset=2)
    with open(PARAM_FILE) as fp: 
        cfg = yaml.load(fp) 

    cfg['exchange']['J']               = float(new_vals[2])
    cfg['hyperfine']['Aa1']            = float(new_vals[3])
    cfg['hyperfine']['Ab1']            = float(new_vals[4])
    cfg['hyperfine']['Aa2']            = float(new_vals[5])
    cfg['hyperfine']['Ab2']            = float(new_vals[6])
    cfg['zfs']['D1']                   = float(new_vals[7])
    cfg['zfs']['D2']                   = float(new_vals[8])
    cfg['zeeman']['B0']                = float(new_vals[9])
    cfg['zeeman']['g_e']               = float(new_vals[10])
    cfg['zeeman']['g_n1']              = float(new_vals[11])
    cfg['zeeman']['g_n2']              = float(new_vals[12])
    cfg['microwave']['nu']             = float(new_vals[13])
    cfg['microwave']['omega1']         = float(new_vals[14])
    cfg['sle']['k_S']                  = float(new_vals[15])
    cfg['sle']['k_D']                  = float(new_vals[16])
    cfg['sle']['p']                    = float(new_vals[17])
    cfg['lockin']['B_mod']             = float(new_vals[18])

    with open(PARAM_FILE, 'w') as fp:
        yaml.dump(cfg, fp)
        logger.info(" params.yaml overwritten.")
        print("")
    
if __name__ == "__main__":

    #_update_param_yaml()

    A, I, params, ks, kd, p, B_mod = load_params()
    fitter = _load_fitter(
        data_path       = Path.home()/"nasa/spectra/src/data/raw/[EDMR]_2G_3V_200MHz.pkl", 
        n_points_per    = 100, 
        default_params  = False, 
        custom_params   = make_fullpvec(params, ks, kd, p, B_mod, A, I), 
        n_jobs          = 2, 
        B_range         = (-85, 85), 
    )
    #fitter.fit()
    fitter._print_pkl_params() 
    fitter.plot_best_fit(n_points=100)

