import numpy as np, pickle
from scipy.optimize import least_squares 
from scipy.interpolate import interp1d
from run_solver import P_ORDER 
from pathlib import Path

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
        self.B      = np.asarray(B_array, dtype=float)
        self.I      = np.asarray(I_array, dtype=float)
        self.p0     = np.asarray(p0, dtype=float)
        self.lower  = np.asarray(lower, dtype=float)
        self.upper  = np.asarray(upper, dtype=float)
        self.edmr   = edmr_func
        self.n_points_per = n_points_per if n_points_per else 85
        self.n_jobs = n_jobs if n_jobs else psutil.cpu_count(logical=False)
        self.result = None

        self.param_names = ["A", "I0"] + P_ORDER + ["k_S", "k_D", "p", "B_mod"]

        # tracking calls 
        self._nfev = 0
        self._calls_per_iter = 1 + self.p0.size  

        assert len(lower)==len(p0)==len(upper), "bounds must match p0 length"

    def _residuals(self, pvec):

        self._nfev += 1
        iter_no = (self._nfev - 1) // self._calls_per_iter + 1

        if (self._nfev - 1) % self._calls_per_iter == 0:
            with open(Path(__file__).resolve().parent / "utils/latest_param.pickle", "wb") as f: 
                pickle.dump(pvec, f)

            logger.info(" Iteration %d ", iter_no)
            for name, val in zip(self.param_names, pvec):
                logging.info(f" * {name:<6} = {val:.3e}")
            print("")

        if len(self.B) > self.n_points_per:
            B_coarse = np.linspace(self.B[0], self.B[-1], self.n_points_per)
        else:
            B_coarse = self.B

        I_coarse = self.edmr(
            B_coarse, pvec, modulate=True,
            n_jobs=self.n_jobs
        )

        spline = interp1d(B_coarse, I_coarse, kind="cubic", assume_sorted=True)
        I_pred = spline(self.B)

        # weights 
        w = np.abs( np.gradient(self.I, self.B) )
        w = w / (w.max() or 1.0)

        return w * (I_pred - self.I)

    def fit(self, ftol=1e-9, xtol=1e-12, verbose=False):
        """
        Run the least-squares fit.

        Args:
            * ftol (float):
                * Tolerance for change in cost function.
            * xtol (float): 
                * Tolerance for change in parameters.
            * verbose (bool): 
                * If True, prints solver progress.

        Returns:
            * np.ndarray: Best-fit parameter vector.
        """
        from utils.bounds import x_scale

        logger.info(
            " Rendering parallelized singlet spectrum - %d points in [%.1f, %.1f] G",
            min(len(self.B), self.n_points_per), min(self.B), max(self.B)
        )
        self.result = least_squares(
            fun=self._residuals,
            x0=self.p0,
            bounds=(self.lower, self.upper),
            ftol=ftol,
            xtol=xtol,
            verbose=2 if verbose else 0,
            jac='2-point', 
            method="dogbox", 
            x_scale=x_scale # type: ignore
        )
        return self.result.x

    @property
    def fitted_params(self):
        """
        Returns the fitted parameters after running fit().

        """
        if self.result is None:
            raise RuntimeError("Call fit() before accessing fitted_params.")
        return self.result.x

    def predict(self, B_array=None, pvec=None):
        """
        Generate model predictions for a given B_array and parameter vector.

        Args:
            * B_array (array-like, optional): 
                * Field values to predict at.
                * Defaults to the original B-array.
            * pvec (array-like, optional): 
                * Parameter vector to use.
                * Defaults to the fitted parameters.

        Returns:
            * np.ndarray: Predicted currents.
        """
        B_eval = np.asarray(B_array, dtype=float) if B_array is not None else self.B
        p_eval = np.asarray(pvec, dtype=float) if pvec is not None else self.fitted_params
        return self.edmr(B_eval, p_eval, modulate=True, n_jobs=self.n_jobs)
    
    def plot_best_fit(self, B_array=None, pvec=None, save=True): 
        import matplotlib.pyplot as plt
        from pathlib import Path

        B = B_array if B_array is not None else self.B
        pvec = pvec if pvec is not None else self.fitted_params
        dI= self.predict(B_array=B, pvec=pvec)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(B, dI, c='r', ls=":", lw=1, label="fit")    # fit line
        ax.plot(self.B, self.I, c='k', lw=2, label="raw")   # raw line
        ax.axvline(0, ls=":", color="grey", alpha = 0.6)
        ax.axhline(0, ls=":", color="grey", alpha = 0.6)
        ax.set(
            xlabel="B [G]",
            ylabel=r"$\mathrm{d}I/\mathrm{d}B$ [nA]",
            title=f"EDMRLSQ Best-Fit",
        )
        ax.legend()

        outdir = Path(__file__).resolve().parent.parent / "media"
        if save:
            outdir.mkdir(parents=True, exist_ok=True)
            fname = outdir / f"EDMRLSQ.png"
            fig.savefig(fname, dpi=300)
            logger.info(" fig::%s \n", fname)
        return fig


if __name__ == "__main__":
    from utils.bounds import lower, upper
    from sle_model import edmr_spectra 
    from run_solver import load_params, make_fullpvec

    raw_edmr_file = Path.home() / "nasa/spectra/src/data/raw/[EDMR]_2G_3V_200MHz.pkl"
    with open(raw_edmr_file, "rb") as f: 
        raw_edmr = pickle.load(f)

    B_data = raw_edmr["B (Gauss)"]
    I_data = raw_edmr["I (nA)"]

    A, I0, hamiltonian_params, ks, kd, pgen, B_mod = load_params()
    
    p0 = make_fullpvec(
        k_s = ks,
        k_d = kd, 
        p   = pgen, 
        params = hamiltonian_params, 
        B_mod = B_mod, 
        A = A, 
        I0 = I0
    )

    fitter = EDMRLSQ(
        B_array=B_data, 
        I_array=I_data, 
        p0=p0, 
        lower=lower, 
        upper=upper, 
        n_points_per=100,
        edmr_func=edmr_spectra, 
    )
    best_params = fitter.fit(verbose=True)

    # pretty-print best-fit params
    print("")
    logger.info(f" Best-Fit Parameters")
    for name, val in zip(fitter.param_names, best_params):
        logger.info(f" {name:<6} = {val:.3e}")
    print("")

    fitter.plot_best_fit()

