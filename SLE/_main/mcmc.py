from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Sequence, Mapping

import numpy as np
import emcee, multiprocessing
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool

import logging
logging.basicConfig(level=logging.INFO)

def _jeffreys_prior_sigma(sigma: float, low: float = 1e-4, high: float = 10.0) -> float:
    if sigma <= low or sigma >= high:
        return -np.inf
    return -np.log(sigma)


@dataclass
class EDMR_MCMC():
    """
    MCMC fit of EDMR/NZFMR spectra. 

    Args: 
        * B_data, I_data: 1d array 
            * field axis [G] and lock-in amplifier signal [nA].
        * singlet_fn: callable 
            * singlet_fn(B, **phys_params) returns singlet populations 
            * at B points passed in. 
        * phys_keys: sequence[str]
            * names for each element in `phys_params`; used for prettyprint.
        * base_phys: mapping[str: float]
            * central values for priors. must cover `phys_keys`.
        * sigma0: float, optional 
            * initial noise guess for walker initialization.
        * dB: float, optional
            * field spacing used in `np.gradient` when `B_data` is nonuniform. 

    """

    B_data      : np.ndarray
    I_data      : np.ndarray
    singlet_fn  : Callable[[np.ndarray, Sequence[float]], np.ndarray]
    phys_keys   : Sequence[str]
    base_phys   : Mapping[str, float]
    sigma0      : float = 1e-3
    dB          : float | None = None

    # runtime containers (set in run_mcmc)
    sampler     : emcee.EnsembleSampler | None = None
    samples     : np.ndarray| None = None
    log_prob    : np.ndarray | None = None
    _map        : np.ndarray | None = None
    logger = logging.getLogger(__name__)

    def _model_derivative(self, theta: Sequence[float]) -> np.ndarray:
        """
        Compute model dI/dB at the stored `B_data` points.
        
        """
        logA, I0, *phys, _ = theta
        A = np.exp(logA)
        phys_dict = dict(zip(self.phys_keys, phys))
        sing = self.singlet_fn(self.B_data, **phys_dict) # type: ignore
        dIdB = np.gradient(sing, self.B_data)
        return A * dIdB + I0

    def log_prior(self, theta: Sequence[float]) -> float:
        logA, I0, *rest = theta
        log_sigma = rest[-1]
        sigma = np.exp(log_sigma)
        phys = rest[:-1]

        Imin, Imax = self.I_data.min(), self.I_data.max()

        lp_logA = -0.5 * (logA / 4.0) ** 2         
        if not (Imin - abs(Imin) < I0 < Imax + abs(Imax)):
            return -np.inf

        # physical parameter box priors (±20 % by default)
        p = dict(zip(self.phys_keys, phys))
        for k, v0 in self.base_phys.items():
            lo, hi = 0.8 * v0, 1.2 * v0
            if not (lo < p[k] < hi):
                return -np.inf

        # jeffreys on sigma
        lp_sigma = _jeffreys_prior_sigma(sigma)
        return lp_logA + lp_sigma

    def log_likelihood(self, theta: Sequence[float]) -> float:
        log_sigma = theta[-1]
        sigma = np.exp(log_sigma)
        model = self._model_derivative(theta)
        resid = self.I_data - model
        return -0.5 * np.sum((resid / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))

    def log_posterior(self, theta: Sequence[float]) -> float:
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def _init_walkers(self, nwalkers: int | None = None) -> np.ndarray:
        ndim = 3 + len(self.phys_keys)
        if nwalkers is None:
            nwalkers = 4 * ndim

        self.logger.info(" Precomputing baseline derivative spectra. ~10min.")
        dS0 = np.gradient(self.singlet_fn(self.B_data, **self.base_phys), self.B_data) # type: ignore
        init_logA  = np.log(self.I_data.ptp() / dS0.ptp())
        init_I0    = np.median(self.I_data)
        init_phys  = [self.base_phys[k] for k in self.phys_keys]
        init_logs  = np.log(self.sigma0)

        theta0 = np.array([
            init_logA,
            init_I0,
            *init_phys,
            init_logs
        ])  

        pos = np.tile(theta0, (nwalkers, 1))

        jitter_logA   = 0.2
        jitter_logsig = 0.2
        pos[:, 0]    += jitter_logA   * np.random.randn(nwalkers)
        pos[:, -1]   += jitter_logsig * np.random.randn(nwalkers)

        jitter_I0     = 0.1
        pos[:, 1]    += jitter_I0 * init_I0 * np.random.randn(nwalkers)

        jitter_phys   = 0.1
        for i in range(len(self.phys_keys)):
            p0 = init_phys[i]
            pos[:, 2 + i] = p0 * (1 + jitter_phys * np.random.randn(nwalkers))

        return pos

    def run_mcmc(
        self,
        nsteps: int,
        *,
        burn: int = 1000,
        nwalkers: int | None = None,
        threads: int | None = None,
        progress: bool = True,
    ) -> np.ndarray:
        pos = self._init_walkers(nwalkers)
        nwalkers, ndim = pos.shape

        threads = threads or multiprocessing.cpu_count()
        pool = ThreadPool(threads)
        print("")
        self.logger.info(f" Launching emcee:")
        self.logger.info(f"     # walkers {nwalkers:>5}")
        self.logger.info(f"     # steps   {nsteps:>5}") 
        self.logger.info(f"     # burn    {burn:>5}") 
        self.logger.info(f"     # threads {threads:>5}")

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior, pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=progress)

        pool.close(); pool.join()

        self.sampler = sampler
        self.samples = sampler.get_chain(discard=burn, flat=True)
        self.log_prob = sampler.get_log_prob(discard=burn, flat=True)
        return self.samples # type: ignore

    def get_map(self) -> np.ndarray:
        if self._map is not None:
            return self._map
        if self.samples is None or self.log_prob is None:
            raise RuntimeError("run_mcmc first")
        self._map = self.samples[np.argmax(self.log_prob)]
        return self._map    # type: ignore

    def summary(self):
        """
        Print the MAP (maximum a posteriori) parameter values,
        converting logA and log_sigma back into linear space.

        """
        theta_map = self.get_map()

        names = ["A", "I0"] + list(self.phys_keys) + ["sigma"]
        vals = theta_map.copy()

        vals[0]  = np.exp(vals[0])  # A in linear units
        vals[-1] = np.exp(vals[-1]) # sigma in linear units
        print("")
        self.logger.info(" Calculated best-fit parameters:")
        for name, val in zip(names, vals):
            self.logger.info(f" * {name:<6}: {val:>10.4g}")
        print("")

    def plot_best_fit(
        self,
        bmin    : float = -40,
        bmax    : float = +40,
        n_points: int = 200,
        *,
        derivative  : bool = True,
        outdir      : Path | None = Path.home() / "nasa/SLE/main/media",
        save        : bool = True,
        full        : bool = True, 
    ):
        """
        Plot the MAP-parameter model against the experimental spectrum.

        Args: 
            * bmin, bmax, n_points
                * range and density of the model curve.
            * derivative
                * ff True (default) plot dI/dB; if False plot the underlying current I.
                * Note: `self.I_data` is already a derivative spectrum. 
            * outdir
                * destination folder for pngs (`~/nasa/SLE/main/media` by default).
            * save
                * save pngs
            * full
                * overrides b sweep range, just matching the min max bounds of raw data. 

        Returns the figure(s) created. 

        """

        if self.B_data is None or self.I_data is None:
            raise RuntimeError("EDMR_MCMC: no data loaded.")

        theta_map = self.get_map()
        logA, I0, *phys_vals, _ = theta_map
        A = np.exp(logA)
        phys_dict = dict(zip(self.phys_keys, phys_vals))

        if full: 
            bmin = self.B_data.min() 
            bmax = self.B_data.max()
        B_dense = np.linspace(bmin, bmax, n_points)
        sing_dense = self.singlet_fn(B_dense, **phys_dict)  # type: ignore

        if derivative:
            model_dense = A * np.gradient(sing_dense, B_dense) + I0
            y_label = r"$\mathrm{d}I/\mathrm{d}B$  (arb.)"
            title   = "Best-fit EDMR derivative"
            fname   = "best_fit_derivative.png"
        else:
            model_dense = A * sing_dense + I0
            y_label = r"$I$  (arb.)"
            title   = "Best-fit current (integrated)"
            fname   = "best_fit_current.png"

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.scatter(
            self.B_data, self.I_data,
            s=5, c="k", marker="+", label="experimental"
        )

        sort_idx = np.argsort(B_dense)
        ax.plot(
            B_dense[sort_idx], model_dense[sort_idx],
            lw=2, c="r", label="best fit"
        )

        ax.axvline(0.0, ls=":", color="grey")
        ax.set_xlabel("B [G]", fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.set_title(title, fontsize=18)
        ax.legend()

        # residual 
        resid_dense = np.interp(self.B_data, B_dense, model_dense) - self.I_data
        fig2, ax2 = plt.subplots(figsize=(10, 2.5))
        ax2.scatter(self.B_data, resid_dense, s=8, marker="+")
        ax2.axhline(0.0, color="grey", lw=1)
        ax2.set_xlabel("B [G]", fontsize=12)
        ax2.set_ylabel("Residual", fontsize=12)
        ax2.set_title("Residual Plot", fontsize=14)

        if save:
            outdir = Path(outdir or Path.home() / "nasa/SLE/main/media")
            outdir.mkdir(parents=True, exist_ok=True)
            fig.savefig(outdir / fname, dpi=300, bbox_inches="tight")
            fig2.savefig(outdir / ("residual_" + fname), dpi=300, bbox_inches="tight")
            plt.close(fig); plt.close(fig2)

        return fig, fig2

