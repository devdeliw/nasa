from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Mapping

import numpy as np
import emcee, multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

import logging
logging.basicConfig(level=logging.INFO)

Array = np.ndarray


# -----------------------------------------------------------------------------
# Helper: default Jeffreys prior on noise
# -----------------------------------------------------------------------------

def _jeffreys_prior_sigma(sigma: float, low: float = 1e-4, high: float = 10.0) -> float:
    if sigma <= low or sigma >= high:
        return -np.inf
    return -np.log(sigma)


@dataclass
class EDMRDerivativeMCMC():
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


    def _model_derivative(self, theta: Sequence[float]) -> Array:
        """
        Compute model dI/dB at the stored `B_data` points.
        
        """
        A, I0, *phys, _ = theta
        phys_dict = dict(zip(self.phys_keys, phys))
        sing = self.singlet_fn(self.B_data, **phys_dict) # type: ignore
        dIdB = np.gradient(sing, self.dB or np.diff(self.B_data).mean())
        return A * dIdB + I0

    def log_prior(self, theta: Sequence[float]) -> float:
        A, I0, *rest = theta
        sigma = rest[-1]
        phys = rest[:-1]

        Imin, Imax = self.I_data.min(), self.I_data.max()

        # amplitude > 0 and not crazy large
        if not (0 < A < 5 * (Imax - Imin)):
            return -np.inf
        # offset near data mean
        if not (Imin - abs(Imin) < I0 < Imax + abs(Imax)):
            return -np.inf

        # physical parameter box priors (±20 % by default)
        p = dict(zip(self.phys_keys, phys))
        for k, v0 in self.base_phys.items():
            lo, hi = 0.8 * v0, 1.2 * v0
            if not (lo < p[k] < hi):
                return -np.inf

        # jeffreys on sigma
        lp = _jeffreys_prior_sigma(sigma)
        return lp

    def log_likelihood(self, theta: Sequence[float]) -> float:
        sigma = theta[-1]
        model = self._model_derivative(theta)
        resid = self.I_data - model
        return -0.5 * np.sum((resid / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))

    def log_posterior(self, theta: Sequence[float]) -> float:
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def _init_walkers(self, nwalkers: int | None = None) -> Array:
        ndim = 3 + len(self.phys_keys)  # A, I0, phys..., sigma
        if nwalkers is None:
            nwalkers = 4 * ndim
        theta0 = np.hstack([
            (self.I_data.max() - self.I_data.min()) / 2,  # A
            np.median(self.I_data),                      # I0
            [self.base_phys[k] for k in self.phys_keys], # phys params
            [self.sigma0],                               # sigma
        ])
        jitter = 1e-2
        return theta0 * (1 + jitter * np.random.randn(nwalkers, ndim))

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
        logging.info(f"Launching emcee: {nwalkers=}, {nsteps=} (burn {burn}), {threads} threads.")

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior, pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=progress)

        pool.close(); pool.join()

        self.sampler = sampler
        self.samples = sampler.get_chain(discard=burn, flat=True)
        self.log_prob = sampler.get_log_prob(discard=burn, flat=True)
        return self.samples # type: ignore

    def get_map(self) -> Array:
        if self._map is not None:
            return self._map
        if self.samples is None or self.log_prob is None:
            raise RuntimeError("run_mcmc first")
        self._map = self.samples[np.argmax(self.log_prob)]
        return self._map    # type: ignore

    def summary(self):
        if self.samples is None:
            raise RuntimeError("run_mcmc first")
        names = ["A", "I0"] + list(self.phys_keys) + ["sigma"]
        m, s = self.samples.mean(0), self.samples.std(0)
        for n, mi, si in zip(names, m, s):
            logging.info(f"{n:<6}: {mi:>10.4g} ± {si:>9.4g}")

    def plot_best_fit(self, bmin=-40, bmax=40, n_points=200, derivative=True, outdir=None, save=False):
        from run_solver import plot_sing_population
        from pathlib import Path
