import numpy as np
import emcee, multiprocessing
from density_functional import make_density_solver, load_base_params
from multiprocessing.dummy import Pool as ThreadPool
from edmr_functional import make_edmr_model
import logging
logging.basicConfig(level=logging.INFO)

class EDMR_MCMC:
    """
    Runs parallelized affine-invariant MCMC to fit 
    `edmr_functional`'s EDMR model to real data from 4H-SiC. 

    """
    def __init__(
        self,
        B_data,     # (N,)
        I_data,     # (N,)
        sigma=1e-3  # initial noise guess
    ):
        """
        Args:
            * B_data: (array-like)
                * 1D array of B-field EDMR data.
            * I_data: (array_like)
                * 1D array of I current EDMR data.
            * sigma: float
                * initial guess for EDMR noise
        """

        self.B_data = B_data
        self.I_data = I_data
        self.sigma0 = sigma                 
        self.logger = logging.getLogger(__name__)

        rho_fn, P_S      = make_density_solver()
        self.base_params = load_base_params()
        self.phys_keys   = [k for k in self.base_params if k != 'B0']
        self.I_model     = make_edmr_model(rho_fn, P_S, self.phys_keys)

        self.logger.info(" Model defined. Initializing MCMC.")

        self.sampler = None
        self.samples = None
        self.log_prob= None            
        self._map    = None 

    def log_prior(self, theta):
        A, I0, *rest  = theta
        sigma, phys   = rest[-1], rest[:-1]
        Imin, Imax    = self.I_data.min(), self.I_data.max()

        # amplitude
        if not (0 < A < 2*(Imax - Imin)):                       return -np.inf

        # offset
        if not (Imin - 0.1*Imin < I0 < Imax + 0.1*Imax):        return -np.inf

        # physical params
        p = dict(zip(self.phys_keys, phys))

        # immutable
        for key in ('hbar', 'mu_B', 'mu_N', 'g_n1', 'g_n2'):
            if not np.isclose(p[key], self.base_params[key]):   return -np.inf

        # exchange
        if not (1e-9 < p['J'] < 1e-6):                          return -np.inf

        # hyperfine
        for key in ('Aa1', 'Aa2', 'Ab1', 'Ab2'):
            if not (1e-8 < p[key] < 1e-6):                      return -np.inf

        # g-factors
        if not (1.9  < p['g_e']  < 2.1):                        return -np.inf

        # ZFS prior (20%)
        for key, frac in (('D1', 0.2), ('D2', 0.2)):
            mu = self.base_params[key]
            if abs(p[key] - mu) > 3 * frac * mu:                return -np.inf

        # dissociation rates
        for key in ('k_S', 'k_D'):
            if not (1e3 < p[key] < 1e8):                        return -np.inf

        # generation rate
        if not (1e2 < p['p'] < 1e5):                            return -np.inf

        # sigma jeffreys prior
        if not (1e-6 < sigma < 1e-1):                           return -np.inf

        # Gaussian ZFS contribution + Jeffreys term
        lp = -np.log(sigma)
        for key, frac in (('D1', 0.2), ('D2', 0.2)):
            mu = self.base_params[key]
            sd = frac * mu
            lp += -0.5 * ((p[key] - mu) / sd)**2

        return lp

    def log_likelihood(self, theta):
        sigma = theta[-1]
        try:
            _ = self.I_model(self.B_data, *theta[:-1])
        except (ValueError, np.linalg.LinAlgError) as e:
            return -np.inf
        resid = self.I_data - self.I_model(self.B_data, *theta[:-1])
        return -0.5 * np.sum((
            resid / sigma)**2 + np.log(2*np.pi*sigma**2
        ))

    def log_posterior(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):                                 return -np.inf
        return lp + self.log_likelihood(theta)

    def _init_walkers(self, nwalkers=None):
        ndim = 3 + len(self.phys_keys)          
        if nwalkers is None: nwalkers = 4 * ndim  # 4 walkers per parameter? 
        self.logger.info(f"     * Initializing {nwalkers} walkers.")

        Imin, Imax = self.I_data.min(), self.I_data.max()
        theta0 = np.hstack((
            [Imax - Imin, np.median(self.I_data)],
            [self.base_params[k] for k in self.phys_keys],
            [self.sigma0]
        ))

        # relative jitter
        eps = 1e-2
        return theta0 * (1 + eps * np.random.randn(nwalkers, ndim))

    def run_mcmc(
            self, 
            nsteps, 
            burn=1000, 
            nwalkers=None, 
            threads=None, 
            progress=True
    ):
        pos    = self._init_walkers(nwalkers)
        nwalkers, ndim = pos.shape

        threads = threads or multiprocessing.cpu_count()
        pool    = ThreadPool(threads)
        self.logger.info(f"     * Parallelizing across {threads} cores.")

        sampler = emcee.EnsembleSampler(
            nwalkers, 
            ndim, 
            self.log_posterior, 
            pool=pool
        )

        self.logger.info(f"     * Running the MCMC.")
        self.logger.info(f"         * {nsteps} steps.")
        self.logger.info(f"         * {nwalkers} walkers.")
        self.logger.info(f"         * {burn} burnin.")
        sampler.run_mcmc(pos, nsteps, progress=progress)

        pool.close(); pool.join()

        self.sampler = sampler
        self.samples = sampler.get_chain(discard=burn, flat=True)
        self.log_prob = sampler.get_log_prob(discard=burn, flat=True)
        return self.samples

    def get_map(self):
        if self._map is not None:
            return self._map
        if self.samples is None or self.log_prob is None:
            raise RuntimeError("run_mcmc first")
        self._map = self.samples[np.argmax(self.log_prob)]
        return self._map

    def summary(self):
        """
        Displays posterior mean + std per parameter. 
        """
        if self.samples is None:               
            raise RuntimeError('no samples')
        names = ['A', 'I0'] + self.phys_keys + ['sigma']
        m, s   = self.samples.mean(0), self.samples.std(0)

        for n, mi, si in zip(names, m, s):
            self.logger.info(f" {n:<5}: {mi:>10.3e} ± {si:>9.3e}")

    def plot(self):  
        """
        Plots best-fit against raw input spectra. 

        """
        import matplotlib.pyplot as plt

        best = self.get_map()
        I_pred = self.I_model(self.B_data, *best[:-1])

        fig, axis = plt.subplots(1, 1, figsize = (10, 5))
        plt.scatter(self.B_data, self.I_data, marker="+", label='spectra', s=5)
        plt.plot(self.B_data, I_pred, label='best fit', linewidth=2)
        axis.set_xlabel('B (Gauss)', fontsize=15)
        axis.set_ylabel('I (nA)', fontsize=15)
        plt.legend()
        return plt



    


