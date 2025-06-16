import numpy as np 
import emcee
import os
import matplotlib.pyplot as plt

# author: Deval Deliwala 
# run from ~/nasa/ 

# Helper Functions # 

def S_model(B, S0, A1, A2, A3, G1, G2, G3):
    """
    Three independent first derivatives of centred Lorentzians plus baseline.

    S(B) = S0 - 2 B Sum(A_k / (B^2 + Gamma_k^2))^2
    Gamma_k > 0  (widths);  

    A_k are signed amplitudes already containing Gamma_k^2 factor.

    Args: 
        * B  : array_like
        * S0 : float
        * A1,A2,A3 : float
        * G1,G2,G3 : positive float (Gamma_k in Gauss)

    Returns: 
        * S_model : ndarray  (same shape as B)

    """

    Gs  = np.array([G1, G2, G3])[:, None]   
    As  = np.array([A1, A2, A3])[:, None]
    term = -2.0 * B * (As / (B**2 + Gs**2)**2).sum(axis=0)
    return S0 + term

def log_prior(theta, s_min, s_max):
    """
    theta = (S0, A1, A2, A3, logG1, logG2, logG3, log_sigma)

    logG*  = ln Gamma_k  to sample widths on log scale -> Jeffreys / log-uniform[0.05,100] G
    log_sigma = ln sigma, sigma > 0  with half-Cauchy(scale=scale_sigma)

    Returns −np.inf outside support.

    """

    S0, A1, A2, A3, lg1, lg2, lg3, lgsig = theta

    # uniform baseline between data extrema 
    if not (s_min <= S0 <= s_max):
        return -np.inf

    # normal(0, 200 nA) on amplitudes (nA absorbed in units) 
    amp_sd = 200.0 / np.sqrt(3) 
    logp_A = -0.5*((A1/amp_sd)**2 + (A2/amp_sd)**2 + (A3/amp_sd)**2) \
             -3.0*np.log(amp_sd) - 0.5*3*np.log(2*np.pi)

    # log-uniform Gamma_k in [0.05,100] G
    for lg in (lg1, lg2, lg3):
        if not (np.log(0.05) <= lg <= np.log(100.0)):        
            return -np.inf
    logp_G = - (lg1 + lg2 + lg3) # jacobian                    

    # half-Cauchy(0, beta) in log space
    beta = 0.3 * (s_max - s_min) # heuristic
    sig  = np.exp(lgsig)
    if sig <= 0.0:
        return -np.inf

    # jacobian
    logp_sigma = np.log(2/np.pi) + np.log(beta) - np.log(beta**2 + sig**2) + lgsig  

    return logp_A + logp_G + logp_sigma

def log_likelihood(theta, B, S_obs):
    S0, A1, A2, A3, lg1, lg2, lg3, lgsig = theta
    G1, G2, G3 = np.exp([lg1, lg2, lg3])
    sigma      = np.exp(lgsig)

    S_pred = S_model(B, S0, A1, A2, A3, G1, G2, G3)
    resid  = S_obs - S_pred
    N      = S_obs.size

    return -0.5*(np.sum(resid**2) / sigma**2 + N*np.log(2*np.pi*sigma**2))

def make_log_prob(B, S_obs):
    s_min, s_max = S_obs.min(), S_obs.max()

    def _log_prob(theta):
        lp = log_prior(theta, s_min, s_max)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, B, S_obs)

    return _log_prob


# driver 
def build_sampler(B, S_obs, n_walkers=32):
    """
    Returns an emcee.EnsembleSampler ready for sampling.

    Starting positions: tiny Gaussian ball around ordinary-least-squares estimates.

    """
    log_prob = make_log_prob(B, S_obs)

    # crude initial fit for sensible starting point
    S0_init  = S_obs.mean()
    A_init   = (S_obs.max() - S_obs.min()) * 0.1
    G_init   = 10.0
    sigma_init = 0.1 * (S_obs.max() - S_obs.min())

    p0_core = np.array([S0_init,  A_init, 0.8*A_init, -0.5*A_init,
                        np.log(G_init), np.log(1.2*G_init), np.log(0.7*G_init),
                        np.log(sigma_init)])

    # walkers in small Gaussian ball
    ndim = len(p0_core)
    p0   = p0_core + 1e-4 * np.random.randn(n_walkers, ndim)

    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)
    return sampler, p0


class EDMRMCMC:
    """
    Minimal wrapper that fits the three-derivative-Lorentzian model with emcee.
    
    Args: 
        * B : 1-D ndarray
            Magnetic-field axis in Gauss.
        * I : 1-D ndarray
            Measured current (same length as B).
        * n_steps : int
            Total MCMC steps per walker (burn-in will be n_steps//4).
        * n_walkers : int, optional
            Size of the ensemble.  Default 64.

    """

    param_names = ["S0", "A1", "A2", "A3",
                   "G1", "G2", "G3", "sigma"]

    def __init__(self, B, I, n_steps, n_walkers=64):
        self.B = np.asarray(B)
        self.I = np.asarray(I)
        self.n_steps = int(n_steps)
        self.n_walkers = int(n_walkers)

        self.sampler, p0 = build_sampler(self.B, self.I, n_walkers=self.n_walkers)
        self._run(p0)
        self._summarize()
        self._save()

    def _run(self, p0):
        burn = self.n_steps // 4
        self.sampler.run_mcmc(p0, burn, progress=False)
        self.sampler.run_mcmc(None, self.n_steps - burn, progress=True)

    def _summarize(self):
        flat = self.sampler.get_chain(discard=0, flat=True)
        med = np.median(flat, axis=0)

        # convert log-space params to linear
        med[4:7] = np.exp(med[4:7])          # G_k
        med[7]   = np.exp(med[7])            # sigma

        lo, hi = np.percentile(flat, [16, 84], axis=0)
        lo[4:7], hi[4:7] = np.exp(lo[4:7]), np.exp(hi[4:7])
        lo[7], hi[7]     = np.exp(lo[7]),    np.exp(hi[7])

        self.results = np.vstack((med, lo, hi)).T  # shape (8,3)

    def _save(self, fname="./mcmc_params.txt"):
        header = "param  median  16th  84th"
        lines = [header]
        for name, (m, l, h) in zip(self.param_names, self.results):
            lines.append(f"{name:<5} {m:.6g} {l:.6g} {h:.6g}")
        with open(fname, "w") as f:
            f.write("\n".join(lines))
        print(f"Saved parameter summary to {os.path.abspath(fname)}")

    def plot_fit(self, fname="fit_vs_data.png", dpi=300):
        p = self.results[:, 0]                    # median parameters
        S0, A1, A2, A3, G1, G2, G3, _ = p

        def dlorentz(B, A, G):
            return -2.0 * B * A / (B**2 + G**2)**2

        B  = self.B
        I  = self.I
        L1 = S0 + dlorentz(B, A1, G1)
        L2 = S0 + dlorentz(B, A2, G2)
        L3 = S0 + dlorentz(B, A3, G3)
        Ltot = S0 + dlorentz(B, A1, G1) + dlorentz(B, A2, G2) + dlorentz(B, A3, G3)

        plt.figure(figsize=(6, 4))
        plt.plot(B, I, ".", ms=3, label="data")
        plt.plot(B, L1, lw=1.2, label="Lorentz 1")
        plt.plot(B, L2, lw=1.2, label="Lorentz 2")
        plt.plot(B, L3, lw=1.2, label="Lorentz 3")
        plt.plot(B, Ltot, lw=2.0, color="black", label="compound")
        plt.xlabel("B  (G)")
        plt.ylabel("Current  (arb. units)")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(f"./{fname}", dpi=dpi)
        plt.close()
        print(f"Saved plot to {os.path.abspath(fname)}")

if __name__ == "__main__":
    import pickle
    file = "./data/raw/[EDMR]_2G_3V_200MHz.pkl" 

    try: 
        with open(file, "rb") as f: 
            spectra = pickle.load(f)

        spectra = spectra[spectra["B (Gauss)"].abs() <= 50]
        B = np.array(spectra["B (Gauss)"])
        I = np.array(spectra["I (nA)"])



        fit = EDMRMCMC(B, I, n_steps=10000, n_walkers=64)
        fit.plot_fit("fit.png")
        print(fit.results)
    except FileNotFoundError: 
        print("file not found")
