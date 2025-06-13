#!/usr/bin/env python3

# author: Deval Deliwala

import os 
import numpy as np
import matplotlib.pyplot as plt
import emcee, corner, pickle, pathlib

# compound lorentz derivative model. 
def triple_dlorentz(B, S0, A1, G1, A2, G2, A3, G3):
    """Sum of three first-derivative Lorentzians centred at B = 0."""
    term1 = -2.0 * B * A1 / (B**2 + G1**2) ** 2
    term2 = -2.0 * B * A2 / (B**2 + G2**2) ** 2
    term3 = -2.0 * B * A3 / (B**2 + G3**2) ** 2
    return S0 + term1 + term2 + term3

def log_prior(theta, Ispan, dB):
    S0, A1, A2, A3, lnG1, lnG2, lnG3, lnsig = theta

    if not (Imin <= S0 <= Imax):
        return -np.inf

    Amax = 1e6 * Ispan
    if not (-0.6 * Amax <= A1 <= 0.6 * Amax):
        return -np.inf
    if not (-Amax <= A2 <= Amax):
        return -np.inf

    # S3 antisymmetric ? 
    if np.sign(A3) == np.sign(A2): 
        return -np.inf 
    if not (50.0 <= abs(A3) <= 100.0): 
        return -np.inf 
    if not (-0.20 * Amax <= A3 <= 0.20 * Amax):
        return -np.inf

    if not (np.log(0.30 * dB) <= lnG1 <= np.log(2.0 * dB)):
        return -np.inf
    if not (np.log(0.05 * dB) <= lnG2 <= np.log(0.30 * dB)):
        return -np.inf
    if not (np.log(0.9) <= lnG3 <= np.log(1.5)):
        return -np.inf

    # noise
    if not (-20 <= lnsig <= np.log(Ispan / 2.0)):
        return -np.inf

    return 0.0  

def log_likelihood(theta, B, I):
    S0, A1, A2, A3, lnG1, lnG2, lnG3, lnsig = theta
    model = triple_dlorentz(
        B,
        S0,
        A1, np.exp(lnG1),
        A2, np.exp(lnG2),
        A3, np.exp(lnG3),
    )
    sigma = np.exp(lnsig)
    return -0.5 * np.sum(((I - model) / sigma) ** 2 + np.log(2.0 * np.pi * sigma ** 2))

def log_posterior(theta, B, I, Ispan, dB):
    lp = log_prior(theta, Ispan, dB)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, B, I)

def run_mcmc(B, I,
             nwalkers=64,
             nsteps=15000,
             burn=3000,
             outdir="media/mcmc/"):

    global Imin, Imax
    Imin, Imax = I.min(), I.max()
    Ispan = Imax - Imin
    dB = B.ptp()

    rng = np.random.default_rng()
    p0 = []
    for _ in range(nwalkers):
        # baseline
        S0 = rng.uniform(Imin, Imax)

        mag1 = 10 ** rng.uniform(-1, 5) * Ispan      
        mag2 = 10 ** rng.uniform(-1, 5) * Ispan
        mag3 = 10 ** rng.uniform(-2, 4) * Ispan      

        sign = rng.choice([-1, 1])
        A1 = sign * mag1
        A2 = sign * mag2            # A1 and A2 same sign
        A3 = -sign * mag3           # A3 opposite sign

        lnG1 = rng.uniform(np.log(0.5 * dB), np.log(1.5 * dB))
        lnG2 = rng.uniform(np.log(0.08 * dB), np.log(0.25 * dB))
        lnG3 = rng.uniform(np.log(0.005 * dB), np.log(0.03 * dB))

        # noise
        lnsig = rng.uniform(np.log(0.02 * Ispan), np.log(0.5 * Ispan))

        p0.append([S0, A1, A2, A3, lnG1, lnG2, lnG3, lnsig])

    sampler = emcee.EnsembleSampler(
        nwalkers, 8,
        log_posterior, args=(B, I, Ispan, dB)
    )
    sampler.run_mcmc(p0, nsteps, progress=True)

    flat   = sampler.get_chain(discard=burn, flat=True)
    lnprob = sampler.get_log_prob(discard=burn, flat=True)
    theta_best = flat[np.argmax(lnprob)]

    if not os.path.exists(outdir): 
        os.makedirs(outdir, exist_ok=True)
    path = pathlib.Path(outdir)
    np.save(path / "chain.npy",   sampler.get_chain())
    np.save(path / "lnprob.npy",  sampler.get_log_prob())

    labels = ["S0", "A1", "A2", "A3", "lnG1", "lnG2", "lnG3", "lnσ"]
    flat_scaled = (flat - flat.mean(axis=0)) / flat.std(axis=0)
    corner.corner(
        flat_scaled, 
        labels=labels, 
        truths=(theta_best-flat.mean(axis=0)) / flat.std(axis=0), 
        label_kwargs={"fontsize": 16},   
        title_kwargs={"fontsize": 16},
        smooth=1.0
    )
    plt.savefig(path / "corner.png", dpi=300)
    plt.close()

    B_plot = np.linspace(B.min(), B.max(), 1500)
    S0, A1, A2, A3, lnG1, lnG2, lnG3, _ = theta_best
    G1, G2, G3 = np.exp(lnG1), np.exp(lnG2), np.exp(lnG3)

    fit = triple_dlorentz(B_plot, S0, A1, G1, A2, G2, A3, G3)
    S1  = -2 * B_plot * A1 / (B_plot**2 + G1**2) ** 2
    S2  = -2 * B_plot * A2 / (B_plot**2 + G2**2) ** 2
    S3  = -2 * B_plot * A3 / (B_plot**2 + G3**2) ** 2

    plt.figure(figsize=(6, 4))
    plt.scatter(B, I, c='r',  s=0.1, label='data')
    plt.plot(B_plot, fit,  'k-', lw=1.5, label='compound')
    plt.plot(B_plot, S1,  'C0--', lw=1, label='S1 (wide)')
    plt.plot(B_plot, S2,  'C1--', lw=1, label='S2 (sharp)')
    plt.plot(B_plot, S3,  'C2--', lw=1, label='S3 (narrow flip)')
    plt.xlabel('B (G)')
    plt.ylabel('I (arb)')
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(path / "fit.png", dpi=300)
    plt.close()

    return theta_best, sampler

if __name__ == "__main__":
    file = "./data/raw/[EDMR]_2G_3V_200MHz.pkl"
    with open(file, "rb") as f:
        d = pickle.load(f)

    d = d[d["B (Gauss)"].abs() <= 50]
    B, I = d["B (Gauss)"].to_numpy(), d["I (nA)"].to_numpy()

    best, sampler = run_mcmc(B, I)
    print("MAP parameters:", best)

