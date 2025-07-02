import pandas as pd
from edmr_mcmc import EDMR_MCMC  
from pathlib import Path 
import logging 
logging.basicConfig(level=logging.INFO)

DATA_DIR = Path.home() / "nasa/spectra/src/data/raw/" 

if __name__ == "__main__":
    
    fname = "[EDMR]_2G_3V_200MHz.pkl"
    df = pd.read_pickle(DATA_DIR / fname)
    
    try: 
        B = df["B (Gauss)"].to_numpy()
        I = df["I (nA)"].to_numpy()
    except KeyError: 
        raise KeyError(f" Invalid column headers for {fname}.")

    model = EDMR_MCMC(B, I, sigma=1e-3)

    nsteps   = 5000   # total steps per walker
    burn     = 1000   # burn-in
    nwalkers = None   # default 4*num_parameters
    threads  = None   # use all CPU cores
    samples  = model.run_mcmc(nsteps, burn=burn, nwalkers=nwalkers, threads=threads)
    

    model.summary()
    fig = model.plot()
    fname = f"{fname}_fit.png"
    fdir  = Path.home() / "nasa/sle/media/EDMR/"
    fdir.mkdir(exist_ok=True, parents=True)
    fig.savefig(fdir / fname, dpi=300)
