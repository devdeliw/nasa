from __future__ import annotations

import logging
import pickle
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt

from pathlib import Path
from ruamel.yaml import YAML

from sle_solver import SteadyStateSLESolver
from run_solver import make_singlet_fn, load_params, projection_operators
from mcmc import EDMRDerivativeMCMC

DEFAULT_PARAMFILE = Path.home() / "nasa/SLE/main/utils/params.yaml"
DEFAULT_HAM       = Path.home() / "nasa/hamiltonian/pickle/spin_hamiltonian.pickle"
DEFAULT_OUTDIR    = Path.home() / "nasa/SLE/main/media/EDMR/"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("__name__")

def fit_edmr(
    df: pd.DataFrame,
    *,
    paramfile           : Path = DEFAULT_PARAMFILE,
    hamiltonian_pickle  : Path = DEFAULT_HAM,
    outdir              : Path = DEFAULT_OUTDIR,
    nsteps              : int = 2000,
    burn                : int = 1000,
    sigma0              : float = 1e-3,
    progress            : bool = True,
):
    """
    Run the EDMR derivative MCMC fit and plot best-fit overlay.

    Args: 
    * df : pandas.DataFrame
        * must contain columns "B (Gauss)" and "I (nA)".
    * paramfile : Path
        * yaml file with all parameters.
    * hamiltonian_pickle : Path
        *.pickle* containing the SymPy 16x16 Hamiltonian.
    * outdir : Path, default `~/nasa/SLE/main/media/EDMR/`
    * nsteps, burn, sigma0 : int / float
        * MCMC configuration.
    * progress : bool
        * show tqdm progress bar.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    B_exp = df["B (Gauss)"].to_numpy()
    I_exp = df["I (nA)"].to_numpy()

    base_phys, ks, kd, p, hbar, phys_keys = load_params(paramfile)
    Lambda_S, Lambda_T = projection_operators()

    with hamiltonian_pickle.open("rb") as f:
        H_sym = pickle.load(f)

    solver = SteadyStateSLESolver(
        H_sym=H_sym,
        Lambda_S=sp.Matrix(Lambda_S),
        Lambda_T=sp.Matrix(Lambda_T),
        Gamma=sp.Matrix(2 * np.eye(16)),
        hbar=hbar,
    )

    singlet_fn = make_singlet_fn(solver, Lambda_S, k_s=ks, k_d=kd, p=p)

    # run mcmc
    mcmc = EDMRDerivativeMCMC(
        B_data=B_exp,
        I_data=I_exp,
        singlet_fn=singlet_fn, # type: ignore
        base_phys=base_phys,
        sigma0=sigma0,
        phys_keys=phys_keys
    )
    mcmc.run_mcmc(nsteps=nsteps, burn=burn, progress=progress)
    mcmc.summary()


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_pickle(Path.home() / "nasa/spectra/src/data/raw/[EDMR]_2G_3V_200MHz.pkl")
    df = df.loc[
        (df["B (Gauss)"] >= -40.0) & (df["B (Gauss)"] <= 40.0)
    ].reset_index(drop=True)

    fit_edmr(df)
