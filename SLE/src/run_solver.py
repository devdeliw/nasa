#!/usr/bin/env python3
from __future__ import annotations

import logging
import psutil
import numpy as np 
import matplotlib.pyplot as plt 

from pathlib import Path
from ruamel.yaml import YAML
from sle_model import singlet_spectra

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_params(
    param_file: Path = Path(__file__).resolve().parent / "utils/params.yaml"
):
    """
    Read and flatten the hamiltonian YAML param_file.

    """

    with open(param_file, "r", encoding="utf-8") as f:
        raw = YAML().load(f)

    def _flatten(tree):
        out: dict[str, float] = {}
        for k, v in tree.items():
            if isinstance(v, dict):
                out.update(_flatten(v))
            else:
                out[k] = float(v)
        return out

    hamiltonian_params = _flatten(
        {
            **raw["exchange"],
            **raw["hyperfine"],
            **raw["zeeman"],
            **raw["zfs"],
            **raw["microwave"],
            **raw["constants"]
        }
    )

    k_s = float(raw["sle"]["k_S"])
    k_d = float(raw["sle"]["k_D"])
    p_gen = float(raw["sle"]["p"])

    # lock‑in settings
    f_mod = float(raw["lockin"]["f_mod"])  # unused
    B_mod = float(raw["lockin"]["B_mod"])

    A  = float(raw["proportion"]["A"])
    I0 = float(raw["proportion"]["I0"])

    return A, I0, hamiltonian_params, k_s, k_d, p_gen, B_mod

# constant to maintain parameteter ordering
P_ORDER = [
    "J",
    "Aa1",
    "Ab1",
    "Aa2",
    "Ab2",
    "D1",
    "D2",
    "B0",  # dummy
    "g_e",
    "g_n1",
    "g_n2",
    "nu",
    "omega1",
]  

def make_pvec(params: dict[str, float], k_s: float, k_d: float, p: float, B_mod: float) -> np.ndarray:
    """
    Packs the 17-element parameter vector expected by `singlet_spectra`.
    
    """

    try:
        core = [params[k] if k != "B0" else 0.0 for k in P_ORDER]  # order properly
    except KeyError as miss:
        raise KeyError(f"missing {miss!s} in params.yaml") from miss

    core.extend([k_s, k_d, p, B_mod])
    return np.asarray(core, dtype=float)

def make_fullpvec(params: dict[str, float], k_s: float, k_d: float, p: float, B_mod: float, A: float, I0: float) -> np.ndarray: 
    """
    Packs the 19-element parameter vector expected by `edmr_spectra`. 

    """
    full = [A, I0]
    try:
        core = [params[k] if k != "B0" else 0.0 for k in P_ORDER]  # order properly
    except KeyError as miss:
        raise KeyError(f"missing {miss!s} in params.yaml") from miss
    full.extend(core)
    full.extend([k_s, k_d, p, B_mod])
    return np.asarray(full, dtype=float)


def plot_singlet_spectrum(
    bmin: float,
    bmax: float,
    n_points: int,
    n_jobs = psutil.cpu_count(logical=False),
    *,
    pvec: np.ndarray,
    modulate: bool = True,
    outdir: Path = Path(__file__).resolve().parent / "media/",
    save: bool = True,
):
    """
    Compute and plot *d(singlet pop)/dB* across a field sweep.
    
    """

    B = np.linspace(bmin, bmax, n_points)
    logger.info(
        " Rendering parallelized singlet spectrum (%s) - %d points in [%.1f, %.1f] G",
        "lock-in" if modulate else "raw", n_points, bmin, bmax,
    )

    dI = singlet_spectra(B, pvec, modulate=modulate, n_jobs=n_jobs)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(B, dI, lw=2, label=f"dI/dB - {'lock-in' if modulate else 'raw'}")
    ax.axvline(0, ls=":", color="grey")
    ax.axhline(0, ls=":", color="grey")
    ax.set(
        xlabel="B [G]",
        ylabel=r"$\mathrm{d}I/\mathrm{d}B$ [arb.]",
        title=f"Singlet derivative sweep - {n_points} steps",
    )
    ax.legend()

    if save:
        outdir.mkdir(parents=True, exist_ok=True)
        tag = f"{pvec[-1]}Gpp" if modulate else "raw"
        fname = outdir / f"singlet_{bmax}G_{n_points}steps_{tag}.png"
        fig.savefig(fname, dpi=300)
        logger.info(" fig::%s \n", fname)

    return fig


if __name__ == "__main__":
    _, _, hamiltonian_params, k_s, k_d, p_gen, B_mod = load_params()
    pvec = make_pvec(
        params=hamiltonian_params, 
        k_s=k_s, 
        k_d=k_d, 
        p=p_gen, 
        B_mod=B_mod
    )
    plot_singlet_spectrum(
        bmin=-100, 
        bmax=+100, 
        n_points=100, 
        pvec=pvec,
        modulate=True
    )
