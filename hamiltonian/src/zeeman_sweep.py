#!/usr/bin/env python3
import os 
import tqdm
import yaml
import numpy as np
import matplotlib.pyplot as plt

from utils.eigensolver import Eigensolver
from utils.update_yaml import update_B0
from pathlib import Path

import logging 
logging.basicConfig(level=logging.INFO)

""" 
Sweeps from -100G to +100G and plots Energy vs. B (G) for Eigenvectors 
of the Zeeman Hamiltonian. 

Since Zeeman is diagonalized already, the eigenvectors are in the coupled basis. 
"""

yaml_path = Path.home() / "nasa/hamiltonian/src/utils" / "params.yaml"

def main(
        verbose: bool = False, 
        outdir = Path.home() / "nasa/hamiltonian/src/" / "media"
):
    solver = Eigensolver("zeeman_only", verbose=verbose)
    solver.load_params()

    B_fields = np.linspace(-100, 100, 201)
    energies = np.zeros((len(B_fields), 16))

    for i, B in enumerate(tqdm.tqdm(B_fields)): 
        update_B0(yaml_path, B)      # type: ignore

        # Build & diagonalize via Eigensolver
        solver._set_hamiltonian()            # populates solver.H
        solver._spectral_decomposition()     # populates solver.w, solver.v
        energies[i] = solver.w.real          # type: ignore grab the eigenvalues

    # Generating Sweep plot
    fig, axis = plt.subplots(1, 1, figsize=(5, 5))
    colors = plt.cm.jet(np.linspace(0, 1, 16)) # type: ignore 

    for k in range(16):
        axis.plot(B_fields, energies[:, k], lw=1, alpha=0.5, c=colors[k], label=f"{solver._BASIS_MAP.get(k, "")}")
    plt.xlabel("Magnetic Field $B_0$ (G)")
    plt.ylabel("Energy (eV)")
    plt.title("Zeeman-only Energy vs. B")
    plt.legend(fontsize=6)
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    fname = outdir / "zeeman_sweep.png"
    fig.savefig(fname , dpi=300)
    logging.info(f"Zeeman Plot saved to {fname}.")
    

if __name__ == "__main__":
    main()
