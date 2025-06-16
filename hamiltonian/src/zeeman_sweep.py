#!/usr/bin/env python3
import os 
import tqdm
import yaml
import numpy as np
import matplotlib.pyplot as plt

from utils.eigensolver import Eigensolver
from pathlib import Path

import logging 
logging.basicConfig(level=logging.INFO)

""" 
Sweeps from -100G to +100G and plots Energy vs. B (G) for Eigenvectors 
of the Zeeman Hamiltonian. 

Since Zeeman is diagonalized already, the eigenvectors are in the coupled basis. 
"""

yaml_path = Path.home() / "nasa" / "hamiltonian" / "src" / "utils" / "params.yaml"

def update_B0_in_yaml(B_value):
    # Load params.yaml, update the zeeman.B0 key, and write back.
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    if 'zeeman' not in data:
        data['zeeman'] = {}
    data['zeeman']['B0'] = float(B_value)
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(data, f)

def main(
        verbose: bool = False, 
        outdir = Path.home() / "nasa" / "hamiltonian" / "src" / "media"
):
    solver = Eigensolver("zeeman_only", verbose=verbose)
    solver.load_params()

    B_fields = np.linspace(-100, 100, 201)
    energies = np.zeros((len(B_fields), 16))

    for i, B in enumerate(tqdm.tqdm(B_fields)):
        update_B0_in_yaml(B )

        # Build & diagonalize via Eigensolver
        solver._set_hamiltonian()            # populates solver.H
        solver._spectral_decomposition()     # populates solver.w, solver.v
        energies[i] = solver.w.real          # grab the eigenvalues

    plt.figure(figsize=(6,4))
    for k in range(16):
        plt.plot(B_fields, energies[:, k], lw=1, label=f"{solver._BASIS_MAP.get(k, "")}")
    plt.xlabel("Magnetic Field $B_0$ (G)")
    plt.ylabel("Energy (arb. units)")
    plt.title("Zeeman-only Energy vs. B")
    plt.legend(fontsize=6)
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    fname = outdir / "zeeman_sweep.png"
    plt.savefig(fname , dpi=300)
    logging.info(f"Zeeman Plot saved to {fname}.")
    

if __name__ == "__main__":
    main()