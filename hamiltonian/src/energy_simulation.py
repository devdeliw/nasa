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

# parameter file location
yaml_path = Path.home() / "nasa/hamiltonian/src/utils" / "params.yaml" 

def sweep(
        method_name: str, 
        b_sweep: tuple  = (-40, 40, 201),
        verbose: bool   = False, 
        outdir: Path    = Path.home() / "nasa/hamiltonian/src/" / "media"
):
    
    """
    Sweeps across B and plots Energy [eV] vs. B [G] for the 
    provided `method_name` Hamiltonian (See Args). 

    I use the Kuhn-Munkres algorithm to compute the optimal one-to-one matching 
    of eigenvectors between successive B-steps, consistent labeling. This is because 
    the singlet-triplet basis isn't the eigenbasis, so this is the next best-thing 
    labelling-wise. 

    Args: 
        * method_name: str 
            The Hamiltonian Combination to Calculate. You can see allowed method_name's in utils/hamil.py. 
        * b_sweep: tuple 
            The Magnetic Field B Sweep Range in Gauss. (-40, 40, 200) will sweep across -40G to 40G in 200 steps. 
        * verbose: bool 
            If you want more information to be printed as the algorithm runs. 
        * outdir: Path 
            Where to place the final generated image files.

    """
    
    solver = Eigensolver(method_name=method_name, verbose=verbose) 
    params = solver.load_params()

    B_fields = np.linspace(*b_sweep)
    energies = np.zeros((len(B_fields), 16))
    
    prev_v = None # hold eigenvectors from previous B 
    labels = None # hold eigenvector labels across B 
    for i, B in enumerate(tqdm.tqdm(B_fields)): 
        update_B0(yaml_path, B)      # type: ignore

        # Build & diagonalize via Eigensolver
        solver._set_hamiltonian()            # populates solver.H
        solver._spectral_decomposition()     # populates solver.w, solver.v
        
        # Maintain eigenvector ordering from LAPACK algorithm.
        if prev_v is None:
            solver._log_eigenvectors()           # populates solver.combos
            energies[i] = solver.w.real          # type: ignore
            prev_v      = solver.v               # (16,16), columns are eigenvectors 
            labels      = solver.labels.copy() 
        else:
            C = np.abs(prev_v.conj().T @ solver.v)      # 16×16 overlap matrix
            order = np.argmax(C, axis=1)                # for each old column, find new one with max overlap 
            
            # Re-order eigenvalues & eigenvectors
            energies[i] = solver.w[order].real  # type: ignore 
            prev_v      = solver.v[:, order]    # type: ignore

            # Also Re-order labels 
            labels = [labels[old_idx] for old_idx in order] # type: ignore

    # Generating Sweep plot
    fig, axis = plt.subplots(1, 1, figsize=(6, 5))
    colors = plt.cm.jet(np.linspace(0, 1, 16)) # type: ignore 

    for k in range(16):
        axis.plot(B_fields, energies[:, k], lw=1, c=colors[k], label=labels[k]) # type: ignore
    
    plt.xlabel("B [G]")
    plt.ylabel("Energy [eV]")
    plt.legend(fontsize=6)
    plt.title(f"{method_name} Energy [eV] vs. B [G]")

    # Showing the simulation parameters in the plot 
    param_lines = [f"{k} = {float(v):.3g}" for k, v in params.items()]
    param_text  = "\n".join(param_lines)
    fig.subplots_adjust(right=0.75)
    axis.text(
        1.02, 0.5,                          
        param_text,
        transform=axis.transAxes,
        va="center", ha="left",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="white",
                  edgecolor="gray",
                  alpha=0.8),
    )
    
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    fname = outdir / f"{method_name}_sweep.png"
    fig.savefig(fname , dpi=300)
    logging.info(f"{method_name} simulation saved to {fname}.")
    
    

if __name__ == "__main__":
    sweep( 
       method_name = "hyperfine_zfs_exchange", 
       b_sweep=(-40, 40, 201), 
       verbose=False, 
       outdir = Path.home() / "nasa/hamiltonian/src/" / "media"
    )

