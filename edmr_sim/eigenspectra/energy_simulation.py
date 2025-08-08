import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from .utils.eigensolver import PARAMETER_FILE, Eigensolver 
from .utils.update_yaml import update_B0
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO)

PARAMETER_FILE = Path(__file__).parent / "utils/params.yaml"

def allowed_method_names(): 
    """ Displays the allowed method names """
    print(Eigensolver(method_name="")._PARAM_SECTIONS.keys())


def sweep(
        method_name: str, 
        b_sweep: tuple  = (-40, 40, 201),
        verbose: bool   = False, 
        outdir: Path    = PARAMETER_FILE.parent.parent / "media/"
):
    """
    Sweeps across B and plots energy [eV] vs. magnetic field B [G] 
    for the provided `method_name` Hamiltonian.

    For allowed `method_name(s)`, see utils.eigensolver::_PARAM_SECTIONS

    Maintains consistent eigenvector labeling via Hungarian (Kuhn–Munkres)
    matching. Since with complicated Hamiltonians the eigenbasis isn't the 
    same as the coupled-triplet basis, this is the next best thing 
    labeling-wise. 

    I assign colors by ordering modes by their average energy, 
    higher energies are redder, lower energies are bluer. 

    """

    logging.info(f" Starting eigensimulation for {method_name!r}")
    
    solver = Eigensolver(
        method_name=method_name, 
        verbose=verbose, 
        yaml_path=PARAMETER_FILE
    ) 
    params = solver.load_params()

    B_fields = np.linspace(*b_sweep)
    n_states = 16
    energies = np.zeros((len(B_fields), n_states))
    
    prev_v = None  # eigenvectors from previous B
    labels = None  # consistent labels across B

    for i, B in enumerate(tqdm.tqdm(B_fields)):
        # updates param file with B
        update_B0(PARAMETER_FILE, B)          
                                             
        # builds hamiltonian and populates solver.w, solver.v 
        # eigenvectors and eigenvalues 
        solver._set_hamiltonian()
        solver._spectral_decomposition()    
    
        if prev_v is None:
            solver._log_eigenvectors()
            energies[i] = solver.w.real
            prev_v      = solver.v          
            labels      = solver.labels.copy()
        else:
            assert labels is not None

            # hungarian matching
            C = np.abs(prev_v.conj().T @ solver.v)
            _, col_idx = linear_sum_assignment(-C)

            # reorder energies, eigenvectors, labels
            energies[i] = solver.w[col_idx].real
            prev_v      = solver.v[:, col_idx]
            labels      = [labels[j] for j in col_idx]

    # find degeneracies  
    intersections = []
    for k in range(n_states):
        for l in range(k+1, n_states):
            delta = energies[:,k] - energies[:,l]
            idx = np.where(delta[:-1] * delta[1:] < 0)[0]
            for j in idx:
                t = delta[j] / (delta[j] - delta[j+1])
                B_cross = B_fields[j] + t*(B_fields[j+1] - B_fields[j])
                E_cross = energies[j,k] + t*(energies[j+1,k] - energies[j,k])
                intersections.append((B_cross, E_cross))
    intersections = np.array(intersections) 

    # red -> blue from top-to-bottom
    mean_energies = energies.mean(axis=0)
    ranks = np.argsort(np.argsort(mean_energies))
    colors = plt.cm.jet(np.linspace(0, 1, n_states)) # type: ignore
    state_colors = colors[ranks]

    fig, axis = plt.subplots(1, 1, figsize=(6.55, 5))
    for k in range(n_states):
        assert labels is not None 
        axis.plot(B_fields, energies[:, k], lw=1, c=state_colors[k], label=labels[k])

    # plot degeneracies 
    if intersections.size:
        axis.scatter(
            intersections[:,0],
            intersections[:,1],
            marker=".",
            s=5,      
            c='k',  
            zorder=5,
            label='_degeneracies_'  
        )

    plt.xlabel("B [G]")
    plt.ylabel("Energy [eV]")
    plt.legend(fontsize=6, loc='upper right', ncol=2)
    plt.title(f"{method_name} Energy [eV] vs. B [G]")

    # displaying parametrs on right of the fig
    unit_map = {
        'J':      'eV',
        'Aa1':    'eV',
        'Aa2':    'eV', 
        'Ab1':    'eV', 
        'Ab2':    'eV', 
        'A2_iso': 'eV',
        'B0':     'G',
        'mu_B':   'eV/G',
        'mu_N':   'eV/G',
        'D1':     'eV',
        'D2':     'eV',
    }
    max_key_len = max(len(k) for k in params)
    param_lines = []
    for k, v in params.items():
        unit = unit_map.get(k, '')
        unit_str = f" {unit}" if unit else ''
        key_str = k.rjust(max_key_len)
        value_str = f"{float(v):.3g}"
        param_lines.append(f"{key_str} = {value_str}{unit_str}")

    max_line = max(len(s) for s in param_lines)
    title   = "Parameters"
    underline = "‾" * max(max_line, len(title))
    param_text = (
        f"{title.center(len(underline))}\n"
        f"{underline}\n"
        + "\n".join(param_lines)
    )

    fig.subplots_adjust(right=0.9)
    axis.text(
        1.05, 0.5,
        param_text,
        transform=axis.transAxes,
        va="center", ha="left",
        fontsize=8,
        family="monospace",
        bbox=dict(
            facecolor="white",
            edgecolor="gray",
            alpha=0.8,
        ),
    )

    file_key = method_name
    if method_name == "zeeman_hyperfine_zfs_exchange":
        file_key = "full_spin_hamiltonian"

    outdir.mkdir(exist_ok=True, parents=True)
    fname = outdir / f"{file_key}_sweep.png"
    plt.tight_layout()
    plt.show()
    fig.savefig(fname, dpi=300)
    logging.info(f" {file_key!r} simulation saved to \n {fname}.\n")


def _run_all(
    b_sweep: tuple = (-40, 40, 200), 
    verbose: bool  = False,
    outdir: Path   = PARAMETER_FILE.parent.parent / "media/"
):
    """ 
    Run the eigensimulation for every Hamiltonian combination. 

    """

    for method_name in [
        "zeeman_only", 
        "hyperfine_only",
        "exchange_only", 
        "zfs_only", 

        "zeeman_hyperfine",
        "zeeman_exchange", 
        "zeeman_zfs", 
        "hyperfine_exchange", 
        "hyperfine_zfs", 
        "zfs_exchange",

        "zeeman_zfs_exchange", 
        "zeeman_hyperfine_zfs", 
        "zeeman_hyperfine_exchange", 
        "hyperfine_zfs_exchange", 

        "zeeman_hyperfine_zfs_exchange", 
    ]: 
        sweep(
            method_name = method_name, 
            b_sweep     = b_sweep,      
            verbose     = verbose, 
            outdir      = outdir, 
        )
     
if __name__ == "__main__":

    # Example usage 
    # ------------- 

    # Performs a single simulation on the Full Spin Hamiltonian 
    # Zeeman + Hyperfine + ZFS + Exchange 
    sweep(
       method_name = "zeeman_hyperfine_zfs_exchange", 
       b_sweep=(-40, 40, 201), 
       verbose=False, 
       outdir = Path.home() / "nasa/hamiltonian/src/" / "media/"
    ) 

    # Performs simulation for all Spin Hamiltonian sub-combinations
    # _run_all()

