from __future__ import annotations 
 
import logging, os
import numpy as np 
import matplotlib.pyplot as plt 

from rich import print 
from pathlib import Path 
from utils.load_params import load_params
from sle_model import compute_singlet_spectra

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

""" 
Key: value dictionaries will be used whenever possible. 
This `PARAMETER_ORDER` will only be used as a check when 
building the parameter vector in the future fitting routines. 

"""

PARAMETER_FILE = Path(os.getcwd()).parent / "utils/params.yaml"

PARAMETER_ORDER = [
    "A", "I0",                      # proportion
    "J",                            # exchange 
    "Aa1", "Aa2", "Ab1", "Ab2",     # hyperfine 
    "g_e", "g_n1", "g_n2",          # zeeman 
    "D1", "D2",                     # zfs 
    "nu", "omega1",                 # microwave 
    "k_S", "k_D", "p",              # rates 
    "B_mod",                        # modulation
    "h", "hbar", "mu_B", "mu_N",    # constants 
]

# unused here, will be used in fitting routines
def _convert_dict_to_theta(parameters: dict[str, float]) -> np.ndarray: 
    """ 
    Converts a parameter dictionary into a consistently-ordered 
    np.ndarray parameter vector for fitting routines. 

    Args: 
        * parameters: dict[str, float] 
            * Model parameters. 
            * see utils::load_params.load_params() 
    Returns: 
        * parameter_vector: np.ndarray 
            * array of floats from parameter values.  
            * consistent ordering from PARAMETER_ORDER.   
    """

    parameter_vector = np.zeros(len(PARAMETER_ORDER))
    for idx, name in enumerate(PARAMETER_ORDER): 
        parameter_vector[idx] = parameters[name] 
    return parameter_vector 

# unused here, will be used in fitting routines
def _convert_theta_to_dict(parameters: np.ndarray) -> dict[str, float]: 
    dict_params = {}
    for idx, value in enumerate(parameters): 
        dict_params[PARAMETER_ORDER[idx]] = value
    return  dict_params

def plot_singlet_spectra(
    bmin: float, 
    bmax: float, 
    n_points: int, 
    parameters: dict[str, float], 
    n_jobs: int, 
    blocks_per_core: int, 
    modulate: bool, 
    outdir: Path, 
    save: bool, 
):
    """
    Computes and plots d(singlet pop)/dB across a field sweep. 

    Args: 
        * bmin, bmax: float 
            * Bounds of simulated B-sweep. Bounds are included.
        * n_points : int 
            * Number of points to sweep across [bmin, bmax]
        * parameters: dict[str, float] 
            * Model parameters. 
            * see utils::load_params.load_params() 
        * n_jobs: int 
            * Number of cores (1 if not multithreading) 
        * blocks_per_core: int 
            * Numer of B-blocks per core. 
        * modulate: bool 
            * If lockin modulation is active 
        * outdir: Path 
            * Directory to place final plot 
        * save: bool 
            * Whether to save plot to `outdir` 

    """

    print(
        f"Rendering parallelized singlet spectrum " 
        f"({"lockin" if modulate else "raw"}) - "
        f"{n_points} points in [{bmin:.1f}, {bmax:.1f}] G",
    )

    B = np.linspace(bmin, bmax, n_points)
    dI = compute_singlet_spectra(
        B_array=B, 
        parameters=parameters,  
        modulate=modulate, 
        n_jobs=n_jobs, 
        blocks_per_core=blocks_per_core, 
        show_progress=True 
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 5)) 
    fig.subplots_adjust(right=0.8)

    ax.plot(B, dI, lw=2, label=f"d(singlet_pop)/dB - {"lock-in" if modulate else "raw"}")

    ax.axvline(0, ls=':', color='grey') 
    ax.axhline(0, ls=':', color='grey') 
    ax.set_xlabel("B [G]", fontsize=12)
    ax.set_ylabel("d(singlet_pop)/dB [arb.]", fontsize=12)
    ax.set_title(f"Singlet Derivative Sweep -- {n_points} steps", fontsize=14)
    ax.legend()

    # Adding parameter box to figure 
    max_parameter_key_len = max(len(k) for k in parameters)

    parameter_lines = [
    f"{k.ljust(max_parameter_key_len)} = {parameters[k]:.4g}"
    for k in parameters
    ]
    parameter_text = "\n".join(parameter_lines)

    bbox_props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
    ax.text(
        1.02, 0.5, parameter_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="center",
        horizontalalignment="left",
        bbox=bbox_props,
        fontfamily="monospace" 
    )

    if save: 
        outdir.mkdir(parents=True, exist_ok=True) 

        fname_mod    = f"{parameters["B_mod"]}Gpp" if modulate else "raw" 
        fname_brange = f"{bmin}_to_{bmax}_in_{n_points}steps" 
        fname_full   = outdir / f"singlet_{fname_brange}-{fname_mod}.png" 

        fig.savefig(fname_full, dpi=300)
        logger.info(" fig::%s \n", fname_full)
    return fig 


if __name__ == "__main__": 
    plot_singlet_spectra( 
        bmin=-85, 
        bmax=+85, 
        n_points=100, 
        parameters = load_params(Path("./utils/params.yaml")), 
        n_jobs=2, 
        blocks_per_core=8, 
        modulate=True, 
        outdir=Path("../media/"), 
        save=True, 
    )


    

    
    

