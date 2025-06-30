import pickle
import numpy as np 
import sympy as sp 
import matplotlib.pyplot as plt    

from tqdm import tqdm   
from pathlib import Path 
from sle_solver import SteadyStateSLESolver

import logging 
logging.basicConfig(level = logging.INFO) 

def load_params(
        param_file: Path = Path.home() / "nasa/SLE/main/utils/params.yaml"
): 
    """
    Loads parameters from `param_file`. 

    """
    from ruamel.yaml import YAML 

    with open(param_file) as f: 
        raw = YAML().load(f) 
    
    def flatten(tree, prefix=""):
        items = {}
        for k, v in tree.items():
            key = f"{prefix}{k}" if prefix == "" else f"{prefix}{k}"
            if isinstance(v, dict):
                items.update(flatten(v, prefix=f"{k}_"))
            else:
                items[key] = float(v)
        return items

    hamiltonian_params = flatten({
        **raw["exchange"],
        **raw["hyperfine"],
        **raw["zeeman"],
        **raw["zfs"]
    })
    phys_keys = list(hamiltonian_params.keys())

    k_s = raw["sle"]["k_S"]
    k_d = raw["sle"]["k_D"]
    p   = raw["sle"]["p"]
    hbar= raw["sle"]["hbar"]
    return hamiltonian_params, k_s, k_d, p, hbar, phys_keys

def projection_operators(): 
    """
    Returns 16x16 singlet-triplet projection operators 
    for a 2-electron x 2-nuclei subspace 

    """
    # build projection operators 
    up, dn = np.array([1, 0]), np.array([0, 1])
    # singlet 
    S = (np.kron(up, dn) - np.kron(dn, up)) / np.sqrt(2) 
    Lambda_S = np.outer(S, S.conj())            # 4x4
    # triplet
    T_plus  = np.kron(up, up)
    T_zero  = (np.kron(up, dn) + np.kron(dn, up)) / np.sqrt(2) 
    T_minus = np.kron(dn, dn)
    Lambda_T = (
        np.outer(T_plus, T_plus.conj()) 
        + np.outer(T_zero, T_zero.conj())
        + np.outer(T_minus, T_minus.conj())
    )                                           # 4x4
    # nuclear
    I_nuc = np.eye(4) 
    Lambda_S = np.kron(Lambda_S, I_nuc)         # 16x16 
    Lambda_T = np.kron(Lambda_T, I_nuc)         # 16x16 
    return Lambda_S, Lambda_T

def make_singlet_fn(
    solver: SteadyStateSLESolver,
    Lambda_S: np.ndarray,
    *,
    k_s: float,
    k_d: float,
    p: float,
):
    """
    Returns singlet population function that takes in `B_array` 
    and outputs Tr(Lambda_S @ rho(B_array)). 

    """
    def singlet_fn(B_array: np.ndarray, **params) -> np.ndarray:
        pop = np.empty_like(B_array, dtype=float)
        for i, B in enumerate(tqdm(B_array)):
            params["B0"] = float(B)
            rho = solver.rho(k_s=k_s, k_d=k_d, p=p, params=params)
            pop[i] = np.real_if_close(np.trace(Lambda_S @ rho))
        return pop
    return singlet_fn

def plot_sing_population(
        bmin, bmax, n_points, 
        derivative=True, 
        outdir = Path.home() / "nasa/SLE/main/media/", 
        save=True
): 
    """ 
    Sweeps from `bmin` to `bmax` through `n_points`, plotting
    the simulated singlet population lineshape as a function of B. 

    The singlet population is directly prop. to the spin current. 
    Should reproduce a clean lorentzian around B=0. 

    Args:
        * bmin, bmax: float
            * min, max B field value for sweep. 
        * n_points: int 
            * number of points to calculate singlet pop. for
            * from bmin to bmax. 
        * derivative: bool 
            * derivative lineshape or not. 
            * (i.e. dI/dB or I)
        * outdir: pathlib.Path 
            * directory to place output png.
        * save: bool
            * whether to save to outdir/
    """

    b_sweep = np.linspace(bmin, bmax, n_points)
    i_rel   = np.empty_like(b_sweep, dtype=float)
    singlet_fn = make_singlet_fn(
        Lambda_S=Lambda_S,
        k_s=k_s,
        k_d=k_d,
        p=p, 
        solver=solver, 
    )
    i_rel = singlet_fn(b_sweep, **hamiltonian_params)

    fig, axis = plt.subplots(1, 1, figsize=(10, 5)) 
    axis.plot(b_sweep, i_rel, lw=2, label="singlet population")
    axis.axvline(0, ls=':', color='grey')
    axis.set_xlabel("B (G)", fontsize=15)
    axis.set_ylabel(r"$\mathrm{Tr}(\Lambda_S\,\rho)$ (arb.)", fontsize=15)
    axis.set_title("singlet population vs. magnetic field (G)", fontsize=20)
 
    if derivative: 
        fig2, axis2 = plt.subplots(1, 1, figsize=(10, 5))
        i_rel = np.gradient(i_rel, b_sweep)

        axis2.plot(b_sweep, i_rel, lw=2, label="singlet population")
        axis2.axvline(0, ls=':', color='grey')
        axis2.set_xlabel("B (G)", fontsize=15)
        axis2.set_ylabel(r"$\frac{\partial\mathrm{Tr}(\Lambda_S\,\rho)}{\partial B}$ (arb.)", fontsize=15)
        axis2.set_title("d(singlet population)/dB vs. B (G)", fontsize=20)

    if save: 
        fname = outdir / "singlet_population.png"
        fig.savefig(fname, dpi=300)
        if derivative: 
            fname = outdir / "singlet_population_derivative.png"
            fig2.savefig(fname, dpi=300) # type: ignore 
    plt.close()

if __name__ == "__main__": 

    # symbolic spin hamiltonian 
    with open(Path.home() / "nasa/hamiltonian/pickle/spin_hamiltonian.pickle", "rb") as f: 
        H_sym = pickle.load(f) 

    # initialize solver
    hamiltonian_params, k_s, k_d, p, hbar, _ = load_params() 
    Lambda_S, Lambda_T = projection_operators() 
    solver = SteadyStateSLESolver( 
        H_sym       = H_sym, 
        Lambda_S    = sp.Matrix(Lambda_S), 
        Lambda_T    = sp.Matrix(Lambda_T), 
        Gamma       = sp.Matrix(2 * np.eye(Lambda_S.shape[0])),
        hbar        = hbar,
    )

    # plot singlet pop and derivative lineshapes
    plot_sing_population(bmin=-200, bmax=+200, n_points=200, derivative=True)
        
