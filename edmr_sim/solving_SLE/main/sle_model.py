# This code was written by Deval Deliwala 
# NASA Glenn Research Center 

import pickle, tqdm, logging
from typing import Tuple
import numpy as np, sympy as sp 

from pathlib import Path 
from functools import lru_cache
from .sle_solver import SteadyStateSLESolver
from concurrent.futures import ProcessPoolExecutor, as_completed
    
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

# SymPy symbols in the Spin Hamiltonian 
g_e, g_n1, g_n2, mu_B, mu_N, B0 = sp.symbols("g_e g_n1 g_n2 mu_B mu_N B0") 
h, hbar, nu, omega1             = sp.symbols("h hbar nu omega1") 
Aa1, Aa2, Ab1, Ab2              = sp.symbols("Aa1 Aa2 Ab1 Ab2") 
J, D1, D2                       = sp.symbols("J D1 D2")

# Load the Static Spin Hamiltonian 
# And the Secular Hyperfine Hamiltonian 
hamiltonian_directory = Path(__file__).parent.parent.parent / "derive_hamiltonian/pickle/"

spin_fname   = hamiltonian_directory / "spin_hamiltonian.pickle"
hf_sec_fname = hamiltonian_directory / "hyperfine_sec.pickle" 
try: 
    with open(spin_fname, "rb") as f: 
        SPIN_HAMILTONIAN = sp.Matrix(pickle.load(f)) 
except FileNotFoundError: 
    logger.error(f"Checked {spin_fname}")
    raise FileNotFoundError("Unable to locate static Spin Hamiltonian.")
try: 
    with open(hf_sec_fname, "rb") as f: 
        HYPERFINE_SECULAR_HAMILTONIAN = sp.Matrix(pickle.load(f))
except FileNotFoundError: 
    logger.error(f"Checked {hf_sec_fname}")
    raise FileNotFoundError("Unable to locate secular Hyperfine Hamiltonian.")

@lru_cache(maxsize=None)
def _build_spin_operators(): 
    """
    Builds the 16x16 Pauli-X and Pauli-Z Operators 
    in the 2-electron + 2-nuclei Hilbert space. 

    """

    idxs = {
        (1, +1): 0, 
        (1, 0) : 1, 
        (0, 0) : 2, 
        (1, -1): 3
    }
    
    # electron 4x4 subspace 
    S_plus_electron = sp.zeros(4) 
    for (s, m),i in idxs.items():
        if m < s:
            S_plus_electron[idxs[(s, m+1)], i] = sp.sqrt(s*(s+1)-m*(m+1))
    S_minus_electron = S_plus_electron.T 
    Sx_electron      = (S_plus_electron + S_minus_electron)/2 
    Sz_electron      = sp.diag(+1, 0, 0, -1)

    # nuclear 4x4 subspace 
    I4 = sp.eye(4) 

    # Full 16x16 Hilbert space 
    Sx_total = sp.kronecker_product(Sx_electron, I4) 
    Sz_total = sp.kronecker_product(Sz_electron, I4) 
    return Sx_total, Sz_total 


@lru_cache(maxsize=None)
def _build_projection_operators(): 
    """
    Builds the 16x16 singlet/triplet projection Operators 
    in the 2-electron + 2-nuclei Hilbert space. 

    """
    up, down = np.array([1, 0]), np.array([0, 1])

    singlet       = (np.kron(up, down) - np.kron(down, up)) / np.sqrt(2) 
    triplet_plus  = np.kron(up, up) 
    triplet_zero  = (np.kron(up, down) + np.kron(down, up)) / np.sqrt(2) 
    triplet_minus = np.kron(down, down) 

    # Projectors in electron 4x4 subspace 
    Lambda_S_electron = np.outer(singlet, singlet.conj()) 
    Lambda_T_electron = (
        np.outer(triplet_plus , triplet_plus .conj()) + 
        np.outer(triplet_zero , triplet_zero .conj()) + 
        np.outer(triplet_minus, triplet_minus.conj()) 
    )

    # Nuclear Projector (identity) 
    I_nuclear = np.eye(4) 

    # Projectors in 16x16 Hilbert Space 
    Lambda_S_total = np.kron(Lambda_S_electron, I_nuclear) 
    Lambda_T_total = np.kron(Lambda_T_electron, I_nuclear)
    return Lambda_S_total, Lambda_T_total

# Build spin and projector operators
Sx_total, Sz_total = _build_spin_operators() 
Lambda_S_total, Lambda_T_total = _build_projection_operators() 

@lru_cache(maxsize=None)
def _make_SLE_solver(
    sign: int, 
    parameters: Tuple[Tuple[str, float], ...], 
    init_check: bool = True
): 
    """
    Builds the class instance ready to solve the SLE. 
    Can build the RWA Hamiltonian solver. 

    Args: 
        * sign: int
            * = 0  : static (without microwave) 
            * = +1 : co-rotating        #(RWA)
            * = -1 : counter_rotating   #(RWA) 
    Returns: 
        sle_solver::SteadyStateSLESolver(...) instance. 

    """
    hamiltonian = SPIN_HAMILTONIAN
    hf_sec_ham  = HYPERFINE_SECULAR_HAMILTONIAN

    if sign in (-1, +1): 
        # Builds co-rotating (+1) or counter-rotating (-1) 
        H_rotator_generator = -h * nu * Sz_total + hf_sec_ham
        H_driver            = (h * omega1)/2 * Sx_total 
        hamiltonian += - H_rotator_generator + sign * H_driver 

    params = dict(parameters)
    return SteadyStateSLESolver(
        H_sym=sp.Matrix(hamiltonian), 
        Lambda_S=Lambda_S_total, 
        Lambda_T=Lambda_T_total, 
        Gamma=2*np.eye(16),
        parameters=params,
        init_check=init_check
    )


def _compute_singlet_population(
        indices: np.ndarray, 
        B_block: np.ndarray, 
        parameters: dict[str, float],
        init_check: bool = True, 
): 
    """
    Computes the theoretical singlet population(s) for an array of B values. 
    This method is built for multithreading. 

    Args: 
        * indices: np.ndarray 
            * stores the B-block indices from the full B array being swept over. 
            * i.e. if you want to divide B_array into 16-parts, with 2 threads, 
            * `indices` would be 
            * [
                array([0, 1, 2, 3, 4, 5, 6, 7]), 
                array([8, 9, 10, 11, 12, 13, 14, 15])
              ]
            * and thread-1 would tackle B_blocks 0-7 and thread-2 would tackle 8-15. 
        * B_block: np.ndarray
            * stores the B_blocks from `indices` 
        * parameters: dict[str, float] 
            * stores the model [variable_name: value] parameters. 
    """
    params_key = tuple(sorted(parameters.items()))

    # static hamiltonian solve 
    static_solver = _make_SLE_solver(
        sign=0, 
        parameters=params_key, 
        init_check=init_check,
    ) 

    # RWA hamiltonian solve
    driver_solver = _make_SLE_solver(
        sign=1, 
        parameters=params_key,
        init_check=init_check, 
    ) 

    static_rho = static_solver.rho 
    driver_rho = driver_solver.rho 
    trace      = np.trace 
    proj       = Lambda_S_total

    i_block = np.empty_like(B_block, dtype=float)
    for j, B in enumerate(B_block):
        parameters["B0"] = B
        rho_static = static_rho(parameters=parameters)
        rho_driver = driver_rho(parameters=parameters)

        singlet_population = trace(
            proj @ (rho_static + 0.5 * rho_driver)
        )  
        i_block[j] = singlet_population.real
    return indices, i_block

def _simulate_lockin(
    I_array: np.ndarray, 
    B_array: np.ndarray, 
    B_pp: float,
    *, 
    n_phase=256, phase=0.0
): 
    """
    Idealized first-harmonic lock-in (in-phase, infinite time-constant).

    Args: 
        * I_arr: np.ndarray  
            * Singlet population (same length and ordering as B_arr)
        * B_arr: np.ndarray 
            * Static-field sweep (monotonic, uniform spacing)
        * B_pp: float 
            * Peak-to-peak modulation amplitude [G]
        * n_phase: int 
            * Discrete phase points per cycle    (resolution)
    Returns: 
        * lock-in trace on same B grid. 
    """

    A = B_pp                                
    theta = np.linspace(0.0, 2*np.pi, n_phase, endpoint=False) 
    sin_theta = np.sin(theta + phase)       
    ref = sin_theta                                             
    norm = 2.0 / n_phase 

    B_inst = B_array[:, None] + A * sin_theta[None, :]        
    I_inst = np.interp(B_inst, B_array, I_array)            
    lockin_I = norm * (I_inst @ ref)

    return lockin_I


def compute_singlet_spectra( 
        B_array: np.ndarray, 
        parameters: dict[str, float], 
        modulate: bool, 
        n_jobs: int, 
        blocks_per_core: int, 
        show_progress: bool, 
        init_check: bool = True 
) -> np.ndarray: 
    """ 
    Computes the full singlet population spectra over `B_array`. 
    This spectra is directly proportional to the current for EDMR. 

    Args: 
        * B_array: np.ndarray 
            * B-field swept over
        * parameters: dict[str, float] 
            * Model Parameters 
        * modulate: bool 
            * If lockin modulation is active 
        * n_jobs: int 
            * Number of cores
        * blocks_per_core: int 
            * Number of B-blocks per core. 
        * show_progress: bool 
            * Displays tqdm progress bar
    Returns: 
        if modulate: 
            the lock-in modulated d(singlet_population)/dB array 
            (same length as B_array)
        else: 
            d(singlet_population)/dB array 
            (same length as B_array) 
    """
    N_values = len(B_array) 
    I_values = np.empty(N_values, dtype=float) 

    if n_jobs == 1 and blocks_per_core != 1: 
        logger.warning("n_jobs = 1 but blocks_per_core != 1") 
        logger.warning("Defaulting blocks_per_core to 1.")
        blocks_per_core = 1 

    idx_blocks = np.array_split(np.arange(N_values), n_jobs * blocks_per_core) 
    _compute_singlet_population_args = [] 
    for indices in idx_blocks:
        B_block = B_array[indices] 
        _compute_singlet_population_args.append(
            (indices, B_block, parameters, init_check)
        ) 

    if n_jobs > 1: 
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [
                ex.submit(_compute_singlet_population, *arg) 
                for arg in _compute_singlet_population_args
            ]
            with tqdm.tqdm(
                total=len(futures), 
                disable=not show_progress
            ) as pbar: 
                for fut in as_completed(futures): 
                    indices, i_block = fut.result() 
                    I_values[indices] = i_block 
                    pbar.update(1) 
    else: 
        indices, i_block = tqdm.tqdm(_compute_singlet_population(
            *_compute_singlet_population_args[0]
        ), disable=not show_progress)
        I_values[indices] = i_block 

    if modulate: 
        return _simulate_lockin(
            I_array=I_values, 
            B_array=B_array, 
            B_pp=parameters["B_mod"]
    )
    return np.gradient(I_values, B_array, edge_order=2)


def compute_edmr_spectra(
        B_array: np.ndarray, 
        parameters: dict[str, float], 
        modulate: bool, 
        n_jobs: int, 
        blocks_per_core: int,
        show_progress: bool,
        init_check: bool = True
) -> np.ndarray:
    """ 
    Computes the full EDMR simulation spectra over `B_array`. 

    Args: 
        * B_array: np.ndarray 
            * B-field swept over
        * parameters: dict[str, float] 
            * Model parameters
            * see utils::load_params.load_params()
        * modulate: bool 
            * If lockin modulation is active 
        * n_jobs: int 
            * Number of cores
        * show_progress: bool 
            * Displays tqdm progress bar

    Returns: 
        if modulate: 
            the lock-in modulated dI/dB array (same length as B_array)
        else: 
            dI/dB array (same length as B_array) 
    """
    dI = compute_singlet_spectra(
        B_array=B_array, 
        parameters=parameters,
        modulate=modulate, 
        n_jobs=n_jobs,
        blocks_per_core=blocks_per_core,
        show_progress=show_progress, 
        init_check=init_check
    ) 
    return parameters["A"] * dI + parameters["I0"] 















