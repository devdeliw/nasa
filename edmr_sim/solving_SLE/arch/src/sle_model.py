import pickle, tqdm
import numpy as np, sympy as sp
import psutil 

from pathlib import Path
from functools import lru_cache
from scipy.ndimage import gaussian_filter1d
from sle_solver import SteadyStateSLESolver
from tqdm.contrib.concurrent import process_map
from concurrent.futures import ProcessPoolExecutor, as_completed  

# immutable, wont take from params.yaml
CONST = dict(
    mu_B = 5.7883818e-9,         #  [eV/G]
    mu_N = 3.1524512e-12,        #  [eV/G]
    h    = 4.135667662e-15,      #  [eV s]
    hbar = 6.58211e-16,          #  [eV s]
)

g_e, mu_B, B0       = sp.symbols("g_e mu_B B0")
g_n1,g_n2, mu_N     = sp.symbols("g_n1 g_n2 mu_N")
h, nu, omega1       = sp.symbols("h nu omega1")
Aa1,Aa2,Ab1,Ab2     = sp.symbols("Aa1 Aa2 Ab1 Ab2")
J,D1,D2             = sp.symbols("J D1 D2")
hbar                = sp.symbols("hbar")

# spin operators 
idx = { (1,+1):0, (1,0):1, (0,0):2, (1,-1):3 }
S_plus = sp.zeros(4)
for (s, m),i in idx.items():
    if m < s:
        S_plus[idx[(s, m+1)], i] = sp.sqrt(s*(s+1)-m*(m+1))
S_minus = S_plus.T
Sx_e = (S_plus+S_minus)/2
Sz_e = sp.diag(+1, 0, 0, -1)

I4 = sp.eye(4)
Sx_tot = sp.kronecker_product(Sx_e, I4)
Sz_tot = sp.kronecker_product(Sz_e, I4)

_SPIN = sp.Matrix(
    pickle.load(
        open(
            Path.home()/"nasa/derive_hamiltonian/pickle/spin_hamiltonian.pickle", "rb"
        )
    )
)

H_HF_SEC = sp.Matrix(
    pickle.load(
        open(
            Path.home()/"nasa/derive_hamiltonian/pickle/hyperfine_sec.pickle", "rb"
        )
    )
)

def projection_operators():
    """
    16x16 singlet/triplet projectors for 2-electron x 2-nuclei subspace. 

    """
    up, dn = np.array([1, 0]), np.array([0, 1])

    # electron singlet/triplet (4x4)
    S = (np.kron(up, dn) - np.kron(dn, up)) / np.sqrt(2)
    Lambda_S = np.outer(S, S.conj())
    T_plus  = np.kron(up, up)
    T_zero  = (np.kron(up, dn) + np.kron(dn, up)) / np.sqrt(2)
    T_minus = np.kron(dn, dn)
    Lambda_T = (
        np.outer(T_plus,  T_plus .conj()) +
        np.outer(T_zero,  T_zero .conj()) +
        np.outer(T_minus, T_minus.conj())
    )

    I_nuc = np.eye(4)
    return np.kron(Lambda_S, I_nuc), np.kron(Lambda_T, I_nuc)

# cache solvers 
@lru_cache(maxsize=None)
def _make_solver(sign: int):
    """
        * sign =  0 : static (no microwave)
        * sign = +1 : co-rotating   (+ 1/2 h w1 Sx  – h v Sz  – H_hf_sec)
        * sign = -1 : counter-rot.  (− 1/2 h w1 Sx  – h v Sz  – H_hf_sec)

    """
    H_eff = _SPIN                

    if sign in (-1, +1):
        H_rot_gen = -h*nu*Sz_tot + H_HF_SEC      #  G  = hvSz  +  H_hf_sec
        H_drive   = (h*omega1/2) * Sx_tot   

        H_eff = H_eff - H_rot_gen + sign*H_drive

    Lambda_S, Lambda_T = projection_operators()

    return SteadyStateSLESolver(
        H_sym   = sp.Matrix(H_eff),
        Lambda_S= Lambda_S,
        Lambda_T= Lambda_T,
        Gamma   = sp.Matrix(2*np.eye(16)),
        hbar    = CONST["hbar"],
    )

def _compute_block(args):
    """
    args = (indices, B_block, pvec, subs_str)
       indices  : slice positions in the full B array
       B_block  : np.array of field points for this worker
       pvec     : 17-parameter vector ( [.., k_s, k_d, p] )
       subs_str : {symbol_name : value}  (field-independent params)
    """
    indices, B_block, pvec, subs_str = args

    S_stat  = _make_solver(0)    # full static Hamiltonian
    S_drive = _make_solver(+1)   # co-rotating RWA Hamiltonian

    i_block = np.empty_like(B_block, dtype=float)

    for j, B in enumerate(B_block):
        subsB = {**subs_str, "B0": float(B)}

        # steady-state density matrices
        ρ  = S_stat .rho(k_s=pvec[13], k_d=pvec[14], p=pvec[15], params=subsB)
        ρ += 0.5*S_drive.rho(k_s=pvec[13], k_d=pvec[14], p=pvec[15], params=subsB)

        # current (or singlet yield, etc.)
        i_block[j] = float(np.real_if_close(np.trace(S_stat.Lambda_S @ ρ)))

    return indices, i_block


def __compute_block(args):
    """
    Args: 
        * args = (indices, B_block, pvec, subs_str)
            * indices: the slice positions of this block in the full array
            * B_block: the field values for this block
            * pvec: parameter 17-vector 
            * subs_str: {str: val} for parameters.
    Returns (indices, i_block)

    """
    indices, B_block, pvec, subs_str = args

    S_plus  = _make_solver(+1)
    S_minus = _make_solver(-1)
    S_stat  = _make_solver(0)

    i_block = np.empty_like(B_block, dtype=float)
    for j, B in enumerate(B_block):
        subsB = { **subs_str, "B0": float(B)}
        rho  = S_stat.rho(k_s=pvec[13], k_d=pvec[14], p=pvec[15], params=subsB)

        rho += 0.5 * S_plus .rho(k_s=pvec[13], k_d=pvec[14], p=pvec[15], params=subsB)
        rho += 0.5 * S_minus.rho(k_s=pvec[13], k_d=pvec[14], p=pvec[15], params=subsB)
        i_block[j] = np.real_if_close(np.trace(S_stat.Lambda_S @ rho))

    return indices, i_block

def singlet_spectra(
        B_array: np.ndarray, pvec: np.ndarray, modulate: bool=True, 
        n_jobs = psutil.cpu_count(logical = False), show_progress: bool =True
) -> np.ndarray:
    """
    Args: 
        * B_array (array-like) 
            * b field sweep 
        * pvec (np.ndarray) 
            * (17-vector):
                [
                    J, Aa1, Ab1, Aa2, Ab2,
                    D1, D2,
                    B0, g_e, g_n1, g_n2,
                    nu, omega1,
                    kS, kD, p,
                    B_mod
                ]
    Returns: 
        * if modulate:
            * lock-in demodulated d(singlet pop)/dB array (same length as B_array).
        * else: 
            * d(singlet pop)/dB array (same length as B_array)
    """

    (
        Jv,                             # exchange 
        Aa1v, Ab1v, Aa2v, Ab2v,         # hyperfine 
        D1v, D2v,                       # zfs
        _, ge, gn1, gn2,                # zeeman / B0 dummy; sweep B externally
        nuv, omega1v,                   # drive 
        kS, kD, p_gen,                  # liovillian
        Bmod                            # modulation
    ) = pvec

    # hamiltonian only 
    subs = { 
        J:Jv, Aa1:Aa1v, Ab1:Ab1v, Aa2:Aa2v, Ab2:Ab2v,
        D1:D1v, D2:D2v,
        g_e:ge, g_n1:gn1, g_n2:gn2,
        mu_B:CONST["mu_B"], mu_N:CONST["mu_N"],
        h:CONST["h"], hbar:CONST["hbar"],
        nu:nuv, omega1:omega1v 
    }
    subs_str = { str(sym): val for sym, val in subs.items() }

    N = len(B_array)
    i_rel = np.empty(N, dtype=float)

    idx_blocks = np.array_split(np.arange(N), n_jobs*8)             # type: ignore 
    args = []                                                       # 8 blocks per core
    for indices in idx_blocks:
        B_block = B_array[indices]
        args.append((indices, B_block, pvec, subs_str))

    if n_jobs > 1:                                                  # type: ignore
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(_compute_block, arg) for arg in args]
            with tqdm.tqdm(total=len(futures), disable=not show_progress) as pbar:
                for fut in as_completed(futures):
                    indices, i_block = fut.result()
                    i_rel[indices] = i_block
                    pbar.update(1)
    else:
        indices, i_block = _compute_block(args[0])
        i_rel[indices] = i_block

    def lockin_exact(I_arr, B_arr, B_pp, *, n_phase=256, phase=0.0):
        """
        Ideal first-harmonic lock-in (in-phase, infinite time-constant).

            I_arr   singlet population (same length and ordering as B_arr)
            B_arr   static-field sweep (monotonic, uniform spacing)
            B_pp    peak-to-peak modulation amplitude  [G]
            n_phase discrete phase points per cycle    (resolution)

        Returns X-channel lock-in trace on the same B grid.
        """
        A = B_pp                     # peak amplitude
        θ = np.linspace(0.0, 2*np.pi, n_phase, endpoint=False)
        sinθ = np.sin(θ + phase)           # reference & modulation waveform
        ref  = sinθ
        norm = 2.0 / n_phase

        out = np.empty_like(I_arr, dtype=float)
        for k, B0 in enumerate(B_arr):
            B_inst = B0 + A * sinθ
            I_inst = np.interp(B_inst, B_arr, I_arr,
                               left=I_arr[0], right=I_arr[-1])
            out[k] = norm * np.dot(I_inst, ref)
        return out


    if modulate:
        return lockin_exact(i_rel, B_array, Bmod)
    return  np.gradient(i_rel, B_array, edge_order=2)


def edmr_spectra(
        B_array: np.ndarray, pvec: np.ndarray, modulate: bool = True, 
        n_jobs = psutil.cpu_count(logical=False), show_progress: bool = True,
) -> np.ndarray:
    """
    EDMR spectra is directly proportional to the singlet population. Hence, 
    this function just returns 

    A * <singlet_spectra> + I0, with two new arbitrary params A, I0 that make 
    the first two indices of pvec, totaling a 19-vector.

    Args: 
        * B_array (array-like) 
            * b field sweep 
        * pvec (np.ndarray) 
            * (19-vector) *order matters*:
                [
                    A, I0, 
                    J, Aa1, Ab1, Aa2, Ab2,
                    D1, D2,
                    B, g_e, g_n1, g_n2,
                    nu, omega1,
                    kS, kD, p,
                    B_mod
                ]
    Returns: 
        * if modulate:
            * lock-in demodulated d(singlet pop)/dB array (same length as B_array).
        * else: 
            * d(singlet pop)/dB array (same length as B_array)
    """
    singlet_vec = pvec[2:]
    dI = singlet_spectra(
        B_array=B_array, 
        pvec=singlet_vec, 
        modulate=modulate, 
        n_jobs=n_jobs, 
        show_progress=show_progress
    )
    return pvec[0] * dI + pvec[1]
