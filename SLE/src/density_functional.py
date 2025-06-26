from ruamel.yaml import YAML 
from pathlib import Path
import cloudpickle as pickle 
import numpy as np, time 
from utils._load_hamiltonian import _load_spin
from sle_numeric_solver import SLE_NUMERIC

def load_base_params(
     param_file: Path = Path.home() / "nasa/SLE/src/utils/params.yaml"   
):
    yml = YAML()
    with open(param_file) as f:
        data = yml.load(f)
    params = {}
    for section in data.values():
        params.update(section)
    return params

def make_density_solver(
    hamiltonian=None,
    pickle_path: Path = Path.home()/"nasa/SLE/src/pickle/density_func.pickle",
):

    # load density function
    with open(pickle_path, "rb") as f:
        density_func = pickle.load(f)
   
    if hamiltonian is None:
        hamiltonian = _load_spin()
    
    # singlet projector
    sle = SLE_NUMERIC(hamiltonian, verbose=False)
    P_S = np.asarray(sle.Lambda_S, dtype=complex)

    def rho_fn(B0, phys_params=None, verbose=False):
        if phys_params is None: 
            phys_params = load_base_params()

        phys = phys_params.copy()
        phys["B0"] = B0

        # solve
        t0 = time.time()
        rho = density_func(**phys)
        dt = time.time()-t0
        if verbose: print(f"{B0} done in {dt:.4f}s.")

        # verifying trace=1
        trace = np.trace(rho)
        real = trace.real 
        imag = trace.imag

        if not (np.isclose(real, 1.0, atol=1e-8) and np.isclose(imag, 0.0, atol=1e-8)): 
            raise Exception("Density matrix is not normalized.")
        
        # hermitian
        if not np.allclose(rho, rho.conj().T, atol=1e-8):
            raise Exception("Density matrix is not Hermitian")
        return rho

    return rho_fn, P_S