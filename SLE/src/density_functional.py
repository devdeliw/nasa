from ruamel.yaml import YAML 
from pathlib import Path
import cloudpickle as pickle 
import numpy as np, time 

def solve_density(
    param_file  = Path.home() / "nasa/SLE/src/utils/params.yaml", 
    density_file= Path.home() / "nasa/SLE/src/pickle/density_func.pickle"
): 
    yml = YAML()
    yml.indent(mapping=2, sequence=4, offset=2)
    yml.preserve_quotes = True 
    try: 
        with open(param_file, 'r') as f:
            data = yml.load(f)
    except FileNotFoundError as e: 
        raise e

    try: 
        with open(density_file, "rb") as f: 
            density_func = pickle.load(f)
    except FileNotFoundError as e: 
        raise e
    
    params = { 
        key: value
        for section in data.values()    
        for key, value in section.items()
    }

    # solve
    t0 = time.time()
    rho = density_func(**params)
    dt = time.time()-t0
    print(f"Solved in {dt:4f}s. Running checks.")

    # verifying trace=1
    trace = np.trace(rho)
    real = trace.real 
    imag = trace.imag

    if not (np.isclose(real, 1.0, atol=1e-8) and np.isclose(imag, 0.0, atol=1e-8)): 
        raise Exception("Density matrix is not normalized.")
    
    # hermitian
    if not np.allclose(rho, rho.conj().T, atol=1e-8):
        raise Exception("Density matrix is not Hermitian")
    
    # check positivity
    if np.min(np.linalg.eigvalsh(rho.real)) < -1e-8:
        raise Exception("Density matrix has negative eigenvalues")
    
    return rho 
