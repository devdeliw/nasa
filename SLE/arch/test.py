from sle_numeric_solver import SLE_NUMERIC
from utils._load_hamiltonian import _load_spin
from density_functional import load_base_params
import numpy as np

H = _load_spin()
sle = SLE_NUMERIC(H, verbose=False)
rho_fn = sle.make_density_func()

base = load_base_params()
base['B0'] = 0.0
rho = rho_fn(**base)

diff = rho - rho.conj().T
max_asym = np.max(np.abs(diff))
print(f"Immediate test: max|rho - rho^†| = {max_asym:.3e}")