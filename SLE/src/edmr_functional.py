# edmr_functional.py

import numpy as np
from density_functional import make_density_solver, load_base_params

def make_edmr_model(
    rho_fn=None,
    P_S=None,
    phys_keys=None
):
    """
    Build and return the model function:
    
        I_model(B_array, A, I0, *phys_vals) -> numpy.ndarray
    
    where:
        * B_array     : 1D array of field values
        * A, I0       : amplitude & offset
        * *phys_vals  : the physical parameters (in order phys_keys)

    Args: 
        * rho_fn 
            * Density Matrix Functinoal 
        * P_S   
            * Singlet Projection Operator 
        * phys_keys 
            * Keys for Physical Parameters 
    """

    if rho_fn is None or P_S is None or phys_keys is None:
        rho_fn, P_S = make_density_solver()
        phys_keys = [k for k in load_base_params().keys() if k != "B0"]
    
    def I_model(B_array, *popt):
        """
        Args: 
            * B_array : [-B0, ..., B0]
            * popt    : [A, I0] + phys_vals in the same order as phys_keys
        """
    
        A, I0, *phys_vals = popt
        phys_params = dict(zip(phys_keys, phys_vals))

        B_array = np.asarray(B_array, dtype=float)
        I_pred  = np.empty_like(B_array)

        for idx, B in enumerate(B_array):
            rho  = rho_fn(B, phys_params=phys_params, verbose=False)
            sing = np.real(np.trace(P_S @ rho))
            I_pred[idx] = A * sing + I0

        return I_pred
    return I_model
