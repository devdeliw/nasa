# This code was written by Deval Deliwala 
# NASA Glenn Research Center 

import yaml 
import numpy as np
from scipy import linalg
from pathlib import Path 
from .hamil import Hamiltonian 

import logging 
logging.basicConfig(level=logging.INFO)

PARAMETER_FILE = Path(__file__).parent / "params.yaml"

class Eigensolver(Hamiltonian): 
    """ 
    Computes the spectral decomposition of a Hamiltonian and saves: 
    * eigenvalues:              - 1D np.array w 
    * diagonal matrix D:        - D = diag(w)
    * eigenvector matrix P:     - columns are eigenvectors in original basis. 

    """

    _BASIS_MAP = {
        0:  '|1,1>|↑,↑>',
        1:  '|1,1>|↑,↓>', 
        2:  '|1,1>|↓,↑>', 
        3:  '|1,1>|↓,↓>', 
        4:  '|1,0>|↑,↑>', 
        5:  '|1,0>|↑,↓>', 
        6:  '|1,0>|↓,↑>', 
        7:  '|1,0>|↓,↓>', 
        8:  '|0,0>|↑,↑>', 
        9:  '|0,0>|↑,↓>', 
        10: '|0,0>|↓,↑>',
        11: '|0,0>|↓,↓>', 
        12: '|1,-1>|↑,↑>', 
        13: '|1,-1>|↑,↓>', 
        14: '|1,-1>|↓,↑>', 
        15: '|1,-1>|↓,↓>'
    }

    _PARAM_SECTIONS = {
        # single hamiltonian
        "zeeman_only":                  ["zeeman"],
        "hyperfine_only":               ["hyperfine"],
        "zfs_only":                     ["zfs"],
        "exchange_only":                ["exchange"],
        # two-component hamiltonian
        "zeeman_hyperfine":             ["zeeman", "hyperfine"],
        "zeeman_zfs":                   ["zeeman", "zfs"],
        "zeeman_exchange":              ["zeeman", "exchange"],
        "hyperfine_zfs":                ["hyperfine", "zfs"],
        "hyperfine_exchange":           ["hyperfine", "exchange"],
        "zfs_exchange":                 ["zfs", "exchange"],
        # three-component hamiltonian
        "zeeman_hyperfine_zfs":         ["zeeman", "hyperfine", "zfs"],
        "zeeman_hyperfine_exchange":    ["zeeman", "hyperfine", "exchange"],
        "zeeman_zfs_exchange":          ["zeeman", "zfs", "exchange"],
        "hyperfine_zfs_exchange":       ["hyperfine", "zfs", "exchange"],
        # full spin hamiltonian
        "zeeman_hyperfine_zfs_exchange":["zeeman", "hyperfine", "zfs", "exchange"], 
    }
    
    def __init__(
        self, 
        method_name: str, 
        verbose: bool   = False,
        yaml_path: Path = PARAMETER_FILE
    ):
        super().__init__()
        self.method_name = method_name 
        self.verbose = verbose 

        self.yaml_path = yaml_path
        self.logger = logging.getLogger(__name__)

    def load_params(self) -> dict:
        """ 
        Loads parameters from `self.yaml_path file`. 

        Only stores parameters necessary for hamiltonian 
        combination defined in `method_name`.

        """

        try:
            with self.yaml_path.open() as f: 
                cfg = yaml.safe_load(f)
        except FileNotFoundError: 
            self.logger.error("Param file {self.yaml_path} not found.")
            raise 

        if self.method_name not in self._PARAM_SECTIONS: 
            raise KeyError(f"No PARAM_SECTIONS entry for {self.method_name!r}") 

        params = {} 
        for section in self._PARAM_SECTIONS[self.method_name]: 
            section_data = cfg.get(section, {}) 
            params.update(section_data) 

        if "hyperfine" in self._PARAM_SECTIONS[self.method_name]: 
            # isotropic A 
            A1_iso = cfg["hyperfine"]["A1_iso"] 
            A2_iso = cfg["hyperfine"]["A2_iso"] 

            params.update({
                "Aa1": A1_iso,
                "Ab1": A1_iso,
                "Aa2": A2_iso,
                "Ab2": A2_iso,
            })

            params.pop("A1_iso", None)  
            params.pop("A2_iso", None)
        return params

    def _set_hamiltonian(self) -> None: 
        """
        Builds the numerical hamiltonian combination from 
        `self.method_name`.

        This is done by substituting the parameters from 
        `self.load_params()` 

        """

        params = self.load_params() 
        method = getattr(self, self.method_name)
        self.H = method(**params)

    def _spectral_decomposition(self) -> None:
        """ 
        Calculates eigenvalues and eigenvectors from 
        the calculated numerical hamiltonian. 

        The eigenvectors are in the eigenbasis, which is (usually) 
        NOT the same coupled electronic + nuclear zeeman basis 
        from the original Hamiltonians. 

        """

        assert (self.H is not None), (
            "Hamiltonian not yet built. Call set_hamiltonian()."
        )

        # hermicity 
        if not np.allclose(self.H, self.H.conj().T, atol=1e-10): 
            self.logger.warning("Hamiltonian is not Hermitian.") 

        if self.verbose: 
            self.logger.info("Starting spectral decomposition\n")

        self.w, self.v = linalg.eig(self.H) # type: ignore  

        if self.verbose: 
            self.logger.info("Eigenvalues: %s", self.w) 

    def _log_eigenvectors(self, abs_tol: float = 1e-6) -> None:
        """ 
        Converts eigenvectors to linear combination of |psi> from 
        original basis. 

        Args: 
            * abs_tol: float  
                * States w/ coefficients below this value are not considered 
                * for eigenvector decomposition in the original basis. 

        E.g., |eigenvector> = a|1,0>|↓,↑> + b|0,0>||↓,↑> + ...  

        """

        assert self.w is not None and self.v is not None, \
        "Diagonalisation missing. Call _spectral_decomposition()"

        self.labels = []   # for each eigenvector, 
                           # will store the dominant coupled-basis state 
        for idx, (lam, vec) in enumerate(zip(self.w, self.v.T)):
            comps = []
            vmax = np.max(np.abs(vec))
            thresh = abs_tol * vmax

            max_coeff = 0
            max_state = ""
            for j, coeff in enumerate(vec):
                if abs(coeff) > thresh:
                    comps.append(f"{coeff:8.4f}{self._BASIS_MAP[j]}")
                if abs(coeff) > max_coeff: 
                    max_coeff = abs(coeff)
                    max_state = self._BASIS_MAP[j]
            self.labels.append(max_state)

            # display eigendecomposition
            combo = " + ".join(comps) if comps else "0"
            if self.verbose: 
                self.logger.info(
                    f"\nEigenvector {idx:2d} (lambda = {lam:8.3e}):\n  {combo}\n"
                )

    def solve(self) -> "Eigensolver": 
        """ 
        Builds the Hamiltonian, diagonalizes it, 
        log spectrum + eigenvectors, then returns self. 

        """
        self._set_hamiltonian() 
        self._spectral_decomposition() 
        self._log_eigenvectors() 
        return self 


if __name__ == "__main__": 

    # Example usage 
    # ------------- 
 
    solver = Eigensolver( 
        "hyperfine_only",
        verbose=True,  
        yaml_path = Path("./utils/params.yaml")
    ) 

    # Calculates eigenvalues and eigenvectors 
    # for numerical Hyperfine Hamiltonian 
    solver.solve()

        





    

            




