## This code was written by Deval Deliwala 
## NASA Glenn Research Center 
## Mentor @ Daniel R. Hart 

import yaml 
import numpy as np
from scipy import linalg
from pathlib import Path 
from hamil import Hamiltonian 

import logging 
logging.basicConfig(level=logging.INFO)

class Eigensolver(Hamiltonian): 
    """ 
    Computes the spectral decomposition of a Hamiltonian and saves: 
    - eigenvalues:              - 1D np.array w 
    - diagonal matrix D:        - D = diag(w)
    - eigenvector matrix P:     - columns are eigenvectors in original basis. 

    """

    _BASIS_MAP = {
        0: "|1,  1> ⊗ |↑, ↑>",
        1: "|1,  0> ⊗ |↑, ↑>",
        2: "|0,  0> ⊗ |↑, ↑>",
        3: "|1, -1> ⊗ |↑, ↑>",
        4: "|1,  1> ⊗ |↑, ↓>",
        5: "|1,  0> ⊗ |↑, ↓>",
        6: "|0,  0> ⊗ |↑, ↓>",
        7: "|1, -1> ⊗ |↑, ↓>",
        8: "|1,  1> ⊗ |↓, ↑>",
        9: "|1,  0> ⊗ |↓, ↑>",
        10: "|0,  0> ⊗ |↓, ↑>",
        11: "|1, -1> ⊗ |↓, ↑>",
        12: "|1,  1> ⊗ |↓, ↓>",
        13: "|1,  0> ⊗ |↓, ↓>",
        14: "|0,  0> ⊗ |↓, ↓>",
        15: "|1, -1> ⊗ |↓, ↓>",
    }

    _PARAM_SECTIONS = {
        "zeeman_only": ["zeeman"],
        "hyperfine_only": ["hyperfine"],
        "zfs_only": ["zfs"],
        "exchange_only": ["exchange"],
        "zeeman_hyperfine": ["zeeman", "hyperfine"],
        "zeeman_zfs": ["zeeman", "zfs"],
        "zeeman_exchange": ["zeeman", "exchange"],
        "hyperfine_zfs": ["hyperfine", "zfs"],
        "hyperfine_exchange": ["hyperfine", "exchange"],
        "zfs_exchange": ["zfs", "exchange"],
        "zeeman_hyperfine_zfs": ["zeeman", "hyperfine", "zfs"],
        "zeeman_hyperfine_exchange": [
            "zeeman",
            "hyperfine",
            "exchange",
        ],
        "zeeman_zfs_exchange": ["zeeman", "zfs", "exchange"],
        "hyperfine_zfs_exchange": ["hyperfine", "zfs", "exchange"],
        "zeeman_hyperfine_zfs_exchange": [
            "zeeman",
            "hyperfine",
            "zfs",
            "exchange",
        ],
    }
    
    def __init__(self, method_name: str):
        super().__init__()
        self.method_name = method_name 
        self.yaml_path = Path("./params.yaml")
        self.logger = logging.getLogger(__name__)

    def load_params(self) -> dict: 
        with self.yaml_path.open() as f: 
            cfg = yaml.safe_load(f)

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
        params = self.load_params() 
        method = getattr(self, self.method_name)
        self.H = method(**params)

    def _spectral_decomposition(self) -> None:
        """ 
        Calculates eigenvalues and eigenvectors from Hamiltonian. 
        The eigenvectors are in the eigenbasis. 

        """

        assert self.H is not None, "Hamiltonian not yet built. Call set_hamiltonian()."
        H = self.H 

        # Hermicity 
        if not np.allclose(H, H.conj().T, atol=1e-10): 
            self.logger.warning("Hamiltonian is not Hermitian; results may be invalid.") 

        self.logger.info(" Starting spectral decomposition \n")
        self.w, self.v = linalg.eigh(H, driver="evd") 
        self.logger.info(" Eigenvalues: %s", self.w) 

    def _log_eigenvectors(self, abs_tol: float = 1e-6) -> None:
        """ 
        Converts eigenvectors back into original Hamiltonian basis. 
        Saves eigenvectors as linear combination of |psi> from 
        original basis. 

        Args: 
            * abs_tol: float  
            States w/ coefficients below this value are not considered 
            for eigenvector decomposition in the original basis. 
        """

        assert self.w is not None and self.v is not None, "Diagonalisation missing."

        for idx, (lam, vec) in enumerate(zip(self.w, self.v.T)):
            comps = []
            vmax = np.max(np.abs(vec))
            thresh = abs_tol * vmax
            for j, coeff in enumerate(vec):
                if abs(coeff) > thresh:
                    comps.append(f"{coeff:8.4f}{self._BASIS_MAP[j]}")
            combo = " + ".join(comps) if comps else "0"
            self.logger.info(
                f"\nEigenvector {idx:2d} (λ = {lam:8.3e}):\n  {combo}\n"
            )

    def solve(self) -> "Eigensolver": 
        """ 
        Builds the Hamiltonian, diagonalizes it, 
        log spectrum + eigenvectors, then returns *self*. 

        """

        self._set_hamiltonian() 
        self._spectral_decomposition() 
        self._log_eigenvectors() 
        return self 


if __name__ == "__main__": 
    solver = Eigensolver( 
        "zeeman_hyperfine", 
    ) 

    solver.solve()
        





    

            




