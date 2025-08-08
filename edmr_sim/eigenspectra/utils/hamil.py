# This code was written by Deval Deliwala 
# NASA Glenn Research Center 

import pickle 
import numpy as np 
import sympy as sp 

import itertools, inspect
from functools import lru_cache
from pathlib import Path

import logging 
logging.basicConfig(level=logging.INFO)

HAMILTONIAN_FOLDER = Path(__file__).parent.parent.parent / "derive_hamiltonian/pickle/"

class Hamiltonian():
    """
    Constructs numerical spin Hamiltonians by combining any subset 
    of the following interactions:
        * Zeeman, 
        * Hyperfine, 
        * Zero-Field Splitting (ZFS),
        * Exchange Interaction.

    Each type of Hamiltonian requires a specific set of physical 
    constants as input parameters.  For example:  
        * Using `zfs_only()` requires only the ZFS parameters `D1` and `D2`.  
        * Using `zeeman_zfs()` requires parameters for Zeeman and ZFS terms, 
          including D1, D2, B0, g_e, mu_B, g_n1, g_n2, mu_N`.

    The full Spin Hamiltonian is the combination of all four interactions, 
    which can be constructed using methods like `zeeman_hyperfine_zfs_exchange()`.

    Parameters for each interaction must be provided according to the chosen 
    combination to accurately define the Hamiltonian matrix.

    """

    # All constants in Hamiltonian 
    _COMPONENT_SYMBOLS = {
        "zeeman"    : ["B0", "g_e", "mu_B", "g_n1", "g_n2", "mu_N"],
        "hyperfine" : ["Aa1", "Aa2", "Ab1", "Ab2"],   
        "zfs"       : ["D1", "D2"],
        "exchange"  : ["J"],
    }

    def __init__(self, *, 
         zeeman: bool           = True, 
         hyperfine: bool        = True, 
         zfs: bool              = True, 
         exchange: bool         = True,
         template_folder: Path  = HAMILTONIAN_FOLDER
    ):
        self.zeeman     = zeeman  
        self.hyperfine  = hyperfine 
        self.zfs        = zfs 
        self.exchange   = exchange
        self.template_folder = template_folder
        self.logger = logging.getLogger(__name__)

    @lru_cache(maxsize=None)
    def _load(self, stem: str) -> sp.Matrix:
        """
        Helper; loads an individual Hamiltonian defined from "stem" name. 
        16x16.

        """
        try:
            with open(self.template_folder / f"{stem}.pickle", "rb") as f:
                return pickle.load(f)   
        except FileNotFoundError:
            self.logger.error("matrix file %s.pickle not found", stem)
            raise

    def _build_spin(self) -> sp.Matrix:
        """
        Helper; builds the Spin Hamiltonian combination requested from __init__.  

        """
        H = sp.zeros(16, 16)
        if self.zeeman:    H += self._load("zeeman")
        if self.hyperfine: H += self._load("hyperfine")
        if self.zfs:       H += self._load("zfs")
        if self.exchange:  H += self._load("exchange")
        return H

    def hamiltonian(self, *,                 
        B0:   float =0, 
        g_e:  float =0, g_n1: float =0, g_n2: float =0, 
        mu_B: float =0, mu_N: float =0,
        Aa1:  float =0, Aa2:  float =0, 
        Ab1:  float =0, Ab2:  float =0,
        D1:   float =0, D2:   float =0, 
        J:    float =0, 
        dtype=float,
    ) -> np.ndarray:
        """
        Converts the symbolic hamiltonian to a numerical one. 
        Substitutes values for all the components based on
        input parameters. 
        
        """
        H_sym = self._build_spin()
        subs = {
            # Zeeman 
            sp.Symbol("B0")     : B0, 
            sp.Symbol("g_e")    : g_e, 
            sp.Symbol("mu_B")   : mu_B,
            sp.Symbol("g_n1")   : g_n1, 
            sp.Symbol("g_n2")   : g_n2, 
            sp.Symbol("mu_N")   : mu_N,
            # ZFS 
            sp.Symbol("D1")     : D1, 
            sp.Symbol("D2")     : D2,
            # Exchange 
            sp.Symbol("J")      : J,
            # Hyperfine 
            sp.Symbol("Aa1")    : Aa1, 
            sp.Symbol("Aa2")    : Aa2,
            sp.Symbol("Ab1")    : Ab1, 
            sp.Symbol("Ab2")    : Ab2,
        }  

        H_num = H_sym.subs(subs).evalf()
        # Numerical Hamiltonian 
        return np.array(H_num.tolist(), dtype=dtype) 

    @staticmethod
    def _zero_params():
        return dict.fromkeys(
            [
                "B0","g_e","mu_B","g_n1","g_n2","mu_N",
                "D1","D2",
                "J",
                "Aa1","Aa2","Ab1","Ab2",
            ], 
            0.0
        )

""" 

The following functions allow us to build combinations of Hamiltonians 
as an attribute of Hamiltonian() class. For example, 
    * `Hamiltonian.zeeman_exchange(<only required parameters>)` will build 
       the zeeman + exchange Hamiltonian, only asking for the required 
       parameters in the zeeman + exchange Hamiltonian itself. 

Allowed Functions: 
------------------ 
Hamiltonian()
    .zeeman_only 
    .hyperfine_only 
    .zfs_only
    .exchange_only
    
    .zeeman_hyperfine 
    .zeeman_zfs 
    .zeeman_exchange 
    .hyperfine_zfs 
    .hyperfine_exchange 
    .zfs_exchange 

    .zeeman_hyperfine_zfs 
    .zeeman_hyperfine_exchange 
    .zeeman_zfs_exchange 
    .hyperfine_zfs_exchange 

    ~/ The Full Spin Hamiltonian /~
    .zeeman_hyperfine_zfs_exchange 

* These functions are built at runtime 

"""
 
def _make_combo(active):
    name = "_".join(active) + ("_only" if len(active)==1 else "")
    needed = list(itertools.chain.from_iterable(
        Hamiltonian._COMPONENT_SYMBOLS[c] for c in active))

    sig = inspect.Signature([
        inspect.Parameter(p, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for p in needed
    ])

    def method(self, *args, **kw):
        bound = sig.bind(*args, **kw)
        p = self._zero_params()
        p.update(bound.arguments)       
        return self.hamiltonian(**p)

    method.__name__ = name
    method.__doc__ = f"Hamiltonian with {' + '.join(active)} term(s) only."
    return name, method

for r in range(1,5):
    for combo in itertools.combinations(["zeeman","hyperfine","zfs","exchange"], r):
        setattr(Hamiltonian, *_make_combo(combo))

if __name__ == "__main__":

    # Example usage 
    # ------------- 

    # Builds the Hyperfine Hamiltonian with given parameters. 
    H = Hamiltonian().hyperfine_only(          # type: ignore 
        Aa1=1.46e-7, Ab1=1.46e-7, Aa2=2.81e-7, Ab2=2.81e-7,
    )
    print(H)

