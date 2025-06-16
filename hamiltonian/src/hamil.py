## This code was written by Deval Deliwala 
## NASA Glenn Research Center 
## Mentor @ Daniel R. Hart 

import pickle 
import numpy as np 
import sympy as sp 

import itertools, inspect
from functools import lru_cache
from pathlib import Path

import logging 
logging.basicConfig(level=logging.INFO)

class Hamiltonian():

    """

    Constructs numerical spin Hamiltonians by combining any subset of the following interactions:
    Zeeman, Hyperfine, Zero-Field Splitting (ZFS), and Exchange Interaction.

    Each type of Hamiltonian requires a specific set of physical constants as input parameters.  
    For example:  
    - Using `zfs_only()` requires only the ZFS parameters `D1` and `D2`.  
    - Using `zeeman_zfs()` requires parameters for Zeeman and ZFS terms, including  
      `D1, D2, B0, g_e, mu_B, g_n1, g_n2, mu_N`.

    The full Spin Hamiltonian is the combination of all four interactions, which can be
    constructed using methods like `zeeman_hyperfine_zfs_exchange()`.

    Parameters for each interaction must be provided according to the chosen combination to
    accurately define the Hamiltonian matrix.

    """

    # All constants in Hamiltonian 
    _COMPONENT_SYMBOLS = {
        "zeeman"  : ["B0", "g_e", "mu_B", "g_n1", "g_n2", "mu_N"],
        "hyperfine": ["hbar", "Aa1", "Aa2", "Ab1", "Ab2"],   
        "zfs"     : ["D1", "D2"],
        "exchange": ["J"],
    }

    def __init__(self, *, 
        zeeman=True, 
        hyperfine=True, 
        zfs=True, 
        exchange=True,
        template_folder=Path("/Users/devaldeliwala/nasa/hamiltonian/pickle")):

        self.zeeman     = zeeman  
        self.hyperfine  = hyperfine 
        self.zfs        = zfs 
        self.exchange   = exchange
        self.template_folder = Path(template_folder)
        self.logger = logging.getLogger(__name__)

    @lru_cache(maxsize=None)
    def _load(self, stem) -> sp.Matrix:
        # Loads an individual Hamiltonian 
        # 16x16 

        try:
            with open(self.template_folder / f"{stem}.pickle", "rb") as f:
                return pickle.load(f)   
        except FileNotFoundError:
            self.logger.error("matrix file %s.pickle not found", stem)
            raise

    def _build_spin(self) -> sp.Matrix:
        # Builds the Spin Hamiltonian Template
        # 16x16

        H = sp.zeros(16, 16)
        if self.zeeman:    H += self._load("zeeman")
        if self.hyperfine: H += self._load("hyperfine")
        if self.zfs:       H += self._load("zfs")
        if self.exchange:  H += self._load("exchange")
        return H

    def hamiltonian(self, *,                 
        B0=0, g_e=0, mu_B=0, g_n1=0, g_n2=0, mu_N=0,
        D1=0, D2=0, J=0,
        hbar=0, Aa1=0, Aa2=0, Ab1=0, Ab2=0,
        dtype=float,
    ) -> np.ndarray:
        # Builds the Numerical Spin Hamiltonian 
        # 16x16

        H_sym = self._build_spin()
        subs = {
            # Zeeman 
            sp.Symbol("B0"):B0, 
            sp.Symbol("g_e"):g_e, 
            sp.Symbol("mu_B"):mu_B,
            sp.Symbol("g_n1"):g_n1, 
            sp.Symbol("g_n2"):g_n2, 
            sp.Symbol("mu_N"):mu_N,

            # ZFS 
            sp.Symbol("D1"):D1, sp.Symbol("D2"):D2,

            # Exchange 
            sp.Symbol("J"):J,

            # Hyperfine 
            sp.Symbol("Aa1"):Aa1, sp.Symbol("Aa2"):Aa2,
            sp.Symbol("Ab1"):Ab1, sp.Symbol("Ab2"):Ab2,

            # Reduced Plancks Constant 
            sp.Symbol("hbar2"):hbar**2,
        }

        H_num = H_sym.subs(subs).evalf() 
        return np.array(H_num.tolist(), dtype=dtype) 

    @staticmethod
    def _zero_params():
        return dict.fromkeys(
            [
                "B0","g_e","mu_B","g_n1","g_n2","mu_N",
                "D1","D2","J","Aa1","Aa2","Ab1","Ab2","hbar"
            ], 
            0.0
        )

""" 

The following functions allow us to build combinations of Hamiltonians 
as an attribute of Hamiltonian() class. For example, 

`Hamiltonian.zeeman_exchange(<only required parameters>)` will build 
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

    .zeeman_hyperfine_zfs_exchange (the full Spin Hamiltonian) 

"""
 
def _make_combo(active):
    name = "_".join(active) + ("_only" if len(active)==1 else "")
    needed = list(itertools.chain.from_iterable(
        Hamiltonian._COMPONENT_SYMBOLS[c] for c in active))

    sig = inspect.Signature([inspect.Parameter(p, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                             for p in needed])

    def method(self, *args, **kw):
        bound = sig.bind(*args, **kw)
        p = self._zero_params()
        p.update(bound.arguments)       
        return self.hamiltonian(**p)

    method.__signature__ = sig
    method.__name__ = name
    method.__doc__ = f"Hamiltonian with {' + '.join(active)} term(s) only."
    return name, method

for r in range(1,5):
    for combo in itertools.combinations(["zeeman","hyperfine","zfs","exchange"], r):
        setattr(Hamiltonian, *_make_combo(combo))

if __name__ == "__main__":
    H = Hamiltonian().zeeman_only(
        B0=0.35, g_e=2.002319, mu_B=5.788e-5,
        g_n1=5.5857, g_n2=5.5857, mu_N=3.152e-8,
    )
    print(H)

