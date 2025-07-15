import sympy as sp 
import pickle 

from pathlib import Path

def load(fname): 
    with open(fname, "rb") as f: 
        H = pickle.load(f) 
    sp.pretty_print(H)

home = Path.home() 
folder = home / "nasa" / "hamiltonian"
fname = folder / "pickle" / "spin_hamiltonian.pickle"
load(fname)
    
