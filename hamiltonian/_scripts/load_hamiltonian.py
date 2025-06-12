import sympy as sp 
import pickle 

def load(fname): 
    with open(fname, "rb") as f: 
        H = pickle.load(f) 
    sp.pretty_print(H)


load("/Users/devaldeliwala/nasa/hamiltonian/spin_hamiltonian.pickle")
    
