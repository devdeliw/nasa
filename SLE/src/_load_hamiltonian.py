import pickle 
from pathlib import Path

PICKLE_PATH = Path.home() / "nasa/hamiltonian/src/pickle/"

def _load_hamiltonian(hamil_name: str): 
    try: 
        with open(PICKLE_PATH / f"{hamil_name}.pickle", "rb") as f:
            hamiltonian = pickle.load(f) 
            return hamiltonian
    except FileNotFoundError as e:
        print(f"{hamil_name}.pickle not found in {PICKLE_PATH}")
        raise e 

def _load_spin(): 
    try: 
        with open(PICKLE_PATH / "spin_hamiltonian.pickle", "rb") as f: 
            spin_hamiltonian = pickle.load(f)
            return spin_hamiltonian
    except FileNotFoundError as e: 
        print(f"`spin_hamiltonian.pickle` not found in {PICKLE_PATH}")
        raise e 



