from ruamel.yaml import YAML 
from pathlib import Path 

import logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

PARAMETER_KEYS = {
    "proportion":   ["A", "I0"], 
    "exchange":     ["J"], 
    "hyperfine":    ["Aa1", "Aa2", "Ab1", "Ab2"], 
    "zeeman":       ["g_e", "g_n1", "g_n2"], 
    "zfs":          ["D1", "D2"], 
    "microwave":    ["nu", "omega1"], 
    "sle":          ["k_S", "k_D", "p"], 
    "lockin":       ["B_mod"], 
    "constants":    ["h", "hbar", "mu_B", "mu_N"],
}

def load_params( 
    param_file: Path 
):
    """
    Loads parameters from a YAML file into a flattened dictionary. 
    
    """
    try: 
        with open(param_file, 'r', encoding="utf-8") as f: 
            raw = YAML().load(f) 
    except FileNotFoundError as e: 
        logger.error("Parameter File Not Found")
        raise e 

    def flatten(tree): 
        out: dict[str, float] = {} 
        for k, v in tree.items(): 
            if isinstance(v, dict): 
                out.update(flatten(v)) 
            else: 
                out[k] = float(v) 
        return out 

    def verify_key(raw: dict, key: str): 
        if key in raw: 
            return raw[key] 
        else:
            from rich import print 

            logger.error(f"`{key}` not found in params.") 
            logger.error(f"`{param_file}` should have: ")
            print(PARAMETER_KEYS)
            raise KeyError(f"`{key}` not found in params.")

    parameters = flatten( 
        {
            **verify_key(raw, "proportion"),
            **verify_key(raw, "exchange"), 
            **verify_key(raw, "hyperfine"), 
            **verify_key(raw, "zeeman"), 
            **verify_key(raw, "zfs"), 
            **verify_key(raw, "microwave"), 
            **verify_key(raw, "sle"), 
            **verify_key(raw, "lockin"), 
            **verify_key(raw, "constants"), 
        }
    )
    return parameters

if __name__ == "__main__": 
    from rich import print

    parameters = load_params(
        param_file = Path("./utils/params.yaml")
    )
    print(parameters)
