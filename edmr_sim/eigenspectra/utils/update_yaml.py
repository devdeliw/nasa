from ruamel.yaml import YAML 
from pathlib import Path

def update_B0(yaml_path: Path, B_value: float):
    """
    Loads the `yaml_path`. 
    Updates the "zeeman.B0" key value, 
    Writes back. 

    Ideally for speed, one would update B0 outside after 
    loading the yaml parameters once. 

    This is done when doing fitting. 
    For eigenvalue simulation its fine. 

    """

    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)

    with open(yaml_path, 'r') as f:
        data = yaml.load(f)

    if 'zeeman' not in data:
        raise KeyError("Key `zeeman` not found. Need dict['zeeman']['B0']")

    data['zeeman']['B0'] = float(B_value)
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
