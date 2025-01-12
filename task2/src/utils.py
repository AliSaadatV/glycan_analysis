# Load YAML Configuration
def load_config(config_file):
    import yaml

    with open(config_file, 'r') as file:
        return yaml.safe_load(file)
    

# For reproducibility
def seed_everything(seed: int):
    import random, os
    import numpy as np
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)