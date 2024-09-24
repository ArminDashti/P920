import numpy as np
import yaml
import os


def load_configs():
    current_dir = os.getcwd()
    yaml_file_path = os.path.join(current_dir, 'config.yaml')
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
