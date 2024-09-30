import numpy as np
import yaml
import os



def load_configs():
    configs_path = os.path.join(os.getcwd(), 'P920', 'configs.yaml')
    with open(configs_path, 'r', encoding='utf-8') as file:
        configs = yaml.safe_load(file)
    return configs
