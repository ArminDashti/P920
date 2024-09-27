import pickle
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import random
from tqdm import tqdm
import yaml
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_dataset():
    dataset_path = os.path.join(os.getcwd(), 'assets', 'dataset', 'D4RL_door_expert-v2.pkl')
    print(f"Loading dataset from \n {dataset_path} \n, please wait ... \n")
    
    with open(dataset_path, 'rb') as file:
        dataset = pickle.load(file)
    print(f"Dataset loaded successfully. \n")
    
    print(f"Gather dataset information. \n")
    state_dim = len(dataset[0][0]['observation'])
    action_dim = len(dataset[0][0]['action'])

    flat_actions = [action for episode in tqdm(dataset, desc="Processing Actions") 
                    for step in episode for action in step['action']]
    min_action = min(flat_actions)
    max_action = max(flat_actions)

    flat_rewards = [step['reward'].tolist() for episode in tqdm(dataset, desc="Processing Rewards") 
                    for step in episode]
    min_reward = min(flat_rewards)
    max_reward = max(flat_rewards)

    dataset = [[{**step, 'in_dist': 1} for step in episode] for episode in dataset]
    
    flat_dataset = [step for episode in dataset for step in episode]
    dataset_info = {
        'state_dim': float(state_dim),
        'action_dim': float(action_dim),
        'min_action': float(min_action),
        'max_action': float(max_action),
        'min_reward': float(min_reward),
        'max_reward': float(max_reward),
    }

    print('Dataset Information: \n', dataset_info, '\n')
    configs_path = os.path.join(os.getcwd(), 'P920', 'configs.yaml')
    with open(configs_path, 'r', encoding='utf-8') as file:
        existing_data = yaml.safe_load(file)
    existing_data.update(dataset_info)
    with open(configs_path, 'w') as file:
        yaml.dump(existing_data, file)

    print('Saving dataset... \n')
    flattened_dataset_path = os.path.join(os.getcwd(), 'assets', 'list_dict_dataset.pkl')
    with open(flattened_dataset_path, 'wb') as f:
        with tqdm(total=len(flat_dataset), desc="Saving", unit="item") as pbar:
            for item in flat_dataset:
                pickle.dump(item, f)
                pbar.update(1)
    print(f'List-Dict dataset successfully stored at \n {flattened_dataset_path} \n')


def append_synthetic_action(dataset, action_dim, number=500000, max_distance=0.25):
    for _ in range(number):
        real_action = None
        synthetic_action = None
        random_step = random.choice(dataset)
        observation = random_step['observation']
        previous_observation = random_step['previous_observation']
        previous_action = random_step['previous_action']
        probability = np.random.rand()

        if probability < 0.25:
            synthetic_action = np.random.uniform(low=-1.0, high=1.0, size=action_dim)
        elif probability < 0.75:
            real_action = random.choice(dataset)['action']
        else:
            base_action = random.choice(dataset)['action']
            synthetic_action = base_action + 0.3 * np.random.randn(*base_action.shape)
            synthetic_action = np.clip(synthetic_action, -1.0, 1.0)

        if synthetic_action is not None:
            distance = np.linalg.norm(synthetic_action - previous_action)
            if distance > max_distance:
                direction = synthetic_action - previous_action
                normalized_direction = direction / np.linalg.norm(direction)
                synthetic_action = previous_action + normalized_direction * max_distance
                synthetic_action = np.clip(synthetic_action, -1.0, 1.0)

        if real_action is not None:
            synthetic_action = real_action
        synthetic_step = {
            'previous_observation': previous_observation,
            'observation': observation,
            'next_observation': random_step['next_observation'],
            'reward': random_step['reward'],
            'previous_action': previous_action,
            'action': synthetic_action,
            'next_action': random_step['next_action'],
            'termination': random_step['termination'],
            'truncation': random_step['truncation'],
            'info': random_step['info'],
            'in_dist': 0
        }
        dataset.append(synthetic_step)

    return dataset



def split_dataset(dataset):
    in_dist_data = [d for d in dataset if d['in_dist'] == 1]
    out_dist_data = [d for d in dataset if d['in_dist'] == 0]
    random.shuffle(in_dist_data)
    random.shuffle(out_dist_data)
    train_in_dist = in_dist_data[:250000]
    test_in_dist = out_dist_data[:250000]
    train_out_dist = out_dist_data[250000:]
    train_data = train_in_dist + train_out_dist
    test_data = test_in_dist
    return train_data, test_data

def normalize_vector(vector):
    min_val = np.min(vector)
    max_val = np.max(vector)
    return (vector - min_val) / (max_val - min_val) if max_val != min_val else vector


def create_dataloader(dataset, batch_size=256, shuffle=True):
    class CustomDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
        
        def __len__(self):
            return len(self.dataset)
    
        def __getitem__(self, idx):
            data_point = self.dataset[idx]
            normalized_previous_observation = normalize_vector(data_point['previous_observation'])
            normalized_observation = normalize_vector(data_point['observation'])
            normalized_next_observation = normalize_vector(data_point['next_observation'])
            normalized_reward = data_point['reward']
            normalized_previous_action = data_point['previous_action']
            normalized_action = data_point['action']
            normalized_next_action = normalize_vector(data_point['next_action'])

            return {
                'previous_observation': normalized_previous_observation,
                'observation': normalized_observation,
                'next_observation': normalized_next_observation,
                'reward': normalized_reward,
                'previous_action': normalized_previous_action,
                'action': normalized_action,
                'next_action': normalized_next_action,
                'termination': data_point['termination'],
                'truncation': data_point['truncation'],
                'info': data_point['info'],
                'in_dist': data_point['in_dist'],
            }

    custom_dataset = CustomDataset(dataset)
    return DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle)


load_dataset()