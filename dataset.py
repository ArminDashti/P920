import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import time
import random
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dataset()
    with open(r'C:\Users\armin\ubuntu\rl.pkl', 'rb') as file:
        dataset = pickle.load(file)

    state_dim = len(dataset[0][0]['observation'])
    action_dim = len(dataset[0][0]['action'])
    print('State Dimension:', state_dim)
    print('Action Dimension:', action_dim)

    flat_actions = [step['action'].tolist() for episode in dataset for step in episode]
    min_action = min(flat_actions)
    max_action = max(flat_actions)
    print('Min Action:', min_action)
    print('Max Action:', max_action)

    flat_rewards = [step['reward'].tolist() for episode in dataset for step in episode]
    min_reward = min(flat_rewards)
    max_reward = max(flat_rewards)
    print('Min Reward:', min_reward)
    print('Max Reward:', max_reward)

    for episode in dataset:
        for step in episode:
            step['in_dist'] = 1

    flat_dataset = [step for episode in dataset for step in episode]
    return flat_dataset



def append_synthetic_action(dataset, number=500000):
    for _ in range(number):
        random_step = random.choice(dataset)
        observation = random_step['observation']
        synthetic_action = np.random.uniform(low=-1.0, high=1.0, size=action_dim)
        
        synthetic_step = {
            'observation': observation,
            'next_observation': random_step['next_observation'],
            'reward': random_step['reward'],
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
    dataset = np.array(dataset)
    real_data = [step for step in dataset if step['in_dist'] == 1]
    synthetic_data = [step for step in dataset if step['in_dist'] == 0]
    train_real_data = random.sample(real_data, 500000)
    train_synthetic_data = random.sample(synthetic_data, 500000)
    train_data = np.array(train_real_data + train_synthetic_data)
    np.random.shuffle(train_data)
    test_data = random.sample(real_data, 500000)
    train_data, test_data = train_test_split(dataset, test_size=0.25, random_state=42)
    return train_data, test_data


def create_dataloader(dataset, batch_size=256, shuffle=True):
    class CustomDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
        
        def __len__(self):
            return len(self.dataset)
    
        def __getitem__(self, idx):
            return self.dataset[idx]

    custom_dataset = CustomDataset(dataset)
    return DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle)
