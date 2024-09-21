import pickle
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import time
import random
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dataset():
    with open(r'C:\Users\armin\ubuntu\rl.pkl', 'rb') as file:
        dataset = pickle.load(file)

    state_dim = len(dataset[0][0]['observation'])
    action_dim = len(dataset[0][0]['action'])
    print('State Dimension:', state_dim)
    print('Action Dimension:', action_dim)

    flat_actions = [action for episode in dataset for step in episode for action in step['action']]
    min_action = min(flat_actions)
    max_action = max(flat_actions)
    print('Min Action:', min_action)
    print('Max Action:', max_action)

    flat_rewards = [step['reward'].tolist() for episode in dataset for step in episode]
    min_reward = min(flat_rewards)
    max_reward = max(flat_rewards)
    print('Min Reward:', min_reward)
    print('Max Reward:', max_reward)

    dataset = [[{**step, 'in_dist': 1} for step in episode] for episode in dataset]
    flat_dataset = [step for episode in dataset for step in episode]
    dataset_info = {
        'state_dim':state_dim,
        'action_dim':action_dim,
        'min_action':min_action,
        'max_action':max_action,
        'min_reward':min_reward,
        'max_reward':max_reward,
        'flat_actions':flat_actions,
        'flat_rewards':flat_rewards,
    }
    return flat_dataset, dataset_info


def append_synthetic_action(dataset, action_dim, number=500000):
    for _ in range(number):
        random_step = random.choice(dataset)
        observation = random_step['observation']
        probability = np.random.rand()

        if probability < 0.25:
            synthetic_action = np.random.uniform(low=-1.0, high=1.0, size=action_dim)
        elif probability < 0.75:
            synthetic_action = random.choice(dataset)['action']
        else:
            base_action = random.choice(dataset)['action']
            perturbation = np.random.normal(loc=0, scale=0.1, size=action_dim)
            synthetic_action = base_action + perturbation
            synthetic_action = np.clip(synthetic_action, -1.0, 1.0)
        
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
