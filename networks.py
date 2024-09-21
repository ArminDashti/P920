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


class state_action_MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, final_layer='sigmoid'):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)
    
    
    
def create_MLP(state_dim, action_dim, hidden_dim=128, final_layer='sigmoid', loss_func='BCELoss'):
    model = state_action_MLP(state_dim, action_dim, hidden_dim=128, final_layer='sigmoid')
    model = model.float().to(device)
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(ood_net.parameters(), lr=0.001)
    return model, loss_func, optimizer


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Log standard deviation for the Gaussian distribution

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc3(x)
        std = torch.exp(self.log_std)
        return mean, std
    
    
class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value