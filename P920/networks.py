import torch
import torch.nn as nn
import utils
import torch.optim as optim
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SafeAction(nn.Module):
    def __init__(self):
        super(SafeAction, self).__init__()
        configs = utils.load_configs()
        state_dim = int(configs['state_dim'])
        action_dim = int(configs['action_dim'])
        hidden_dim = int(configs['hidden_dim'])
        self.fc1 = nn.Linear(state_dim + action_dim , hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)


    def forward(self, state, action, prev_state, prev_action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.sigmoid(x)


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        configs = utils.load_configs()
        state_dim = int(configs['state_dim'])
        action_dim = int(configs['action_dim'])
        hidden_dim = int(configs['hidden_dim'])
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()


    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        std = torch.exp(log_std)
        return mean, std
    
    
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        configs = utils.load_configs()
        state_dim = int(configs['state_dim'])
        action_dim = int(configs['action_dim'])
        hidden_dim = int(configs['hidden_dim'])
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()


    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value
