import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class state_action_MLP(nn.Module):
    def __init__(self, state_dim, action_dim, reward_dim, hidden_dim=128, final_layer='sigmoid'):
        super(state_action_MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim + state_dim + action_dim , hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action, prev_state, prev_action):
        x = torch.cat([state, action, prev_state, prev_action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.sigmoid(x)
    

def create_MLP(state_dim=39, action_dim=28, hidden_dim=128, reward_dim = 1, final_layer='sigmoid', loss_func='BCELoss'):
    model = state_action_MLP(state_dim, action_dim, reward_dim, hidden_dim=128, final_layer='sigmoid')
    model = model.float().to(device)
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, loss_func, optimizer


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

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