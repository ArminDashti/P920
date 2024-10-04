import torch
import torch.nn as nn
import utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SafeAction(nn.Module):
    def __init__(self, args):
        super(SafeAction, self).__init__()
        state_dim = args.state_dim
        action_dim = args.action_dim
        hidden_dim = 128
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        state_dim = args.state_dim
        action_dim = args.action_dim
        hidden_dim = args.actor_hidden_dim
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        mean = torch.tanh(self.mean_layer(x))
        log_std = torch.clamp(self.log_std_layer(x), min=-5, max=2)
        std = torch.exp(log_std) + 1e-6
        return mean, std


class Value(nn.Module):
    def __init__(self, args):
        super(Value, self).__init__()
        state_dim = args.state_dim
        hidden_dim = args.value_hidden_dim
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x