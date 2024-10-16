import torch
import torch.nn as nn
from torch.distributions import Normal
import utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class SafeAction(nn.Module):
    def __init__(self, args):
        super(SafeAction, self).__init__()
        state_dim = args.state_dim
        action_dim = args.action_dim
        hidden_dim = args.safe_action_hidden_dim
        self.latent_state = nn.Linear(state_dim, hidden_dim)
        self.latent_action = nn.Linear(action_dim, hidden_dim)
        self.fc1 = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.fc2 = nn.Linear(2*hidden_dim, 1)


    def forward(self, state, action):
        latent_state = torch.relu(self.latent_state(state))
        latent_action = torch.relu(self.latent_action(action))
        x = torch.cat([latent_state, latent_action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x



def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)



class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        state_dim = args.state_dim
        action_dim = args.action_dim
        hidden_dim = args.actor_hidden_dim
      
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
      
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        self.apply(initialize_weights)


    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        mean = self.mean_layer(x)  # Do not apply tanh here
        log_std = torch.clamp(self.log_std_layer(x), min=-20, max=2)
        std = torch.exp(log_std)
        return mean, std


    def select_action(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        z = dist.rsample()  # Reparameterized sampling
        action = torch.tanh(z)  # Apply Tanh to bound actions between -1 and 1
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        # log_prob = dist.log_prob(z)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob



class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        state_dim = args.state_dim
        action_dim = args.action_dim
        hidden_dim = args.value_hidden_dim
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  # Add LayerNorm after the first layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)  # Add LayerNorm after the second layer
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.apply(initialize_weights)


    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)  # Concatenate state and action along the last dimension
        x = torch.relu(self.ln1(self.fc1(x)))  # Apply LayerNorm after fc1
        x = torch.relu(self.ln2(self.fc2(x)))  # Apply LayerNorm after fc2
        x = self.fc3(x)
        return x

