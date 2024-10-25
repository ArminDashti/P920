import torch
import torch.nn as nn
from torch.distributions import Normal
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class SafeAction(nn.Module):
    def __init__(self, args):
        super(SafeAction, self).__init__()
        # Extract parameters from args
        state_dim = args.state_dim
        action_dim = args.action_dim
        hidden_dim = args.safe_action_hidden_dim

        # Define layers for processing state and action separately
        self.state_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.action_layer = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Define combined layers
        self.combined_layer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action):
        # Process state and action separately
        state_processed = self.state_layer(state)
        action_processed = self.action_layer(action)

        # Concatenate processed state and action
        combined_input = torch.cat([state_processed, action_processed], dim=-1)

        # Pass through the combined layers
        output = self.combined_layer(combined_input)
        return output


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_limit=1.0):
        super().__init__()
        self.action_limit = action_limit
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

        # Proper initialization
        self.reset_parameters()

        # Dictionary to store gradient norms for each layer
        self.grad_norms = {'net_0': [], 'net_2': [], 'mean': [], 'log_std': []}

        # Register hooks to capture gradient norms
        self.register_hooks()

    def reset_parameters(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.xavier_uniform_(self.log_std.weight)
        nn.init.zeros_(self.log_std.bias)

    def register_hooks(self):
        def hook_fn(module, grad_input, grad_output, layer_name):
            # Calculate the gradient norm and store it
            grad_norm = grad_output[0].norm().item()
            self.grad_norms[layer_name].append(grad_norm)

        # Register hooks for each linear layer in the network
        self.net[0].register_backward_hook(lambda m, g_in, g_out: hook_fn(m, g_in, g_out, 'net_0'))
        self.net[2].register_backward_hook(lambda m, g_in, g_out: hook_fn(m, g_in, g_out, 'net_2'))
        self.mean.register_backward_hook(lambda m, g_in, g_out: hook_fn(m, g_in, g_out, 'mean'))
        self.log_std.register_backward_hook(lambda m, g_in, g_out: hook_fn(m, g_in, g_out, 'log_std'))

    def forward(self, state):
        x = self.net(state)
        mean = torch.clamp(self.mean(x), -0.5, 0.5)
        # mean = self.mean(x)
        # log_std = torch.clamp(self.log_std(x), -10, 10)
        log_std = torch.nn.functional.softplus(self.log_std(x))
        # log_std = self.log_std(x).tanh()
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_limit

        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        # self.plot_grad_norms()
        return action, log_prob

    def plot_grad_norms(self):
        # Plot all gradient norms on the same figure
        plt.figure(figsize=(10, 6))
        for layer_name, norms in self.grad_norms.items():
            plt.plot(norms, label=f'Grad Norm - {layer_name}')
        plt.xlabel('Training Step')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norms During Training for Each Layer')
        plt.legend()
        plt.savefig('C:/users/armin/plots/gradient_norms.png')
        plt.show()



    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1_net = self._build_q_network(state_dim, action_dim)
        self.q2_net = self._build_q_network(state_dim, action_dim)

    def _build_q_network(self, state_dim, action_dim):
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1_net(sa), self.q2_net(sa)