import time
import torch
import logging
import utils
import networks
from tqdm import tqdm
import os
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def train(dataloder):
    actor_model_dir = os.path.join(os.getcwd(), 'assets', 'actor_model.pth')
    if os.path.exists(actor_model_dir):
        print(f"actor_model.pth is exists in {os.path.join(os.getcwd(), 'assets', 'actor_model.pth')} \n")
        return
    
    configs = utils.load_configs()
    num_epochs = 30
    actor = networks.Actor()
    critic = networks.Critic()
    actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
    critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        actor.train()
        critic.train() if critic else None
        epoch_policy_loss = 0
        epoch_critic_loss = 0
        
        for batch in tqdm(dataloder, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
            state = batch['observation'].float().to(device)
            action = batch['action'].float().to(device)
            reward = batch['reward'].float().to(device)
            next_state = batch['next_observation'].float().to(device)
            # done = batch['done'].float().to(device)
            done = 0

            predicted_mean, predicted_std = actor(state)
            dist = torch.distributions.Normal(predicted_mean, predicted_std)
            predicted_action = dist.sample()
            log_prob = dist.log_prob(predicted_action).sum(dim=-1)

            value = critic(state, predicted_action).squeeze()
            next_value = critic(next_state, predicted_action).squeeze() 
            next_value = next_value * (1 - done)

            target_value = reward + 0.99 * next_value
            critic_loss = torch.nn.MSELoss()(value, target_value.detach())

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            advantage = target_value - value
            actor_loss = -(log_prob * advantage.detach()).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            epoch_policy_loss += actor_loss.item()
            epoch_critic_loss += critic_loss.item()

        avg_policy_loss = epoch_policy_loss / len(dataloder)
        avg_critic_loss = epoch_critic_loss / len(dataloder)

        log_message = f"Epoch {epoch+1}/{num_epochs}, Policy Loss: {avg_policy_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}"
        print(log_message)
        logging.info(log_message)

        separator = "=" * 30
        print(separator)
        logging.info(separator)

        save_dir = os.path.join(os.getcwd(), 'assets', 'actor_model.pth')
        torch.save(actor.state_dict(), save_dir)

        save_dir = os.path.join(os.getcwd(), 'assets', 'critic_model.pth')
        torch.save(critic.state_dict(), save_dir)
