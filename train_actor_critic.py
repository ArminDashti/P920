import play
import torch
import logging
import networks
from tqdm import tqdm
import os
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def train(dataloader, args):
    actor_model_path = os.path.join(os.getcwd(), 'assets', 'actor_model.pth')
    
    if os.path.exists(actor_model_path):
        print(f"actor_model.pth exists at {actor_model_path}\n")
        return
        
    small_value = 1e-3
    num_epochs = args.ac_num_epochs
    actor = networks.Actor(args).to(device)
    critic = networks.Value(args).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.value_lr)

    for epoch in range(num_epochs):
        actor.train()
        critic.train()

        actor_grad_norm = 0
        critic_grad_norm = 0
        epoch_policy_loss = 0
        epoch_critic_loss = 0
        batch_number = 0

        for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch', mininterval=1.0):
            batch_number += 1
            state = batch['observation'].float().to(device)
            reward = batch['reward'].float().to(device)
            reward = reward / 20.0
            next_state = batch['next_observation'].float().to(device)
            done = torch.zeros_like(reward).to(device)
            done_mask = (batch['termination'].float() == 1) | (batch['truncation'].float() == 1)
            done[done_mask] = 1

            predicted_mean, predicted_std = actor(state)
            dist = torch.distributions.Normal(predicted_mean, predicted_std)
            predicted_action = dist.sample()
            log_prob = dist.log_prob(predicted_action).sum(dim=-1)

            value = critic(state).squeeze()
            next_value = critic(next_state).squeeze()
            next_value = next_value * (1 - done)

            target_value = reward + 0.99 * next_value
            critic_loss = torch.nn.MSELoss()(value, target_value.detach())

            critic_optimizer.zero_grad()
            critic_loss.backward()
            # torch.nn.utils.clip_grad_norm_(critic.parameters(), args.critic_max_norm)

            critic_grad_norm += torch.norm(torch.stack([param.grad.norm() for param in critic.parameters() if param.grad is not None])).item()

            critic_optimizer.step()
            epoch_critic_loss += critic_loss.item()

            advantage = (target_value - value).detach()
            weight = torch.exp(advantage / 1)
            actor_loss = -(log_prob * weight).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(actor.parameters(), args.actor_max_norm)

            actor_grad_norm += torch.norm(torch.stack([param.grad.norm() for param in actor.parameters() if param.grad is not None])).item()

            actor_optimizer.step()
            epoch_policy_loss += actor_loss.item()

        avg_policy_loss = epoch_policy_loss / len(dataloader)
        avg_critic_loss = epoch_critic_loss / len(dataloader)
        actor_avg_grad_norm = actor_grad_norm / len(dataloader)
        critic_avg_grad_norm = critic_grad_norm / len(dataloader)

        log_message = (
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Actor Loss: {avg_policy_loss:.4f}, "
            f"Critic Loss: {avg_critic_loss:.4f}, "
            f"Actor Average Gradient Norm: {actor_avg_grad_norm:.4f}, "
            f"Critic Average Gradient Norm: {critic_avg_grad_norm:.4f}"
        )
        print(log_message)

        separator = "=" * 30
        print(separator)

        actor_save_path = os.path.join(args.output_dir, 'state_dicts', f'actor_state_dict.pth')
        torch.save(actor.state_dict(), actor_save_path)

        critic_save_path = os.path.join(args.output_dir, 'state_dicts', f'value_state_dict.pth')
        torch.save(critic.state_dict(), critic_save_path)

        play.play(args)