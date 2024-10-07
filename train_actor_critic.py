import play
import torch
import logging
import networks
from tqdm import tqdm
import os
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(dataloader, args):
    actor_model_path = os.path.join(os.getcwd(), 'assets', 'actor_model.pth')
    
    if os.path.exists(actor_model_path):
        print(f"actor_model.pth exists at {actor_model_path}\n")
        return

    small_value = 1e-3
    num_epochs = args.ac_num_epochs

    actor = networks.Actor(args).to(device)
    critic1 = networks.Value(args).to(device)
    critic2 = networks.Value(args).to(device)

    critic1_target = networks.Value(args).to(device)
    critic2_target = networks.Value(args).to(device)
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())

    cql_alpha = getattr(args, 'cql_alpha', 0.1)

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()), lr=1e-4, weight_decay=1e-5)

    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)

    gamma = 0.99

    target_entropy = -torch.prod(torch.tensor(args.action_dim).to(device)).item()

    # Check for existing checkpoints and resume training if possible
    checkpoint_dir = os.path.join(args.output_dir, 'check_points')
    if os.path.exists(checkpoint_dir):
        checkpoint_folders = [f for f in os.listdir(checkpoint_dir) if f.startswith('epoch_') and os.path.isdir(os.path.join(checkpoint_dir, f))]
        if checkpoint_folders:
            latest_epoch_folder = max(checkpoint_folders, key=lambda x: int(x.split('_')[1]))
            checkpoint_path = os.path.join(checkpoint_dir, latest_epoch_folder, 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                actor.load_state_dict(checkpoint['actor_state_dict'])
                critic1.load_state_dict(checkpoint['critic1_state_dict'])
                critic2.load_state_dict(checkpoint['critic2_state_dict'])
                actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
                alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
                log_alpha = checkpoint['log_alpha']
                start_epoch = checkpoint['epoch']
                print(f"Resuming training from epoch {start_epoch}\n")

                for param_group in actor_optimizer.param_groups:
                    param_group['lr'] = args.actor_lr
                for param_group in critic_optimizer.param_groups:
                    param_group['lr'] = args.value_lr
                for param_group in alpha_optimizer.param_groups:
                    param_group['lr'] = args.alpha_lr
            else:
                start_epoch = 0
        else:
            start_epoch = 0
    else:
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        actor.train()
        critic1.train()
        critic2.train()

        actor_grad_norm = 0
        critic_grad_norm = 0
        alpha_grad_norm = 0
        epoch_policy_loss = 0
        epoch_critic_loss = 0

        for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch', mininterval=1.0):
            state = batch['observation'].float().to(device)
            reward = batch['reward'].float().to(device)
            
            reward_scale = 1 / 20.0
            reward = reward * reward_scale
            
            next_state = batch['next_observation'].float().to(device)
            action = batch['action'].float().to(device)
            done = torch.zeros_like(reward).to(device)
            done_mask = (batch['termination'].float() == 1) | (batch['truncation'].float() == 1)
            done[done_mask] = 1

            with torch.no_grad():
                next_mean, next_std = actor(next_state)
                next_std = next_std.clamp(min=1e-6)
                next_dist = Normal(next_mean, next_std)
                next_action = next_dist.sample()
                log_prob_next = next_dist.log_prob(next_action).sum(dim=-1)
                next_q1 = critic1_target(next_state, next_action).squeeze()
                next_q2 = critic2_target(next_state, next_action).squeeze()
                next_q = torch.min(next_q1, next_q2) - log_alpha.exp().detach() * log_prob_next
                target_value = reward + gamma * next_q * (1 - done)

            q1 = critic1(state, action).squeeze()
            q2 = critic2(state, action).squeeze()
            critic_loss = F.mse_loss(q1, target_value.detach()) + F.mse_loss(q2, target_value.detach())

            policy_mean, policy_std = actor(state)
            policy_std = policy_std.clamp(min=1e-6)
            policy_dist = Normal(policy_mean, policy_std)
            policy_actions = policy_dist.sample()

            random_actions = torch.empty_like(action).uniform_(-1, 1).to(device)
            q1_rand = critic1(state, random_actions)
            q2_rand = critic2(state, random_actions)
            q1_policy = critic1(state, policy_actions)
            q2_policy = critic2(state, policy_actions)

            q1_cat = torch.cat([q1_rand, q1_policy], dim=0)
            q2_cat = torch.cat([q2_rand, q2_policy], dim=0)

            cql_q1_loss = (torch.logsumexp(q1_cat, dim=0) - q1).mean()
            cql_q2_loss = (torch.logsumexp(q2_cat, dim=0) - q2).mean()
            cql_loss = cql_alpha * (cql_q1_loss + cql_q2_loss)

            total_critic_loss = critic_loss + cql_loss

            critic_optimizer.zero_grad()
            total_critic_loss.backward()

            if args.critic_max_norm != 0.0:
                critic_grad_norm += torch.nn.utils.clip_grad_norm_(list(critic1.parameters()) + list(critic2.parameters()), args.critic_max_norm).item()
            else:
                critic_grad_norm += sum(p.grad.norm().item() for p in list(critic1.parameters()) + list(critic2.parameters()) if p.grad is not None)

            critic_optimizer.step()
            epoch_critic_loss += total_critic_loss.item()

            tau = 0.005
            for param, target_param in zip(critic1.parameters(), critic1_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(critic2.parameters(), critic2_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            predicted_mean, predicted_std = actor(state)
            predicted_std = predicted_std.clamp(min=1e-6)
            dist = Normal(predicted_mean, predicted_std)
            predicted_action = dist.sample()
            log_prob = dist.log_prob(predicted_action).sum(dim=-1)
            q1_actor = critic1(state, predicted_action).squeeze()
            q2_actor = critic2(state, predicted_action).squeeze()
            q_actor = torch.min(q1_actor, q2_actor)

            actor_loss = (log_alpha.exp().detach() * log_prob - q_actor).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()

            if args.actor_max_norm != 0.0:
                actor_grad_norm += torch.nn.utils.clip_grad_norm_(actor.parameters(), args.actor_max_norm).item()
            else:
                actor_grad_norm += sum(p.grad.norm().item() for p in actor.parameters() if p.grad is not None)

            actor_optimizer.step()
            epoch_policy_loss += actor_loss.item()

            alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()

            alpha_optimizer.zero_grad()
            alpha_loss.backward()

            if args.alpha_max_norm != 0.0:
                alpha_grad_norm += torch.nn.utils.clip_grad_norm_([log_alpha], args.alpha_max_norm).item()
            else:
                alpha_grad_norm += log_alpha.grad.abs().item()

            alpha_optimizer.step()

            log_alpha.data.clamp_(-5.0, 5.0)

        avg_policy_loss = epoch_policy_loss / len(dataloader)
        avg_critic_loss = epoch_critic_loss / len(dataloader)
        actor_avg_grad_norm = actor_grad_norm / len(dataloader)
        critic_avg_grad_norm = critic_grad_norm / len(dataloader)
        alpha_avg_grad_norm = alpha_grad_norm / len(dataloader)
        alpha_value = log_alpha.exp().item()

        log_message = (
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Actor Loss: {avg_policy_loss:.4f}, "
            f"Critic Loss: {avg_critic_loss:.4f}, "
            f"Actor Grad Norm: {actor_avg_grad_norm:.4f}, "
            f"Critic Grad Norm: {critic_avg_grad_norm:.4f}, "
            f"Alpha Grad Norm: {alpha_avg_grad_norm:.4f}, "
            f"Alpha Value: {alpha_value:.4f}"
        )
        print(log_message)
        print("=" * 30)

        if alpha_value <= torch.exp(torch.tensor(-5.0)).item() + 1e-2:
            print("Warning: Alpha is approaching the lower bound. Consider adjusting the target entropy or learning rate.")
        if alpha_value >= torch.exp(torch.tensor(5.0)).item() - 1e-2:
            print("Warning: Alpha is approaching the upper bound. Consider adjusting the target entropy or learning rate.")

        checkpoint_dir = os.path.join(args.output_dir, 'check_points', f'epoch_{epoch + 1}')
        os.makedirs(checkpoint_dir, exist_ok=True)

        torch.save(actor.state_dict(), os.path.join(checkpoint_dir, 'actor_state_dict.pth'))
        torch.save(critic1.state_dict(), os.path.join(checkpoint_dir, 'value1_state_dict.pth'))
        torch.save(critic2.state_dict(), os.path.join(checkpoint_dir, 'value2_state_dict.pth'))
        torch.save({
            'epoch': epoch + 1,
            'actor_state_dict': actor.state_dict(),
            'critic1_state_dict': critic1.state_dict(),
            'critic2_state_dict': critic2.state_dict(),
            'actor_optimizer_state_dict': actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': critic_optimizer.state_dict(),
            'alpha_optimizer_state_dict': alpha_optimizer.state_dict(),
            'log_alpha': log_alpha,
        }, os.path.join(checkpoint_dir, 'checkpoint.pth'))

        log_file_path = os.path.join(checkpoint_dir, 'log.txt')
        with open(log_file_path, 'w') as log_file:
            log_file.write(log_message + '\n')

        play.play(args)
