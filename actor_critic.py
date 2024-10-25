import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from networks import Actor, Critic, SafeAction
from utils import load_dataset, soft_update, normalize_rewards
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_checkpoint(checkpoint_path, model):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])


def update_critic(critic, critic_target, actor, state_batch, action_batch, reward_batch, next_state_batch, done_batch, args, safe_action=None):
    with torch.no_grad():
        next_action, next_log_prob = actor.sample(next_state_batch)



        target_q1, target_q2 = critic_target(next_state_batch, next_action)
        target_q = torch.min(target_q1, target_q2) - args.alpha * next_log_prob
        target_q = reward_batch + (1 - done_batch) * args.gamma * target_q



    current_q1, current_q2 = critic(state_batch, action_batch)
    critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
    return critic_loss, next_log_prob.sum().item(), target_q.sum().item()


def update_actor(actor, critic, state_batch, action_batch, args, safe_action=None):
    action_new, log_prob_new = actor.sample(state_batch)
    if torch.rand(1).item() < args.safe_action_weight:
        safe_action_output = safe_action(state_batch, action_new).round()
        log_prob_new *= safe_action_output

    
    q1_new, q2_new = critic(state_batch, action_new)
    q_new = torch.min(q1_new, q2_new)

    bc_loss = nn.MSELoss()(action_new, action_batch)
    # Set postive log_prob_new but before it training again
    actor_loss = (args.alpha * (-log_prob_new+0.1) - q_new).mean() + 0.1 * bc_loss
    
    return actor_loss, log_prob_new.sum().item(), q_new.sum().item()


def train(actor, critic, critic_target, safe_action, actor_optimizer, critic_optimizer, dataloader, args):
    for epoch in range(args.actor_critic_num_epochs):
        total_critic_grad_norm = 0
        total_actor_grad_norm = 0
        total_log_prob_new = 0
        total_q_new = 0
        total_next_log_prob = 0
        total_target_q = 0
        num_batches = 0

        for state_batch, action_batch, reward_batch, next_state_batch, done_batch in tqdm(dataloader, desc='Training Batches', leave=True):
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = (
                state_batch.to(device),
                action_batch.to(device),
                reward_batch.to(device),
                next_state_batch.to(device),
                done_batch.to(device))

            reward_batch = normalize_rewards(reward_batch)


            critic_loss, next_log_prob_sum, target_q_sum_batch = update_critic(critic, critic_target, actor, state_batch, action_batch, reward_batch, next_state_batch, done_batch, args, safe_action)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            if args.critic_max_norm > 0:
                total_critic_grad_norm += torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=args.critic_max_norm)
            else:
                total_critic_grad_norm += torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=float('inf'))
            critic_optimizer.step()
            total_next_log_prob += next_log_prob_sum
            total_target_q += target_q_sum_batch
            num_batches += 1

            actor_loss, log_prob_new_sum, q_new_sum = update_actor(
                actor, critic, state_batch, action_batch, args, safe_action)
            actor_optimizer.zero_grad()
            actor_loss.backward()
            if args.actor_max_norm > 0:
                total_actor_grad_norm += torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=args.actor_max_norm)
            else:
                total_actor_grad_norm += torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=float('inf'))
            actor_optimizer.step()

            total_log_prob_new += log_prob_new_sum
            total_q_new += q_new_sum

            soft_update(critic, critic_target, args.tau)

        avg_critic_grad_norm = total_critic_grad_norm / num_batches
        avg_actor_grad_norm = total_actor_grad_norm / num_batches

        # Print metrics
        print(
            f"Epoch {epoch}, Critic Loss: {critic_loss.item()},\n"
            f"Actor Loss: {actor_loss.item()},\n"
            f"Avg Critic Grad Norm: {avg_critic_grad_norm},\n"
            f"Avg Actor Grad Norm: {avg_actor_grad_norm},\n"
            f"Log Prob New in Actor: {total_log_prob_new},\n"
            f"Q New in Actor: {total_q_new},\n"
            f"Next Log Prob in Critic: {total_next_log_prob},\n"
            f"Target Q in Critic: {total_target_q}"
        )

def run(args):
    actor = Actor(args.state_dim, args.action_dim, 1.0).to(device)
    critic = Critic(args.state_dim, args.action_dim).to(device)
    critic_target = Critic(args.state_dim, args.action_dim).to(device)
    safe_action = SafeAction(args).to(device)

    load_checkpoint(
        os.path.join(args.outputs_dir, 'check_points', 'safe_action', 'latest_checkpoint.pth'),
        safe_action)
    
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.qf_lr)

    dataset = load_dataset(
        os.path.join(args.outputs_dir, 'datasets', 'actor_critic_datasets.pkl')
    )
    dataloader = DataLoader(dataset, batch_size=args.actor_critic_bs, num_workers=args.actor_critic_nw, shuffle=True)

    train(actor, critic, critic_target, safe_action, actor_optimizer, critic_optimizer, dataloader, args)
