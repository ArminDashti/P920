import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm
import os
import utils
import sys
import numpy as np
import play
import networks
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)


def train(dataloader, args):
    num_epochs = args.ac_num_epochs
    policy = networks.Policy(args).to(device)
    qf1 = networks.Critic(args).to(device)
    qf2 = networks.Critic(args).to(device)
    target_qf1 = networks.Critic(args).to(device)
    target_qf2 = networks.Critic(args).to(device)
    target_qf1.load_state_dict(qf1.state_dict())
    target_qf2.load_state_dict(qf2.state_dict())
    qf_criterion = nn.MSELoss()
    policy_optimizer = optim.Adam(policy.parameters(), lr=args.policy_lr)
    qf1_optimizer = optim.Adam(qf1.parameters(), lr=args.qf_lr)
    qf2_optimizer = optim.Adam(qf2.parameters(), lr=args.qf_lr)
    log_alpha = torch.zeros(1, requires_grad=True, device=device) if args.alpha_tuning else torch.tensor(1.0, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=args.policy_lr) if args.alpha_tuning else None
    tau = args.tau
    target_entropy = -args.action_dim
    epoch_counter = [0]
  
    start_epoch = utils.load_checkpoint(
        args,
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        policy_optimizer,
        qf1_optimizer,
        qf2_optimizer,
        alpha_optimizer,
        log_alpha,
        epoch_counter)

    for epoch in range(start_epoch, num_epochs):
        min_reward = -0.467
        max_reward = 20.053
        policy.train()
        qf1.train()
        qf2.train()
        epoch_actor_loss = 0.0
        epoch_cql_loss = 0.0
        epoch_alpha_loss = 0.0
        total_policy_grad_norm = 0.0
        total_qf_grad_norm = 0.0
        total_alpha_grad_norm = 0.0
        total_log_prob = 0.0
        batch_count = 0
      
        for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch', mininterval=1.0):
            states = batch['observation'].float().to(device)
            rewards = batch['reward'].float().to(device)
            # rewards = (rewards - min_reward) / (max_reward - min_reward)
            next_states = batch['next_observation'].float().to(device)
            actions = batch['action'].float().to(device)
            done = (batch['termination'].float() == 1).to(device) | (batch['truncation'].float().to(device) == 1)
            terminals = torch.where(done, torch.ones_like(rewards).to(device), torch.zeros_like(rewards).to(device)).to(device)
            
            ############## Start Policy Loss Calculation ############
            new_state_actions, log_pi = policy.select_action(states)
          
            if args.alpha_tuning:
                alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_grad_norm = utils.compute_grad_norm([log_alpha])
                total_alpha_grad_norm += alpha_grad_norm
                alpha_optimizer.step()
                # with torch.no_grad():
                #     log_alpha = log_alpha.clamp(-5.0, 5.0)
            else:
                alpha_loss = torch.tensor(0.0)
                alpha = torch.tensor(1.0, device=device)

            qf1_pred = qf1(states, new_state_actions)
            # print(qf1_pred.size())
            qf2_pred = qf2(states, new_state_actions)
            q_batch_state_action = torch.min(qf1_pred, qf2_pred)
            policy_loss = (alpha*log_pi - q_batch_state_action.detach()).mean()
            # print(policy_loss.mean())

            ############## Start Q1 and Q2 Loss Calculation ############
            q1_pred = qf1(states, actions)
            q2_pred = qf2(states, actions)

            new_next_actions, _= policy(next_states)
            # print(new_next_actions.size())
            target_q_values = torch.min(target_qf1(next_states, new_next_actions), target_qf2(next_states, new_next_actions))
            # print(target_q_values.size())

            # next_actions_temp, _ = utils.get_policy_actions(next_states, num_actions=2, network=policy)
            # target_qf1_values = utils.get_tensor_values(next_states, next_actions_temp, network=target_qf1).max(1)[0].view(-1, 1)
            # target_qf2_values = utils.get_tensor_values(next_states, next_actions_temp, network=target_qf2).max(1)[0].view(-1, 1)
            # target_q_values = torch.min(target_qf1_values, target_qf2_values)
            # print(target_q_values.size())
            q_target = args.reward_scale * rewards.unsqueeze(1) + (1. - terminals.unsqueeze(1)) * args.discount * target_q_values
            q_target = q_target.detach()
            # print(q_target.size())
            # sys.exit()
            
            qf1_loss = qf_criterion(q1_pred, q_target)
            
            qf2_loss = qf_criterion(q2_pred, q_target)

            ############## CQL ############
            random_actions_tensor = torch.FloatTensor(q2_pred.shape[0] * args.num_random, actions.shape[-1]).uniform_(-1, 1).to(device)
            curr_actions_tensor, curr_log_pis = utils.get_policy_actions(states, num_actions=args.num_random, network=policy)
            new_curr_actions_tensor, new_log_pis = utils.get_policy_actions(next_states, num_actions=args.num_random, network=policy)
            q1_rand = utils.get_tensor_values(states, random_actions_tensor, network=qf1)
            q2_rand = utils.get_tensor_values(states, random_actions_tensor, network=qf2)
            q1_curr_actions = utils.get_tensor_values(states, curr_actions_tensor, network=qf1)
            q2_curr_actions = utils.get_tensor_values(states, curr_actions_tensor, network=qf2)
            q1_next_actions = utils.get_tensor_values(states, new_curr_actions_tensor, network=qf1)
            q2_next_actions = utils.get_tensor_values(states, new_curr_actions_tensor, network=qf2)
        
            cat_q1 = torch.cat([q1_rand, q1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions], 1)
            cat_q2 = torch.cat([q2_rand, q2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions], 1)

            min_qf1_loss = torch.logsumexp(cat_q1 / args.temp, dim=1).mean() * args.min_q_weight * args.temp
            min_qf2_loss = torch.logsumexp(cat_q2 / args.temp, dim=1).mean() * args.min_q_weight * args.temp

            min_qf1_loss = min_qf1_loss - q1_pred.mean() * args.min_q_weight
            min_qf2_loss = min_qf2_loss - q2_pred.mean() * args.min_q_weight

            qf1_loss = qf1_loss + min_qf1_loss
            qf2_loss = qf2_loss + min_qf2_loss

            qf1_optimizer.zero_grad()
            qf1_loss.backward(retain_graph=True)
            qf1_optimizer.step()

            qf2_optimizer.zero_grad()
            qf2_loss.backward(retain_graph=True)
            qf2_optimizer.step()


            policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=False)
            # torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            total_policy_grad_norm += utils.compute_grad_norm(policy.parameters())
            policy_optimizer.step()

            utils.soft_update(qf1, target_qf1, tau)
            utils.soft_update(qf2, target_qf2, tau)

            epoch_actor_loss += policy_loss.item()
            epoch_cql_loss += (qf1_loss.item() + qf2_loss.item())
            epoch_alpha_loss += alpha_loss.item() if args.alpha_tuning else 0.0
            total_log_prob += log_pi.sum().item()
            
            total_qf_grad_norm += utils.compute_grad_norm(list(qf1.parameters()) + list(qf2.parameters()))
            total_alpha_grad_norm += alpha_grad_norm if args.alpha_tuning else 0.0
            batch_count += 1

        avg_policy_loss = epoch_actor_loss / batch_count
        avg_policy_grad_norm = total_policy_grad_norm / batch_count

        qf1_grad_norm = utils.compute_grad_norm(qf1.parameters())
        qf2_grad_norm = utils.compute_grad_norm(qf2.parameters())

        log_message = (
            f"Epoch {epoch + 1}/{num_epochs}\n"
            f"Policy Loss: {avg_policy_loss:.4f}\n"
            f"qf1 Loss: {qf1_loss.item():.4f}\n"
            f"qf2 Loss: {qf2_loss.item():.4f}\n"
            f"Policy Avg Grad Norm: {avg_policy_grad_norm:.4f}\n"
            f"qf1 Grad Norm: {qf1_grad_norm:.4f}\n"
            f"qf2 Grad Norm: {qf2_grad_norm:.4f}\n"
            f"Sum of Log Probabilities: {total_log_prob:.4f}"
        )
        print(log_message)
        print("=" * 30)

        checkpoint_dir = os.path.join(args.output_dir, 'check_points', f'epoch_{epoch + 1}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'policy_state_dict': policy.state_dict(),
            'qf1_state_dict': qf1.state_dict(),
            'qf2_state_dict': qf2.state_dict(),
            'qf1_target_state_dict': target_qf1.state_dict(),
            'qf2_target_state_dict': target_qf2.state_dict(),
            'policy_optimizer_state_dict': policy_optimizer.state_dict(),
            'qf1_optimizer_state_dict': qf1_optimizer.state_dict(),
            'qf2_optimizer_state_dict': qf2_optimizer.state_dict(),
            'alpha_optimizer_state_dict': alpha_optimizer.state_dict() if args.alpha_tuning else None,
            'log_alpha': log_alpha,
        }, os.path.join(checkpoint_dir, 'checkpoint.pth'))

        log_file_path = os.path.join(checkpoint_dir, 'log.txt')
        with open(log_file_path, 'w') as log_file:
            log_file.write(log_message + '\n')

        # play.play(args)