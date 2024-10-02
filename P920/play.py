import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
import os
import networks
import torch


def load_models():
    safe_action = networks.SafeAction()
    state_dict = torch.load(os.path.join(os.getcwd(), 'assets', 'safe_action_model.pth'))
    safe_action.load_state_dict(state_dict)
    safe_action.eval()

    actor = networks.Actor()
    state_dict = torch.load(os.path.join(os.getcwd(), 'assets', 'actor_model.pth'))
    actor.load_state_dict(state_dict)
    actor.eval()

    critic = networks.Critic()
    state_dict = torch.load(os.path.join(os.getcwd(), 'assets', 'critic_model.pth'))
    critic.load_state_dict(state_dict)
    critic.eval()

    return safe_action, actor, critic


def normalize_vector(vector):
    min_val = np.min(vector)
    max_val = np.max(vector)
    return (vector - min_val) / (max_val - min_val) if max_val != min_val else vector

def propose_actions(safe_action, actor, critic, state, num=5):
    for i in range(0,2000000):
        mean, std = actor(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        # action = (torch.rand(1, 28) - 0.5) * 2
        is_safe_action = safe_action(state, action, None, None)
        is_safe_action = torch.argmax(is_safe_action, dim=1)
        if is_safe_action:
            print("Goooooooooooooood")
            return action
    return action

def take_action(safe_action, actor, critic, observation):
    state = torch.from_numpy(observation).float().unsqueeze(0)
    with torch.no_grad():
        mean, std = actor(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        is_safe_action = safe_action(state, action, None, None)
        is_safe_action = torch.argmax(is_safe_action, dim=1)
        if is_safe_action == 1:
            print('safe_Action')
            return action.squeeze(0).cpu().numpy()
        else:
            # return action.squeeze(0).cpu().numpy()
            print('NOT safe_Action')
            action = propose_actions(safe_action, actor, critic, state, num=5)
            return action.squeeze(0).cpu().numpy()


def play():
    safe_action, actor, critic = load_models()
    env = gym.make('AdroitHandDoor-v1', max_episode_steps=400, render_mode='rgb_array')
    video_dir = 'C:\\users\\armin\\v\\'
    os.makedirs(video_dir, exist_ok=True)
    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda episode_id: True,
        name_prefix="AdroitHandDoor")
    
    observation, info = env.reset()
    total_reward = 0
    
    for time_step in range(400):
        observation = normalize_vector(observation)
        action = take_action(safe_action, actor, critic, observation)
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if done or truncated:
            break

    print('total_reward', total_reward)
    return time_step, total_reward



def run():
    in_out_dist, actor = load_models()
    actor.eval()
    in_out_dist.eval()
    
