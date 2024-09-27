import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
import os
import torch


def load_models():
    pass


def propose_actions(actor, critic, state, num=5):
    pass


def take_action(env, state, in_out_dist, actor, critic):
    in_out_dist.eval()
    actor.eval()
    state = torch.from_numpy(state)
    with torch.no_grad():
        print(state.unsqueeze(0).size())
        action = actor(state.unsqueeze(0))
        state_action = torch.cat([state.unsqueeze(0), action], dim=-1)
        pred_in_out_dist = in_out_dist(state_action)
        pred_in_out_dist = torch.argmax(pred_in_out_dist, dim=1)
        if pred_in_out_dist == 1:
            return action.squeeze(0).cpu().numpy()
        else:
            return pred_in_out_dist
            proposed = propose_actions(actor, critic, state)


def play(in_out_dist, actor, critic):
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
        action = take_action(env, observation, in_out_dist, actor, critic)
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
    
