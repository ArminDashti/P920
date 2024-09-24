import gymnasium as gym
import torch


def load_models():
    pass





def play():
    in_out_dist, actor = load_models()
    actor.eval()
    in_out_dist.eval()
    
