import os
from pathlib import Path
import gymnasium as gym
import minari
import numpy as np
import pickle


def download_dataset(dataset_id='D4RL/door/expert-v2'):
    dataset = minari.load_dataset(dataset_id, True)
    return dataset


def dataset_to_list(dataset):
    episodes = []
    for episode in dataset.iterate_episodes():
        observations = episode.observations
        actions = episode.actions

        # Prepare next and previous observations/actions with padding
        next_observations = np.vstack([observations[1:], np.zeros((1, observations.shape[1]))])
        next_actions = np.vstack([actions[1:], np.zeros((1, actions.shape[1]))])
        previous_observations = np.vstack([np.zeros((1, observations.shape[1])), observations[:-1]])
        previous_actions = np.vstack([np.zeros((1, actions.shape[1])), actions[:-1]])

        episode_steps = []
        for i in range(len(episode.rewards)):
            step = {
                'previous_observation': previous_observations[i],
                'observation': observations[i],
                'next_observation': next_observations[i],
                'previous_action': previous_actions[i],
                'action': actions[i],
                'next_action': next_actions[i],
                'reward': episode.rewards[i],
                'termination': episode.terminations[i],
                'truncation': episode.truncations[i],
                'info': episode.infos['success'][i] if 'success' in episode.infos else None 
            }
            episode_steps.append(step)
        episodes.append(episode_steps)
    return episodes

def get_pickle_path(dataset_id, script_path):
    filename = dataset_id.replace('/', '_') + '.pkl'
    assets_dir = script_path.parent / 'assets'
    assets_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = assets_dir / filename
    return pickle_path


def main():
    dataset_id = 'D4RL/door/expert-v2'
    print(f"Downloading dataset '{dataset_id}'...")
    data = download_dataset(dataset_id)
    
    print("Converting dataset to list...")
    dataset = dataset_to_list(data)
    
    script_path = Path(__file__).resolve()
    
    pickle_path = get_pickle_path(dataset_id, script_path)
    
    print(f"Saving dataset to '{pickle_path}'...")
    with open(pickle_path, 'wb') as file:
        pickle.dump(dataset, file)
    
    print(f"Dataset successfully saved to '{pickle_path}'.")

if __name__ == "__main__":
    main()
