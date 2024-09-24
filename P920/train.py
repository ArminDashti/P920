import torch
import dataset
import time
from tqdm import tqdm
import logging
import os
import networks
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def evaluate_model(model, dataloader):
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            previous_state = batch['previous_observation'].float().to(device)
            state = batch['observation'].float().to(device)
            action = batch['action'].float().to(device)
            in_dist = batch['in_dist'].float().unsqueeze(1).to(device)
            previous_action = batch['previous_action'].float().to(device)
            pred_in_dist = model(state, action, previous_state, previous_action)
            correct_predictions += (pred_in_dist.round() == in_dist).sum().item()
            total_samples += in_dist.size(0)

    accuracy = correct_predictions / total_samples
    return accuracy, correct_predictions


logging.basicConfig(
    filename='training_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

def train_in_out_dist(model, 
                    train_dataloader, 
                    test_dataloader, 
                    loss_fn, 
                    optimizer, 
                    num_epochs):
    
    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0
        model.train()
        
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
            previous_state = batch['previous_observation'].float().to(device)
            state = batch['observation'].float().to(device)
            previous_action = batch['previous_action'].float().to(device)
            action = batch['action'].float().to(device)
            in_dist = batch['in_dist'].float().unsqueeze(1).to(device)
            pred_in_dist = model(state, action, previous_state, previous_action)
            loss = loss_fn(pred_in_dist, in_dist)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        elapsed_time = time.time() - start_time
        log_message = f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Time: {elapsed_time:.2f}s"
        print(log_message)
        logging.info(log_message)

        test_accuracy, correct_predictions = evaluate_model(model, test_dataloader)
        log_message = f"Epoch {epoch+1}/{num_epochs}, Test Accuracy: {test_accuracy}, Correct Predictions: {correct_predictions:,}"
        print(log_message) 
        logging.info(log_message)

        separator = "=" * 30
        print(separator)
        logging.info(separator) 

    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, 'AwareNet.pth')
    torch.save(model.state_dict(), save_dir)
    return model


def compute_euclidean_distance(dataset_list):
    dist = []
    for step in dataset_list:
        observation = step['observation']
        next_observation = step['next_observation']
        euclidean_distance = np.linalg.norm(observation - next_observation)
        dist.append(euclidean_distance)
    mean = sum(dist) / len(dist)
    return mean




def train_AC(train_dl,
            test_dl,
            actor, 
            actor_optimizer, 
            critic, 
            critic_optimizer, 
            num_epochs, 
            gamma=0.99):
    
    for epoch in range(num_epochs):
        actor.train()
        critic.train() if critic else None
        epoch_policy_loss = 0
        epoch_critic_loss = 0
        
        for batch in tqdm(train_dl, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
            state = batch['observation'].float().to(device)
            action = batch['action'].float().to(device)
            reward = batch['reward'].float().to(device)
            next_state = batch['next_observation'].float().to(device)
            # done = batch['done'].float().to(device)
            done = 0

            predicted_mean, predicted_std = actor(state)
            dist = torch.distributions.Normal(predicted_mean, predicted_std)
            predicted_action = dist.sample()  # Sample an action
            log_prob = dist.log_prob(predicted_action).sum(dim=-1)  # Log probability of the action

            value = critic(state, predicted_action).squeeze()  # V(s)
            next_value = critic(next_state, predicted_action).squeeze()  # V(s')
            next_value = next_value * (1 - done)  # Set next value to zero if done

            target_value = reward + gamma * next_value
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

        avg_policy_loss = epoch_policy_loss / len(train_dl)
        avg_critic_loss = epoch_critic_loss / len(train_dl)
        
        log_message = f"Epoch {epoch+1}/{num_epochs}, Policy Loss: {avg_policy_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}"
        print(log_message)
        logging.info(log_message)

        separator = "=" * 30
        print(separator)
        logging.info(separator)


def filter_dataloader(dataloader):
    filtered_batches = []
    for batch in dataloader:
        in_dist = batch['in_dist'].float()
        non_zero_indices = in_dist != 0
        if non_zero_indices.any():
            filtered_batch = {key: value[non_zero_indices] for key, value in batch.items()}
            filtered_batches.append(filtered_batch)
    return filtered_batches


def create_filtered_dataloader(filtered_batches, batch_size):
    filtered_dataset = []
    for batch in filtered_batches:
        filtered_dataset.append(batch)
    filtered_dl = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True)
    return filtered_dl


def train():
    dataset_list, dataset_info = dataset.load_dataset()
    modified_dataset = dataset.append_synthetic_action(dataset=dataset_list, action_dim=dataset_info['action_dim'])
    train_data, test_data = dataset.split_dataset(modified_dataset)
    train_dl = dataset.create_dataloader(train_data)
    test_dl = dataset.create_dataloader(test_data)

    # in_out_dist = networks.InOutDist(dataset_info['state_dim'], dataset_info['action_dim'], 128)
    # in_out_dist_loss_func = nn.BCELoss()
    # in_out_dist_optimizer = optim.Adam(in_out_dist.parameters(), lr=0.001)
    # in_out_dist = train_in_out_dist(model=in_out_dist,
    #                                   train_dataloader=train_dl,
    #                                   test_dataloader=test_dl,
    #                                   loss_fn=in_out_dist_loss_func,
    #                                   optimizer=in_out_dist_optimizer,
    #                                   num_epochs=1)

    actor = networks.Actor(dataset_info['state_dim'], dataset_info['action_dim'], 128)
    actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)

    critic = networks.Critic(dataset_info['state_dim'], dataset_info['action_dim'], 1288)
    critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

    train_data_filtered = filter_dataloader(train_dl)
    train_dl_filtered = create_filtered_dataloader(train_data_filtered, batch_size=train_dl.batch_size)

    train_AC(train_dl=train_dl,
             test_dl=test_dl,
            actor=actor, 
            actor_optimizer=actor_optimizer, 
            critic=critic, 
            critic_optimizer=critic_optimizer, 
            num_epochs=10)

