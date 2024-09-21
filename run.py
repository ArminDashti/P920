import dataset
import networks
import train
import numpy as np


dataset_list, dataset_info = dataset.load_dataset()

dist = []
for step in dataset_list:
    observation = step['observation']
    next_observation = step['next_observation']
    euclidean_distance = np.linalg.norm(observation - next_observation)
    dist.append(euclidean_distance)

mean = sum(dist) / len(dist)

modified_dataset = dataset.append_synthetic_action(dataset=dataset_list, action_dim=dataset_info['action_dim'])
train_data, test_data = dataset.split_dataset(modified_dataset)
train_dl = dataset.create_dataloader(train_data)
test_dl = dataset.create_dataloader(test_data)

model, loss_func, optimizer = networks.create_MLP()

train.train(model, train_dl, test_dl, loss_func, optimizer, num_epochs=10)

