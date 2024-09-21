import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import time
import random
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model(model, dataloader):
    model.eval()  # Set model to evaluation mode
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            state = batch['observation'].float().to(device)
            action = batch['action'].float().to(device)
            in_dist = batch['in_dist'].float().unsqueeze(1).to(device)

            pred_in_dist = model(state, action)
            correct_predictions += (pred_in_dist.round() == in_dist).sum().item()
            total_samples += in_dist.size(0)

    accuracy = correct_predictions / total_samples
    return accuracy


def train(model, train_dataloader, test_dataloader, loss_fn, optimizer, num_epochs=5, save_dir=r'C:\Users\armin\Documents\GitHub\P920\net.pth'):
    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = 0
        model.train()
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
            state = batch['observation'].float().to(device)
            action = batch['action'].float().to(device)
            in_dist = batch['in_dist'].float().unsqueeze(1).to(device)
            pred_in_dist = model(state, action)
            loss = loss_fn(pred_in_dist, in_dist)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Time: {elapsed_time:.2f}s")

        test_accuracy = evaluate_model(model, test_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Test Accuracy: {test_accuracy:.2f}")
        
    torch.save(model.state_dict(), save_dir)
    return model