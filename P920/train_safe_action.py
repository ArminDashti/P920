import time
import torch
import logging
from tqdm import tqdm
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, 
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
    torch.save(model.state_dict(), r"C:\Users\armin\Documents\GitHub\P920\assets\InOutDist.pth")
    return model

