import torch
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import logging


def evaluate_model(model, dataloader):
    model.eval()  # Set model to evaluation mode
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
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def train_OOD(model, train_dataloader, test_dataloader, loss_fn, optimizer, num_epochs=5, save_dir=r'C:\Users\armin\Documents\GitHub\P920\net.pth'):
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
        
    torch.save(model.state_dict(), save_dir)


def train():
    OOD_model = train_OOD()
    