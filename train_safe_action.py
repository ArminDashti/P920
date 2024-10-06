import time
import torch
import logging
import utils
import networks
from tqdm import tqdm
import os
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(train_dataloader, test_dataloader, args):
    safe_action_dir = os.path.join(args.output_dir, 'state_dicts', 'safe_action_state_dict.pth')
    # if os.path.exists(safe_action_dir):
    #     print(f"safe_action_model.pth is exists \n")
    #     return
    
    model = networks.SafeAction(args)
    loss_func = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), args.safe_action_lr)

    num_epochs = args.sa_num_epochs
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
            pred_in_dist = model(state, action)
            loss = loss_func(pred_in_dist, in_dist)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # critic_grad_norm += torch.norm(torch.stack([param.grad.norm() for param in critic.parameters() if param.grad is not None])).item()
        
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

        save_dir = os.path.join(args.output_dir, 'state_dicts', 'safe_action_state_dict.pth')
        torch.save(model.state_dict(), save_dir)
    return model


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
            pred_in_dist = model(state, action)
            correct_predictions += (pred_in_dist.round() == in_dist).sum().item()
            total_samples += in_dist.size(0)

    accuracy = correct_predictions / total_samples
    return accuracy, correct_predictions