import dataset
import networks
import train

dataset_list, dataset_info = dataset.load_dataset()
modified_dataset = dataset.append_synthetic_action(dataset=dataset_list, action_dim=dataset_info['action_dim'])
train_data, test_data = dataset.split_dataset(modified_dataset)
train_dl = dataset.create_dataloader(train_data)
test_dl = dataset.create_dataloader(test_data)

model, loss_func, optimizer = networks.create_MLP()

OOD_detection = train.train(model, train_dl, test_dl, loss_func, optimizer, num_epochs=10)

