import dataset
import train_safe_action
import train_actor_critic
import play
import os


def clean():
    folder_path = os.path.join(os.getcwd(), 'assets')
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)


def train(clean=False):
    if clean:
        clean()
    dataset.load_dataset()
    dataset.append_synthetic_action()
    train_dl, test_dl = dataset.create_appended_dataloader()
    train_safe_action.train(train_dl, test_dl)
    dataloder = dataset.create_non_appended_dataloader()
    train_actor_critic.train(dataloder)



# train()
# play.play()

