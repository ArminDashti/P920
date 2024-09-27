# import dataset
# import train_safe_action
# import train_actor_critic
# import play
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
    train_safe_action.train()
    train_actor_critic.train()


def play():
    play.play()