# import dataset
# import train_safe_action
# import train_actor_critic
# import play
import os

file_path = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(file_path))

def clean():
    folder_path = os.path.join(root_dir, 'assets')
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)

clean()
def train(clean=False):
    train_safe_action.train()
    train_actor_critic.train()


def play():
    play.play()