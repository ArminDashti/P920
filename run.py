import argparse
import dataset
import train_safe_action
import train_actor_critic
import play
import os



def clean_assets_directory():
    folder_path = os.path.join(os.getcwd(), 'assets')
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)


def train_model(args):
    # dataset.load_dataset()
    # dataset.append_synthetic_action()
    # train_dl, test_dl = dataset.create_appended_dataloader(args)
    # train_safe_action.train(train_dl, test_dl, args)
    dataloader = dataset.create_non_appended_dataloader(args)
    train_actor_critic.train(dataloader, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='C:\\Users\\armin\\P920_dataset')
    parser.add_argument('--output_dir', type=str, default='C:\\Users\\armin\\P920_output')
    parser.add_argument('--action_dim', type=int, default=28)
    parser.add_argument('--append_data_size', type=int, default=1000000)
    parser.add_argument('--safe_action_hidden_dim', type=int, default=128)
    parser.add_argument('--actor_hidden_dim', type=int, default=128)
    parser.add_argument('--value_hidden_dim', type=int, default=256)
    parser.add_argument('--max_distance', type=float, default=0.25)
    parser.add_argument('--sa_num_epochs', type=int, default=20)
    parser.add_argument('--state_dim', type=int, default=39)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--safe_action_lr', type=float, default=0.001)
    parser.add_argument('--actor_lr', type=float, default=0.001)
    parser.add_argument('--value_lr', type=float, default=0.001)
    parser.add_argument('--alpha_lr', type=float, default=0.001)
    parser.add_argument('--train_sa', action='store_true')
    parser.add_argument('--train_ac', action='store_true')
    parser.add_argument('--ac_num_epochs', type=int, default=100)
    parser.add_argument('--actor_max_norm', type=float, default=0.0)
    parser.add_argument('--critic_max_norm', type=float, default=0.0)
    parser.add_argument('--alpha_max_norm', type=float, default=0.0)
    parser.add_argument('--clean', action='store_true', help='Clean assets directory before training')
    
    args = parser.parse_args()

    train_model(args)
    play.play(args)


if __name__ == '__main__':
    main()