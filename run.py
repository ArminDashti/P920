import argparse
import dataset
import safe_action
import actor_critic



def train_model(args):
    # train_safe_action.train(train_dl, test_dl, args)
    dataloader = dataset.create_non_appended_dataloader(args)
    train_actor_critic.train(dataloader, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs_dir', type=str, default='C:\\Users\\armin\\P920_output')
    parser.add_argument('--state_dim', type=int, default=39)
    parser.add_argument('--action_dim', type=int, default=28)
    parser.add_argument('--env', type=str, default='None')
    parser.add_argument('--synthetic_size', type=int, default=1000000)
    parser.add_argument('--safe_action_train_bs', type=int, default=512)
    parser.add_argument('--safe_action_test_bs', type=int, default=128)
    parser.add_argument('--safe_action_hidden_dim', type=int, default=256)
    parser.add_argument('--safe_action_num_epochs', type=int, default=200)
    parser.add_argument('--safe_action_lr', type=float, default=0.0001)
    parser.add_argument('--actor_critic_bs', type=int, default=1024)
    parser.add_argument('--actor_hidden_dim', type=int, default=128)
    parser.add_argument('--critic_hidden_dim', type=int, default=128)
    parser.add_argument('--policy_lr', type=float, default=0.0001)
    parser.add_argument('--safe_action_weight', type=float, default=0.99)
    parser.add_argument('--qf_lr', type=float, default=0.0001)
    parser.add_argument('--actor_critic_nw', type=int, default=0)
    parser.add_argument('--actor_critic_num_epochs', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--actor_max_norm', type=float, default=0.0)
    parser.add_argument('--critic_max_norm', type=float, default=5.0)
    args = parser.parse_args()

    # dataset.run(args)
    # safe_action.train(args)
    actor_critic.run(args)


if __name__ == '__main__':
    main()