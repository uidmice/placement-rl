from experiment import Experiment_on_data

import argparse, os, json

def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return json.load(open(f, 'rb'))

def get_args():
    parser = argparse.ArgumentParser(description='Placement Experiment Arguments')
    parser.add_argument('--logdir',
                        default='runs',
                        help='exterior log directory')
    parser.add_argument('--logdir_suffix',
                        default='',
                        help='log directory suffix')
    parser.add_argument('--cuda',
                        action='store_true',
                        help='run on CUDA (default: False)')
    parser.add_argument('--noise',
                        default=0,
                        type=float,
                        help='noise level 0-1 (default: 0)')
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        help='learning rate (default: 0.01)')
### Dataset global ###
    parser.add_argument("--device_net_path",
                        default = "./data/device_networks",
                        help="input directory for device network params")

    parser.add_argument("--op_net_path",
                        default = "./data/op_networks",
                        help= "input directory for operator network params")

    parser.add_argument("-p", "--data_parameters",
                        type=validate_file,
                        default='parameters.txt',
                        help="json text file specifying the training/testing dataset parameters", metavar="FILE")

### policy learning ###
    parser.add_argument('--gamma',
                        default=0.97,
                        type=float,
                        help="discount factor gamma in [0,1] (default: 0.97)")

    parser.add_argument('--output_dim',
                        default=10,
                        type=int,
                        help="output dimension for the embedding (default 10)")

    parser.add_argument('--hidden_dim',
                        default=64,
                        type=int,
                        help="hidden dimension for the network (default 64)")

    parser.add_argument('--disable_random_training_pair',
                        action='store_false',
                        dest='random_training_pair',
                        help='disable random selection of training pairs (network, program, init_mapping). Each combination of network/prorgam/mapping will occur for a fixed number of times during training')

    parser.add_argument('--disable_dataset_loading',
                        action='store_false',
                        dest='load_data',
                        help='disable dataset loading (training/testing data will be generated)')

    parser.add_argument('--num_episodes_per_setting',
                        type=int,
                        default=0,
                        help='number of training episodes for each program-network-initial mapping')

    parser.add_argument('--num_of_samples_per_episode',
                        default=100,
                        type=int,
                        help='number of iterations per episode (default 100)')

    parser.add_argument('--num_training_episodes',
                        default=200,
                        type=int,
                        help='total number of training episodes if episode_per_program is not specified (default: 200)')

    parser.add_argument('--memory_capacity',
                        default=4,
                        type=int,
                        help='capacity of the memory buffer for storing placement  (default: 10)')


    parser.add_argument('--use_op_selection',
                        action='store_true',
                        help='use two-step heuristic method (operator selection network + est device selection)')

    parser.add_argument('--eval',
                        action='store_true',
                        help='Evaluates a policy on test dataset every 20 episode')


    parser.add_argument('--num_testing_episodes',
                        default=2,
                        type=int,
                        help='number of testing episodes (default: 2)')

    parser.add_argument(
        '--num_tuning_episodes',
        default=0,
        type=int,
        help='number of episodes for tuning (default: 0)')
    return parser.parse_args()

if __name__ == '__main__':
    # Get user arguments and construct config
    exp_cfg = get_args()
    print(exp_cfg)

    # experiment = Experiment(exp_cfg)
    experiment = Experiment_on_data(exp_cfg)
    experiment.train()
    experiment.test()