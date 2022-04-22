from experiment import Experiment

import argparse

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


    ### Training parameters ###
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        help='learning rate (default: 0.01)')

    # Training dataset
    parser.add_argument('--num_devices_training',
                        default=[20],
                        type=int,
                        nargs='+',
                        help='list of number of devices in the network for training (default: [20])')

    parser.add_argument('--num_operators_training',
                        default=[10],
                        type=int,
                        nargs='+',
                        help='list of number of operators in the application graph for training (default: [10])')

    parser.add_argument('--application_graph_seeds_training',
                        default=[0, 1, 2, 3, 4],
                        type=int,
                        nargs='+',
                        help='number of applications per num_operators for training (default: [0-4])')

    parser.add_argument('--init_mapping_seeds_training',
                        default=[0],
                        type=int,
                        nargs='+',
                        help='seeds for determining initial mappings for training (default: [0])')

    # policy
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


    parser.add_argument('--num_episodes_per_setting',
                        type=int,
                        default=0,
                        help='number of training episodes for each program-network-initial mapping')

    parser.add_argument('--max_iterations_per_episode',
                        default=50,
                        type=int,
                        help='max number of iterations per episode (default 50)')

    parser.add_argument('--num_training_episodes',
                        default=1000,
                        type=int,
                        help='total number of training episodes if episode_per_program is not specified (default: 1000)')


    parser.add_argument('--use_op_selection',
                        action='store_true',
                        help='use two-step heuristic method (operator selection network + est device selection)')

    parser.add_argument(
        '--eval',
        action='store_true',
        help='Evaluates a policy on test dataset every 20 episode')



    ### Testing ###
    parser.add_argument('--testing_episodes',
                        default=5,
                        type=int,
                        help='number of testing episodes (default: 5)')

    parser.add_argument(
        '--tuning_spisodes',
        default=0,
        type=int,
        help='number of episodes for tuning (default: 0)')

    #dataset
    parser.add_argument('--num_devices_testing',
                        default=[20],
                        type=int,
                        nargs='+',
                        help='list of number of devices in the network for testing (default: [20])')

    parser.add_argument('--num_operators_testing',
                        default=[10],
                        type=int,
                        nargs='+',
                        help='list of number of operators in the application graph for testing (default: [10])')

    parser.add_argument('--application_graph_seeds_testing',
                        default=[0, 1, 2, 3, 4],
                        type=int,
                        nargs='+',
                        help='number of applications per num_operators for testing (default: [0-4])')

    parser.add_argument('--init_mapping_seeds_testing',
                        default=[0],
                        type=int,
                        nargs='+',
                        help='seeds for determining initial mappings for testing (default: [0])')

    return parser.parse_args()

if __name__ == '__main__':
    # Get user arguments and construct config
    exp_cfg = get_args()
    print(exp_cfg)

    experiment = Experiment(exp_cfg)
    experiment.train()
    experiment.test()