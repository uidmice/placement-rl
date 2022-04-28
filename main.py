from experiment import Experiment_on_data

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

    parser.add_argument('--num_of_constraint_types',
                        default=5,
                        type=int,
                        help='Number of hardware types for placement constraint')

### Training dataset ###
    # device network
    parser.add_argument('--num_devices_training',
                        default=[20],
                        type=int,
                        nargs='+',
                        help='list of number of devices in the network for training (default: [20])')
    parser.add_argument('--network_seeds_training',
                        default=[0,1],
                        type=int,
                        nargs='+',
                        help='seeds used to generate each network config (default: [0,1,2,3,4])')
    parser.add_argument('--type_constraint_prob_training',
                        default=[0.2],
                        type=float,
                        nargs='+',
                        help='list of hardware type constraint probabilities 0-1 (default: [0.2])')
    parser.add_argument('--compute_speed_training',
                        default=[5],
                        type=int,
                        nargs='+',
                        help='list of average compute speed of devices in the network for training (default: [5])')

    parser.add_argument('--bw_training',
                        default=[100],
                        type=int,
                        nargs='+',
                        help='list of average bandwidth between devices in the network for training (default: [100])')

    parser.add_argument('--delay_training',
                        default=[10],
                        type=int,
                        nargs='+',
                        help='list of average communication delay between devices in the network for training (default: [10])')

    parser.add_argument('--beta_bw_training',
                        default=[0.2],
                        type=float,
                        nargs='+',
                        help='list of bandwidth variation between devices in the network for training [0-1] (default: [0.2])')

    parser.add_argument('--beta_speed_training',
                        default=[0.2],
                        type=float,
                        nargs='+',
                        help='list of compute speed variation among devices in the network for training [0,1] (default: [0.2])')

    # program graph
    parser.add_argument('--vs_training',
                        default=[10, 20, 40],
                        type=int,
                        nargs='+',
                        help='list of number of operators in the application graph for training (default: [10, 20, 30])')

    parser.add_argument('--graph_seeds_training',
                        default=[0, 1, 2],
                        type=int,
                        nargs='+',
                        help='seeds used to generate application graphs for training (default: [0-4])')

    parser.add_argument('--init_mapping_seeds_training',
                        default=[0],
                        type=int,
                        nargs='+',
                        help='seeds for determining initial mappings for training (default: [0])')

    parser.add_argument("--connect_probability_training",
                        default=[0.2],
                        type=float,
                        nargs="+",
                        help="connect probability when generating graphs for training 0-1 (default: [0.2])")
    parser.add_argument("--alphas_training",
                        default=[0.1, 0.5],
                        type=float,
                        nargs="+",
                        help="alphas for generating program graphs for training (default: [0.1, 0.5])")

    parser.add_argument('--computes_training',
                        default=[100],
                        type=int,
                        nargs='+',
                        help='list of average compute of operators in the graph for training (default: [100])')

    parser.add_argument('--bytes_training',
                        default=[100],
                        type=int,
                        nargs='+',
                        help='list of average bytes for data links in the graph for training (default: [100])')

    parser.add_argument('--beta_compute_training',
                        default=[0.2],
                        type=float,
                        nargs='+',
                        help='list of compute variation among operators for training [0-1] (default: [0.2])')

    parser.add_argument('--beta_byte_training',
                        default=[0.2],
                        type=float,
                        nargs='+',
                        help='list of byte variation across edges in the graph for training [0-1] (default: [0.2])')

### Evaluation dataset ###
    # device network
    parser.add_argument('--num_devices_testing',
                        default=[20],
                        type=int,
                        nargs='+',
                        help='list of number of devices in the network for testing (default: [20])')
    parser.add_argument('--network_seeds_testing',
                        default=[20, 21],
                        type=int,
                        nargs='+',
                        help='seeds used to generate each network config (default: [0,1,2,3,4])')
    parser.add_argument('--type_constraint_prob_testing',
                        default=[0.2],
                        type=float,
                        nargs='+',
                        help='list of hardware type constraint probabilities 0-1 (default: [0.2])')
    parser.add_argument('--compute_speed_testing',
                        default=[5],
                        type=int,
                        nargs='+',
                        help='list of average compute speed of devices in the network for testing (default: [5])')

    parser.add_argument('--bw_testing',
                        default=[100],
                        type=int,
                        nargs='+',
                        help='list of average bandwidth between devices in the network for testing (default: [100])')

    parser.add_argument('--delay_testing',
                        default=[10],
                        type=int,
                        nargs='+',
                        help='list of average communication delay between devices in the network for testing (default: [10])')

    parser.add_argument('--beta_bw_testing',
                        default=[0.2],
                        type=float,
                        nargs='+',
                        help='list of bandwidth variation between devices in the network for testing [0-1] (default: [0.2])')

    parser.add_argument('--beta_speed_testing',
                        default=[0.2],
                        type=float,
                        nargs='+',
                        help='list of compute speed variation among devices in the network for testing [0,1] (default: [0.2])')

    # program graph
    parser.add_argument('--vs_testing',
                        default=[20],
                        type=int,
                        nargs='+',
                        help='list of number of operators in the application graph for testing (default: [10, 20, 30])')

    parser.add_argument('--graph_seeds_testing',
                        default=[20],
                        type=int,
                        nargs='+',
                        help='seeds used to generate application graphs for testing (default: [0-4])')

    parser.add_argument('--init_mapping_seeds_testing',
                        default=[20],
                        type=int,
                        nargs='+',
                        help='seeds for determining initial mappings for testing (default: [0])')

    parser.add_argument("--connect_probability_testing",
                        default=[0.2],
                        type=float,
                        nargs="+",
                        help="connect probability when generating graphs for testing 0-1 (default: [0.2])")
    parser.add_argument("--alphas_testing",
                        default=[0.1, 0.5],
                        type=float,
                        nargs="+",
                        help="alphas for generating program graphs for testing (default: [0.1, 0.5])")

    parser.add_argument('--computes_testing',
                        default=[100],
                        type=int,
                        nargs='+',
                        help='list of average compute of operators in the graph for testing (default: [100])')

    parser.add_argument('--bytes_testing',
                        default=[100],
                        type=int,
                        nargs='+',
                        help='list of average bytes for data links in the graph for testing (default: [100])')

    parser.add_argument('--beta_compute_testing',
                        default=[0.2],
                        type=float,
                        nargs='+',
                        help='list of compute variation among operators for testing [0-1] (default: [0.2])')

    parser.add_argument('--beta_byte_testing',
                        default=[0.2],
                        type=float,
                        nargs='+',
                        help='list of byte variation across edges in the graph for testing [0-1] (default: [0.2])')

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

    parser.add_argument('--random_training_pair',
                        action='store_true',
                        help='randomly select training pairs (network, program, init_mapping)')

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