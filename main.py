from experiment import Experiment, Experiment_on_data

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
    parser.add_argument("--v_range",
                        default=[60,100],
                        type=int,
                        nargs="+",
                        help="v range for program data")
    parser.add_argument("--alpha_range",
                        default=[0.5, 0.5],
                        type=float,
                        nargs="+",
                        help="alpha range for program data")
    parser.add_argument("--seed_range",
                        default=[1,10],
                        type=int,
                        nargs="+",
                        help="random seed range for program data")
    parser.add_argument("--ccr_range",
                        default=[1.0,1.0],
                        type=float,
                        nargs="+",
                        help="communication to compute ratio range for program data")
    parser.add_argument("--beta_range",
                        default=[0.25,0.25],
                        type=float,
                        nargs="+",
                        help="beta range for program data")
    parser.add_argument("--comm_range",
                        default=[1000,1000],
                        type=int,
                        nargs="+",
                        help="avergae communication range for program data")

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
    parser.add_argument("--device_net_path",
                        default = "./data/device_networks",
                        help="input directory for device network params")
    parser.add_argument("--op_net_path",
                        default = "./data/op_networks/ndevice_20_ntype_5_speed_0.5_bw_3_delay_200",
                        help= "input directory for operator network params")

    return parser.parse_args()

if __name__ == '__main__':
    # Get user arguments and construct config
    exp_cfg = get_args()
    print(exp_cfg)

    # experiment = Experiment(exp_cfg)
    experiment = Experiment_on_data(exp_cfg)
    experiment.train()
    experiment.test()