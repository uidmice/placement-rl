from experiment import Experiment_on_data

import argparse, os, json

def validate_file(f):
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return json.load(open(f, 'rb'))

def validate_dir(f):
    if not os.path.isdir(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

def get_args():
    parser = argparse.ArgumentParser(description='Placement Experiment Arguments')
    parser.add_argument('--logdir',
                        default='runs',
                        help='exterior log directory')
    parser.add_argument('--logdir_suffix',
                        default='',
                        help='log directory suffix')
    parser.add_argument('--disable_cuda',
                        action='store_false',
                        dest='cuda',
                        help='disable running on CUDA (default True is available)')
    parser.add_argument('--noise',
                        default=0,
                        type=float,
                        help='noise level 0-1 (default: 0)')
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed',
                        default=123445,
                        type=int,
                        help='random seed for experiments')
### Dataset global ###
    parser.add_argument("--device_net_path",
                        default = "./data/device_networks",
                        help="input directory for device network params")

    parser.add_argument("--op_net_path",
                        default = "./data/op_networks",
                        help= "input directory for operator network params")

    parser.add_argument("-p", "--data_parameters",
                        type=validate_file,
                        default='parameters/single_network.txt',
                        help="json text file specifying the training/testing dataset parameters", metavar="FILE")

    parser.add_argument('--disable_train',
                        action='store_false',
                        dest='train',
                        help='disable training')

    parser.add_argument('--disable_test',
                        action='store_false',
                        dest='test',
                        help='disable testing at the end of training')

    parser.add_argument('--run_folder',
                        type=validate_dir,
                        dest='load_dir',
                        help='directory to load existing run data')

    parser.add_argument('--embedding_model',
                        default='embedding.pk',
                        type=str,
                        help='file name of the embedding parameters')

    parser.add_argument('--policy_model',
                        default='policy.pk',
                        help='file name of the policy parameters')

    parser.add_argument("--test_parameters",
                        type=validate_file,
                        dest='test_para',
                        help="json text file specifying the testing dataset parameters", metavar="FILE")


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
                        default=16,
                        type=int,
                        help="hidden dimension for the network (default 64)")

    parser.add_argument('--enable_dataset_loading',
                        action='store_true',
                        dest='load_data',
                        help='enable dataset loading (training/testing data will not be generated)')

    parser.add_argument('--samples_to_ops_ratio',
                        default=2,
                        type=int,
                        help='the ratio of the number of iterations per episode to the number of operators during training (default: 1.5)')

    parser.add_argument('--max_num_training_episodes',
                        default=200,
                        type=int,
                        help='max number of training episodes (default: 200)')

    parser.add_argument('--min_num_training_episodes',
                        default=50,
                        type=int,
                        help='min number of training episodes (default: 50)')

    parser.add_argument('--memory_capacity',
                        default=4,
                        type=int,
                        help='capacity of the memory buffer for storing placement  (default: 10)')


    parser.add_argument('--disable_eval',
                        action='store_false',
                        dest='eval',
                        help='Disable evaluating a policy on test dataset every eval_frequency episodes')


    parser.add_argument('--num_of_eval_cases',
                        default=20,
                        type=int,
                        help='number of test cases for evaluation (default: 20)')

    parser.add_argument('--eval_frequency',
                        default=5,
                        help='The number of training episodes between two evaluations (default: 5)')

    parser.add_argument('--num_testing_cases',
                        default=200,
                        type=int,
                        help='max number of testing cases (default: 200)')

    parser.add_argument('--num_testing_cases_repeat',
                        default=2,
                        type=int,
                        help='number of times repeating each testing case (default: 2)')

    parser.add_argument(
        '--num_tuning_episodes',
        default=0,
        type=int,
        help='number of episodes for tuning (default: 0)')

### baselines ###
    parser.add_argument(
        '--use_placeto',
        action='store_true',
        help='Use placeto')

    parser.add_argument(
        '--placeto_k',
        type=int,
        default=8,
        help='Number of layers for placeto (default: 8)'
    )

    parser.add_argument(
        '--use_rl_op_est_device',
        dest='use_op_selection',
        action='store_true',
        help='Use RL operator selection and earlist start time device selection')

    return parser.parse_args()

if __name__ == '__main__':
    # Get user arguments and construct config
    exp_cfg = get_args()

    experiment = Experiment_on_data(exp_cfg)

    if exp_cfg.train:
        experiment.train()
    if exp_cfg.test:
        experiment.test(exp_cfg.test_para, exp_cfg.num_testing_cases)