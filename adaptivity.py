import torch

from env.utils import generate_network, generate_graph
from env.program import Program
from env.network import FullNetwork
from placement_rl.placement_env import PlacementEnv
from placement_rl.placement_agent import PlacementAgent
from placement_rl.placeto_agent import PlaceToAgent
from experiment import run_episodes

from copy import deepcopy
import argparse, os, json, pickle

import numpy as np


def validate_dir(f):
    if not os.path.isdir(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

def get_args():
    parser = argparse.ArgumentParser(description='Adaptivity Experiment Arguments')
    parser.add_argument('-d',
                        type=validate_dir,
                        dest='load_dir',
                        help='directory to load existing run data')
    return parser.parse_args()


def adaptivity_networks():
    net = generate_network(20, 1,
                     num_types=5,
                     type_prob=0.2,
                     avg_speed=5,
                     avg_bw=100,
                     avg_delay=10,
                     b_bw=0.8,
                     b_speed=0.8)
    last_network = FullNetwork(net['delay'], net['comm_speed'], net['speed'],
                net['device_constraints'])

    networks = [last_network]
    removed_devices = set()
    existing_devices = set(range(last_network.n_devices))

    for i in range(10):
        while True:
            random_device = np.random.choice(list(existing_devices))
            types = set().union(*[last_network.device_constraints[n] for n in existing_devices - {random_device}])
            if len(types) == 5:
                break
        removed_devices.add(random_device)
        existing_devices.remove(random_device)
        last_network = deepcopy(last_network)
        last_network.device_constraints[random_device] = []
        networks.append(last_network)

    for i in range(10):
        random_device = np.random.choice(list(removed_devices))
        removed_devices.remove(random_device)
        last_network = deepcopy(last_network)
        for j in range(5):
            if np.random.rand() < 0.2:
                last_network.device_constraints[random_device].append(j)
        if len(last_network.device_constraints[random_device]) == 0:
            last_network.device_constraints[random_device].append(np.random.choice(range(5)))
        last_network.comp_rate[random_device] = np.random.uniform(5 * 0.8, 5 * 1.2)
        networks.append(last_network)

    return networks

if __name__ == '__main__':
    # Get user arguments and construct config
    exp_cfg = get_args()
    logdir = exp_cfg.load_dir
    exp_cfg = pickle.load(open(os.path.join(logdir, 'args.pkl'),'rb'))

    networks = adaptivity_networks()
    programs = [Program(generate_graph(0.3,
                       10,
                       0.1,
                       seed,
                       5,
                       100,
                       100,
                       b_comp=0.2,
                       b_comm=0.2) )for seed in range(10)]

    env = PlacementEnv(networks, programs)
    pickle.dump(env, open('adaptivity_test.pk', 'wb'))
    if exp_cfg.use_placeto:
        agent = PlaceToAgent(len(PlacementEnv.PLACETO_FEATURES),
                                  exp_cfg.output_dim,
                                  device=torch.device('cpu'),
                                  n_device=20,
                                  k=exp_cfg.placeto_k,
                                  hidden_dim=exp_cfg.hidden_dim,
                                  lr=exp_cfg.lr,
                                  gamma=exp_cfg.gamma)
    else:
        agent = PlacementAgent(PlacementEnv.get_node_feature_dim(), PlacementEnv.get_edge_feature_dim(),
                               exp_cfg.output_dim,
                               device=torch.device('cpu'),
                               hidden_dim=exp_cfg.hidden_dim, lr=exp_cfg.lr, gamma=exp_cfg.gamma)

    _, cdir = os.walk(logdir)
    logdir = cdir[0]
    agent.policy.load_state_dict(torch.load(os.path.join(logdir, 'policy.pk')))
    agent.embedding.load_state_dict(torch.load(os.path.join(logdir, 'embedding.pk')))
    results = [[] for _ in range(len(programs))]
    for i in range(len(programs)):
        print('===========================================================================')
        print(f"RUNNING {i}th testing program ({i + 1}/{len(programs)}).")
        rt = run_episodes(env, agent,
                         [i] * 2 * len(networks),
                         list(range(len(networks))) * 2,
                         [20] * 2 * len(networks),
                         device=torch.device('cpu'),
                         use_placeto=exp_cfg.use_placeto,
                         use_full_graph=not exp_cfg.use_op_selection,
                         explore=True,
                         samples_to_ops_ratio=exp_cfg.samples_to_ops_ratio,
                         update_policy=False,
                         save_data=False,
                         noise=0)
        results[i].append(rt)
    pickle.dump(results,open(os.path.join(logdir, "adaptivity.pk"), "wb") )






