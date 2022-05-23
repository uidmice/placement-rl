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

n_devices = 20
seed = 1

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
    num_types = 5
    type_prob = 0.3
    avg_speed = 5
    avg_bw = 100
    avg_delay = 10
    b_bw = 0.2
    b_speed = 0.2

    net = generate_network(n_devices, seed,
                     num_types=num_types,
                     type_prob=type_prob,
                     avg_speed=avg_speed,
                     avg_bw=avg_bw,
                     avg_delay=avg_delay,
                     b_bw=b_bw,
                     b_speed=b_speed)
    last_network = FullNetwork(net['delay'], net['comm_speed'], net['speed'],
                net['device_constraints'])

    networks = [last_network]
    removed_devices = set()
    existing_devices = set(range(last_network.n_devices))

    for i in range(20):
        if i % 4 == 0 or   i % 4 == 1:
            n_device_removed = (i % 4 + 1)*2
            while True:
                random_devices = np.random.choice(list(existing_devices), n_device_removed, replace=False).tolist()
                types = set().union(*[last_network.device_constraints[n] for n in existing_devices - set(random_devices)])
                if len(types) == 5:
                    break
            removed_devices.update(random_devices)
            existing_devices.difference_update(random_devices)
            last_network = deepcopy(last_network)
            for d in random_devices:
                last_network.device_constraints[d] = []
            networks.append(last_network)

        else:
            n_device_added = (i % 4 - 1)*2
            random_devices = np.random.choice(list(removed_devices), n_device_added, replace=False).tolist()
            removed_devices.difference_update(random_devices)
            existing_devices.update(random_devices)
            last_network = deepcopy(last_network)
            for d in random_devices:
                last_network.comp_rate[d] = 1/np.random.uniform(avg_speed * (1-b_speed/2), avg_speed* (1+b_speed/2))
                for j in range(num_types):
                    if np.random.rand() < type_prob:
                        last_network.device_constraints[d].append(j)

                if len(last_network.device_constraints[d]) == 0:
                    last_network.device_constraints[d].append(np.random.choice(range(num_types)))
                last_network.device_constraints[d] = list(set(last_network.device_constraints[d]))
            networks.append(last_network)
    return networks

def adaptivity_programs():
    return  [Program(generate_graph(0.3,
                       10,
                       0.1,
                       seed,
                       5,
                       100,
                       100,
                       b_comp=0.2,
                       b_comm=0.2) )for seed in range(10)]

def adaptivity_test_data():
    networks = adaptivity_networks()
    programs = adaptivity_programs()
    env = PlacementEnv(networks, programs)

    pickle.dump(env, open('adaptivity_dataset.pk', 'wb'))


if __name__ == '__main__':
    # Get user arguments and construct config
    exp_cfg = get_args()
    logdir = exp_cfg.load_dir
    exp_cfg = pickle.load(open(os.path.join(logdir, 'args.pkl'),'rb'))

    env = pickle.load(open('adaptivity_dataset.pk', 'rb'))
    networks = env.networks
    programs = env.programs

    if exp_cfg.use_placeto:
        agent = PlaceToAgent(len(PlacementEnv.PLACETO_FEATURES),
                                  exp_cfg.output_dim,
                                  device=torch.device('cpu'),
                                  n_device=n_devices,
                                  k=exp_cfg.placeto_k,
                                  hidden_dim=exp_cfg.hidden_dim,
                                  lr=exp_cfg.lr,
                                  gamma=exp_cfg.gamma)
    else:
        agent = PlacementAgent(PlacementEnv.get_node_feature_dim(), PlacementEnv.get_edge_feature_dim(),
                               exp_cfg.output_dim,
                               device=torch.device('cpu'),
                               hidden_dim=exp_cfg.hidden_dim, lr=exp_cfg.lr, gamma=exp_cfg.gamma)

    name = logdir.split('_')[-1]
    try:
        _, cdir = os.walk(logdir)
        dir = cdir[0]
    except:
        dir = logdir
    logdir = dir

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
                         samples_to_ops_ratio=2,
                         update_policy=False,
                         save_data=False,
                         noise=0)
        results[i].append(rt)
    pickle.dump(results,open( f"adaptivity_{name}.pk", "wb") )






