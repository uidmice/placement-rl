from experiment import generate_data
import torch
import pickle
import argparse, os, json, time
import itertools

from placement_rl.HDP import Placer
from placement_rl.placement_env import PlacementEnv
import numpy as np
import networkx as nx
import torch

from placement_rl.baseline import random_placement, heft, random_op_eft_dev, get_placement_constraints
from os import listdir
from os.path import isfile, join
from env.program import Program
from env.latency import evaluate, computation_latency, communication_latency, simulate


def validate_dir(f):
    if not os.path.isdir(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f


parser = argparse.ArgumentParser(description='Placement Experiment Arguments')
parser.add_argument('--noise',
                    default=0,
                    type=float)

parser.add_argument('--data_folder',
                    type=validate_dir,
                    help='Path to a folder that contains "test_data.pkl" to load for testing. '
                    )
parser.add_argument('--logdir',
                    type=validate_dir,
                    default='.',
                    help='Folder to log data (default: root directory)'
                    )

dir=parser.parse_args().data_folder
noise = parser.parse_args().noise
logdir = parser.parse_args().logdir


def get_test_cases(logdir):
    # _, cdir = os.walk(logdir)
    # logdir = cdir[0]
    networks, programs = pickle.load(open(join(logdir, 'test_data.pkl'), 'rb'))
    run_data = json.load(open(join(logdir, 'run_data.txt'), 'r'))
    n_id, p_id, init_map = [s[0] for s in run_data['test_sequence']], [s[1] for s in run_data['test_sequence']], [s[2]
                                                                                                                  for s
                                                                                                                  in
                                                                                                                  run_data[
                                                                                                                      'test_sequence']]
    norm_serial_min = []
    norm_serial_avg = []
    norm_cp = []

    for program_id, network_id in zip(p_id, n_id):

        prg = programs[program_id]
        net = networks[network_id]

        constraints = get_placement_constraints(prg, net)

        comp_min = []
        comm_min = []

        comp_a = []
        comm_a = []

        for n in prg.P.nodes():
            comp_t = [computation_latency(prg, net, n, dev) for dev in constraints[n]]
            comp_min.append(min(comp_t))
            comp_a.append(np.mean(comp_t))
            for e in prg.P.out_edges(n):
                prg.P.edges[e]['weight'] = min(comp_t)

        for e in prg.P.edges():
            d1 = constraints[e[0]]
            d2 = constraints[e[1]]
            comm_t = [communication_latency(prg, net, e[0], e[1], dev1, dev2) for dev1 in d1 for dev2 in d2]
            comm_min.append(min(comm_t))
            comm_a.append(np.mean(comm_t))
        norm_serial_min.append(np.sum(comp_min) + np.sum(comm_min))
        norm_serial_avg.append(np.sum(comp_a) + np.sum(comm_a))
        norm_cp.append(
            nx.shortest_path_length(prg.P, source=0, target=prg.P.number_of_nodes() - 1, weight='weight').item() +
            comp_min[-1])
    norm_cp = [a.item() for a in norm_cp]
    return logdir, p_id, n_id, init_map, programs, networks, norm_serial_min, norm_serial_avg, norm_cp

_, p_id, n_id, _, programs, networks, _, _, norm_cp = get_test_cases(dir)

def rnn_result(p_id, n_id, programs, networks, train_steps, noise, use_mask=True, hidden_dim=64, lr=0.1):
    env = PlacementEnv(networks, programs)
    results = []
    runtime = []
    size = []
    rewards = []
    for program_id, network_id in zip(p_id, n_id):
        prg = programs[program_id]
        net = networks[network_id]
        MAX_NUM_NODES = prg.n_operators
        MAX_OUT_DEGREE = max(dict(prg.P.out_degree()).values())
        NUM_DEV = net.n_devices
        agent = Placer(6 + MAX_OUT_DEGREE + MAX_NUM_NODES,
                       hidden_dim,
                       NUM_DEV,
                       lr=lr)
        R = []
        L = []
        running_time = 0
        update_time = 0
        for i in range(train_steps):
            start_time = time.time()
            embedding = env.get_hdp_embedding(program_id, MAX_OUT_DEGREE, MAX_NUM_NODES)
            B = 4
            s = embedding.unsqueeze(0).repeat(B, 1, 1)
            constraints = env.get_placement_constraints(program_id, network_id)
            if use_mask:
                mask = torch.zeros(MAX_NUM_NODES, NUM_DEV)
                for j in range(MAX_NUM_NODES):
                    mask[j, constraints[j]] = 1
                ret, log_prob = agent(s, mask=mask)
            else:
                ret, log_prob = agent(s)
            reward = torch.zeros(B)
            lat = []
            for j in range(B):
                placement = [a[j] for a in ret]
                validity = [p not in g for p, g in zip (placement, constraints)]
                if any(validity):
                    reward[j] = -100
                    latency = -1
                else:
                    latency, path, G_stats = env.simulate(program_id, network_id, placement, noise)
                    reward[j] = -torch.sqrt(latency)
                lat.append(latency)

            running_time += time.time() - start_time

            R.append(torch.mean(reward))
            L.append(torch.min(torch.tensor([l for l in lat if l > 0])))

            start_time = time.time()
            agent.update(log_prob, reward-torch.mean(torch.tensor(R)))
            update_time += time.time() - start_time

        rewards.append(R)
        runtime.append([running_time, update_time])
        results.append(L)
        size.append([MAX_NUM_NODES, MAX_OUT_DEGREE])
    return results, rewards, runtime, size

# print(len(p_id1))
results, rewards, runtime, size = rnn_result(p_id, n_id, programs, networks, 100, noise)
pickle.dump([results, rewards,  runtime, size], open(join(logdir, f'rnn_noise_{noise}.pk'), 'wb'))