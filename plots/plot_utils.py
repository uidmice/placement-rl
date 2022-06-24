import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
import random

from placement_rl.baseline import random_placement, heft, random_op_eft_dev, get_placement_constraints
from os import listdir
from os.path import isfile, join
from env.utils import generate_graph, generate_network
from env.network import StarNetwork, FullNetwork
from env.program import Program
from env.latency import evaluate, computation_latency, communication_latency, simulate

from placement_rl.placement_agent import PlacementAgent

import os, itertools
import pickle, json

import time


num_x = 20


def get_eval_cases(logdir):
    name = 'eval_data.pkl'
    for root, dirs, files in os.walk(logdir):
        if name in files:
            dir = os.path.join(root, name)
            break
    print(dir)
    networks, programs = pickle.load(open(dir, 'rb'))
    run_data = json.load(open(join(root, 'run_data.txt'), 'r'))
    n_id, p_id, init_map = [s[0] for s in run_data['eval_sequence']], \
                           [s[1] for s in run_data['eval_sequence']], \
                           [s[2] for s in run_data['eval_sequence']]

    norm_cp = []

    for program_id, network_id in zip(p_id, n_id):

        prg = programs[program_id]
        net = networks[network_id]

        constraints = get_placement_constraints(prg, net)

        comp_min = []

        for n in prg.P.nodes():
            comp_t = [computation_latency(prg, net, n, dev) for dev in constraints[n]]
            comp_min.append(min(comp_t))
            for e in prg.P.out_edges(n):
                prg.P.edges[e]['weight'] = min(comp_t)

        norm_cp.append(
            nx.shortest_path_length(prg.P, source=0, target=prg.P.number_of_nodes() - 1, weight='weight').item() +
            comp_min[-1])
    norm_cp = [a.item() for a in norm_cp]
    return p_id, n_id, init_map, programs, networks, norm_cp


def load_eval_data(logdir, p_id, n_id, init_map):
    eval_data = pickle.load(open(join(logdir, 'eval.pk'), 'rb'))
    results = []
    for test_program, test_network, init_seed in zip(p_id, n_id, init_map):
        e = [list(filter(lambda episode:
                         episode['network_id'] == test_network and
                         episode['program_id'] == test_program and
                         episode['init_seed'] == init_seed, record)) for record in eval_data]
        latency = [[a['latency_trace'] for a in r] for r in e]
        results.append(latency)
    return results


def get_test_cases(logdir):
    name = 'test_data.pkl'
    for root, dirs, files in os.walk(logdir):
        if name in files:
            dir = os.path.join(root, name)
            break
    print(dir)
    networks, programs = pickle.load(open(dir, 'rb'))
    run_data = json.load(open(join(root, 'run_data.txt'), 'r'))
    n_id, p_id, init_map = [s[0] for s in run_data['test_sequence']], \
                           [s[1] for s in run_data['test_sequence']], \
                           [s[2] for s in run_data['test_sequence']]

    norm_cp = []

    for program_id, network_id in zip(p_id, n_id):
        prg = programs[program_id]
        net = networks[network_id]

        constraints = get_placement_constraints(prg, net)

        comp_min = []

        for n in prg.P.nodes():
            comp_t = [computation_latency(prg, net, n, dev) for dev in constraints[n]]
            comp_min.append(min(comp_t))
            for e in prg.P.out_edges(n):
                prg.P.edges[e]['weight'] = min(comp_t)

        norm_cp.append(
            nx.shortest_path_length(prg.P, source=0, target=prg.P.number_of_nodes() - 1, weight='weight').item() +
            comp_min[-1])
    norm_cp = [a.item() for a in norm_cp]
    return p_id, n_id, init_map, programs, networks, norm_cp


def get_random_sample_data(p_id, n_id, programs, networks, repeat, noise=0):
    results = []
    for program_id, network_id in zip(p_id, n_id):
        prg = programs[program_id]
        net = networks[network_id]

        constraints = get_placement_constraints(prg, net)
        idx = np.linspace(1, 2 * prg.n_operators, num_x).astype(int) - 1

        sampled_latencies = []
        for i in range(repeat):
            _, random_latency = random_placement(prg, net, constraints, 2 * prg.n_operators, noise)
            sampled_latencies.append(random_latency)

        sampled_latencies = np.array(sampled_latencies)
        results.append(sampled_latencies[:, idx])
    return results


def get_random_op_data(p_id, n_id, init_map, programs, networks, repeat, noise=0):
    results = []
    for program_id, network_id, init_seed in zip(p_id, n_id, init_map):
        prg = programs[program_id]
        net = networks[network_id]

        np.random.seed(init_seed)

        constraints = get_placement_constraints(prg, net)
        init_mapping = [np.random.choice(constraints[i]) for i in range(prg.n_operators)]
        idx = np.linspace(1, 2 * prg.n_operators, num_x).astype(int) - 1

        sampled_latencies = []
        for i in range(repeat):
            _, random_latency = random_op_eft_dev(prg, net, init_mapping, constraints, 2 * prg.n_operators, noise)
            sampled_latencies.append(random_latency)

        sampled_latencies = np.array(sampled_latencies)
        results.append(sampled_latencies[:, idx])

    return results


def get_heft_data(p_id, n_id, programs, networks, repeat, noise):
    results = []
    for p, n in zip(p_id, n_id):
        prg = programs[p]
        net = networks[n]
        constraints = get_placement_constraints(prg, net)
        mapping = heft(prg, net, constraints)
        _, l, _ = simulate(mapping, prg, net, noise, repeat)
        results.append(l)
    return results

def get_test_data(test_dir, p_id, n_id, init_map):
    results = []
    for test_program, test_network, init_seed in zip(p_id, n_id, init_map):
        e = pickle.load(
            open(join(test_dir, f'test_program_{test_program}_network_{test_network}_seed_{init_seed}.pk'), 'rb'))
        latency = [a['latency_trace'] for a in e]
        idx = np.linspace(1, len(latency[0]), num_x).astype(int) - 1
        lat_train = np.array(latency)
        results.append(lat_train[:, idx])

    return results

marker = [['s', 'dodgerblue', '--'], ['s', 'deepskyblue', '--'], ['s', 'cadetblue', '--'],
          ['X', 'green', ':'], ['D', 'coral', '--'], ['d', 'red', ':'],
          ['x', 'orange', '--'], ['D', 'peru', ':'], ['d', 'plum', '--'],
          ['P', 'palegreen', '--'], ['D', 'aqua', ':'], ['d', 'magenta', '--']]


def plot_eval_performance_across_training_episodes(logdir, plot_best=True, title=None):
    p_id, n_id, init_map, programs, networks, norms = get_eval_cases(logdir)
    dirs = []
    for a in os.listdir(logdir):
        p = os.path.join(logdir, a)
        if os.path.isdir(p):
            dirs.append(p)

    names = [p.split('_')[-1] for p in dirs]

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.set_ylabel('Average SLR')

    for p, name, k in zip(dirs, names, range(len(names))):
        eval_data = load_eval_data(p, p_id, n_id, init_map)
        results = [[] for _ in range(len(eval_data[0]))]
        for latency, test_program, norm in zip(eval_data, p_id, norms):
            if plot_best:
                latency = [[[min(a[:i + 1]) for i in range(len(a))] for a in r] for r in latency]

            idx = np.linspace(1, len(latency[0][0]), num_x).astype(int) - 1

            for trace, i in zip(latency, range(len(latency))):
                lat_train = np.array(trace) / norm
                results[i].append(lat_train[:, idx])

        comb = [np.concatenate(a, axis=0) for a in results]
        output_results = np.array([a[:, -1] for a in comb])
        average = np.average(output_results, axis=1)
        std = np.std(output_results, axis=1) / 20
        ax.plot((np.arange(0, len(average)) + 1) * 5, average, marker=marker[k][0], color=marker[k][1],
                linestyle=marker[k][2], label=name, ms=2)
        ax.fill_between((np.arange(0, len(average)) + 1) * 5, average + std, average - std, color=marker[k][1],
                        alpha=.1, edgecolor='none')

        best_idx = np.argmin(average)
        print(f'{name}: best case is index {best_idx}, model index {best_idx * 5 + 5}')

    ax.legend(fontsize=10, ncol=3, loc='upper center', bbox_to_anchor=(0.53, -0.15, 0, 0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('# of training episodes', size=13)
    ax.set_ylabel('Average SLR', size=13)
    if title:
        ax.set_title(title, size=14)

    fig.tight_layout()
    plt.savefig(os.path.join(logdir, 'training.pdf'), format="pdf", dpi=200, bbox_inches="tight")
    plt.show()
    return


def plot_performance_slr_samples(logdir, ax, noise, plot_best=True, update=False, ylim=None, legend=True, title=None):
    p_id, n_id, init_map, programs, networks, norms = get_test_cases(logdir)

    dirs = []
    for a in os.listdir(logdir):
        p = os.path.join(logdir, a)
        if os.path.isdir(p):
            dirs.append(p)

    names = [p.split('_')[-1] for p in dirs]
    ax.set_ylabel('SLR', size=14)

    for p, name, k in zip(dirs, names, range(len(names))):
        for a in os.listdir(p):
            test_dir = join(p, a)
            if os.path.isdir(test_dir) and 'test_' in a:
                print(test_dir)
                if os.path.exists(join(test_dir, 'test_results.pk')):
                    data = pickle.load(open(join(test_dir, 'test_results.pk'), 'rb'))
                else:
                    data = get_test_data(test_dir, p_id, n_id, init_map)
                    pickle.dump(data, open(join(test_dir, 'test_results.pk'), 'wb'))
                results = []
                for d, norm in zip(data, norms):
                    c = d / norm
                    if plot_best:
                        c = [[min(trace[:i + 1]) for i in range(num_x)] for trace in c]
                    results.extend(c)
                results = np.array(results)
                average = np.mean(results, axis=0)
                x = np.linspace(0, 2, num_x)
                ax.plot(x, average, marker=marker[k][0], color=marker[k][1], linestyle=marker[k][2], label=name, ms=5)
                break
    # plot random sample baseline
    if not os.path.exists(join(logdir, 'test_random_samples.pk')) or update:
        random_samples = get_random_sample_data(p_id, n_id, programs, networks, 2, noise=noise)
        pickle.dump(random_samples, open(join(logdir, 'test_random_samples.pk'), 'wb'))
    else:
        random_samples = pickle.load(open(join(logdir, 'test_random_samples.pk'), 'rb'))
    results = []
    for d, norm in zip(random_samples, norms):
        c = d / norm
        if plot_best:
            c = [[min(trace[:i + 1]) for i in range(num_x)] for trace in c]
        results.extend(c)
    results = np.array(results)
    average = np.mean(results, axis=0)
    x = np.linspace(0, 2, num_x)
    ax.plot(x, average, marker=marker[k+1][0], color=marker[k+1][1], linestyle=marker[k+1][2], label='Random Samples', ms=5)


    # plot random op baseline
    if not os.path.exists(join(logdir, 'test_random_op.pk')) or update:
        random_op = get_random_op_data(p_id, n_id, init_map, programs, networks, 2, noise=noise)
        pickle.dump(random_op, open(join(logdir, 'test_random_op.pk'), 'wb'))
    else:
        random_op = pickle.load(open(join(logdir, 'test_random_op.pk'), 'rb'))
    results = []
    for d, norm in zip(random_op, norms):
        c = d / norm
        if plot_best:
            c = [[min(trace[:i + 1]) for i in range(num_x)] for trace in c]
        results.extend(c)
    results = np.array(results)
    average = np.mean(results, axis=0)
    x = np.linspace(0, 2, num_x)
    ax.plot(x, average, marker=marker[k+2][0], color=marker[k+2][1], linestyle=marker[k+2][2], label='Random Op', ms=5)


    ax.set_xlabel('# of samples/# of tasks in the graph')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if legend:
        ax.legend()
    if title:
        ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    return ax
# plot_eval_performance_across_training_episodes('runs2/multiple_networks')

fig, ax = plt.subplots(1,1, figsize=(5,4))
plot_performance_slr_samples('runs2/multiple_networks', ax, 0)
plt.show()


