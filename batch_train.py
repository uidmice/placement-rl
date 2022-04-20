import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

from baseline import exhaustive, random_placement, heft, random_op_greedy_dev, random_op_est_dev
from env.utils import generate_program, generate_network, load_pickle
from env.network import StarNetwork
from env.program import Program
from env.latency import evaluate

from placement_rl.placement_env import PlacementEnv
from placement_rl.placement_agent import PlacementAgent

import os
import pickle


# Load data, split into train and val.
def load_data_from_dir(network_path, program_path, train_ratio = 0.8):
    network_fns = os.listdir(network_path)
    program_fns = os.listdir(program_path)

    network_fn = np.random.choice(network_fns[0])
    network = load_pickle(network_fn)
    delay, bw, speed = network["delay"], network["bw"], network["speed"]
    network = StarNetwork(delay, bw, speed)

    programs = []
    for program_fn in program_fns:
        program = load_pickle(program_fn)
        G, constraints = program["G"], program["constraints"]
        program = Program(G, constraints)
        programs.append(program)

    n_train = int(len(programs) * train_ratio)
    n_val = len(programs) - n_train

    networks_train = [network] * n_train
    networks_val = [network] * n_val

    env_train = PlacementEnv(networks_train, programs[:networks_train])
    env_val = PlacementEnv(networks_val, programs[networks_train+1:])

    return env_train, env_val


# Training for the same device network.
def train_for_one_cluster(env,
                          agent,
                          max_iter = 50,
                          noise = 0,
                          update_op_net = True,
                          update_dev_net = True,
                          use_baseline = True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_training_instance = len(env.programs)

    mapping_dict = {}

    latency_improvements = []

    # loop over all programs-network instance
    for idx_instance in tqdm(range(num_training_instance)):
        network = env.networks[idx_instance]
        program = env.programs[idx_instance]

        mask_dev = torch.zeros(network.n_devices).to(device)
        mask_op = torch.ones(program.n_operators).to(device)
        mask_op[0] = mask_op[-1] = 0

        # if idx_instance not in mapping_dict:
        #     mapping = program.random_mapping()
        #     mapping_dict[idx_instance] = mapping
        # else:
        #     mapping = mapping_dict[idx_instance]

        mapping = program.random_mapping()

        init_latency, path, G_stats = env.simulate(idx_instance, idx_instance, noise)

        # For each of program-network instance, we do max_iter times update
        for t in range(max_iter):
            mapping = mapping.copy()

            g = env.get_cardinal_graph(idx_instance,idx_instance, mapping, path, G_stats).to(device)
            s = agent.op_selection(g, mask_op)

            # Assume greedy device selection
            action = agent.dev_selection_est(program,
                                             network,
                                             cur_mapping,
                                             G_states,
                                             s,
                                             program.placement_constraints[s])
            mapping[s] = action
        final_latency, final_path, final_G_stats = env.simulate(idx_instance, idx_instance, noise)
        agent.finish_episode(update_op_net, update_dev_net, use_baseline)

        latency_improvements.append(init_latency - final_latency)

    return latency_improvements



def val_for_one_cluster(env,
                        agent,
                        max_iter = 50,
                        noise = 0,
                        update_op_net = True,
                        update_dev_net = True,
                        use_baseline = True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_val_instance = len(env.programs)

    mapping_dict = {}

    latency_improvements = []

    with torch.no_grad():
        for idx_instance in tqdm(range(num_val_instance)):
            network = env.networks[idx_instance]
            program = env.programs[idx_instance]

            mask_dev = torch.zeros(network.n_devices).to(device)
            mask_op = torch.ones(program.n_operators).to(device)
            mask_op[0] = mask_op[-1] = 0

            # if idx_instance not in mapping_dict:
            #     mapping = program.random_mapping()
            #     mapping_dict[idx_instance] = mapping
            # else:
            #     mapping = mapping_dict[idx_instance]

            mapping = program.random_mapping()

            init_latency, path, G_stats = env.simulate(idx_instance, idx_instance, noise)

            for t in range(max_iter):
                mapping = mapping.copy()

                g = env.get_cardinal_graph(idx_instance,idx_instance, mapping, path, G_stats).to(device)
                s = agent.op_selection(g, mask_op)

                # Assume greedy device selection
                action = agent.dev_selection_est(program,
                                                 network,
                                                 cur_mapping,
                                                 G_states,
                                                 s,
                                                 program.placement_constraints[s])
                mapping[s] = action
            final_latency, final_path, final_G_stats = env.simulate(idx_instance, idx_instance, noise)
            latency_improvements.append(init_latency - final_latency)

    return latency_improvements



def main_for_one_cluster(network_path = "./data/network_20",
                         program_path = "./data/dag_network_20",
                         n_epoch = 30,
                         visualize_training = False):
    env_train, env_val = load_data_from_dir(network_path, program_path, train_ratio = 0.8)

    agent = PlacementAgent(env.get_node_feature_dim(),
                           env.get_edge_feature_dim(),
                           10, 10)

    # Scheduler for operator optimizer
    op_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(agent.op_network_optim,
                                                              mode = "max",
                                                              factor = 0.8,
                                                              patience = 2,
                                                              threshold = 0.01)

    improves_train = []
    improves_val = []
    for epoch in range(n_epoch):
        improve_train = train_for_one_cluster(env_train, agent, max_iter = 50)
        avg_improve_train = sum(improve_train) / len(improve_train)
        improves_train.append(avg_improve_train)


        improve_val = val_for_one_cluster(env_val, agent, max_iter = 50)
        avg_improve_val = sum(improve_val) / len(improve_val)
        improves_val.append(avg_improve_val)

        op_scheduler.step(avg_improve_val)

    if visualize_training:
        plt.plot(list(range(len(improves_train))), improves_train, label = "train")
        plt.plot(list(range(len(improves_val))), improves_val, label = "val")
        plt.xlabel("epoch")
        plt.ylabel("latency")
        plt.legend()
        plt.show()


    return improves_train, improves_val















