import numpy as np
import torch
import matplotlib.pyplot as plt
import random

from functions import exhaustive
from env.utils import generate_program, generate_network
from env.network import StarNetwork
from env.program import Program

from placement_rl.placement_env import PlacementEnv
from placement_rl.placement_agent import PlacementAgent

import os
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def get_config(n_devices, n_programs, s1=0, s2=[0]):
#     p = f'data/N_{n_devices}'
#     delay, bw, speed = pickle.load(open(os.path.join(p, f'network_{n_devices}_seed_{s1}.pk'), 'rb'))
#
#     network = StarNetwork(delay, bw, speed)
#     programs = []
#     for seed in s2:
#         DAG, constraints = pickle.load(open(os.path.join(p, f'program_{n_programs}_seed_{seed}.pk'), 'rb'))
#         programs.append(Program(DAG, constraints, network))
#     return network, programs


n_devices = 20
n_operators = 10
network = StarNetwork(*generate_network(n_devices, seed=1))
DAG, constraints = generate_program(n_operators, n_devices, seed=2)
program = Program(DAG, constraints)
env = PlacementEnv(network, program)
agent = PlacementAgent(env.get_node_feature_dim(),
                       env.get_edge_feature_dim(),
                       10, 10)
mapping = program.random_mapping()
print(program.placement_constraints)

m_matrix = np.zeros((n_operators, n_devices))
m_matrix[0, program.pinned[0]] = 1
m_matrix[-1, program.pinned[1]] = 1

# min_mapping, min_L, solution = exhaustive(m_matrix, program, network)

def train(env, agent, init_mapping, episodes, max_iter=50, update_op_net=True, update_dev_net=True, greedy_dev_selection=False):
    op_rewards = []

    lat_records = []
    act_records = []

    mask = torch.zeros(env.n_devices).to(device)

    for i in range(episodes):
        cur_mapping = init_mapping.copy()
        last_latency, _ = env.evaluate(init_mapping)

        print(f'=== Episode {i} ===')
        latencies = [last_latency]
        actions = []
        ep_reward = 0
        for t in range(max_iter):
            graphs = []
            temp_mapping = cur_mapping.copy()
            g = env.get_placement_graph(temp_mapping).to(device)
            s = agent.op_selection(g)

            if greedy_dev_selection:
                action = agent.dev_selection_greedy(cur_mapping, s, env.program.placement_constraints[s], env)
                action = random.choice(action)
            else:
                parallel = env.op_parallel[s]
                constraints = env.program.placement_constraints[s]
                mask[:] = 0
                mask[constraints] = 1
                for d in range(n_devices):
                    temp_mapping[s] = d
                    t_g = env.get_placement_graph(temp_mapping).to(device)
                    graphs.append(t_g)
                action = agent.dev_selection(graphs, s, parallel, mask=mask)

            # print(f'Mapping operator {s} to device {action}')
            cur_mapping[s] = action
            latency, _ = env.evaluate(cur_mapping)
            # if latency < last_latency:
            #     reward = 1
            # else:
            #     reward = 0
            reward = (last_latency - latency)/10
            last_latency = latency

            latencies.append(latency)
            actions.append([s, action])
            agent.saved_rewards.append(reward)
            ep_reward = reward + ep_reward * agent.gamma
        agent.finish_episode(update_op_net, update_dev_net)
        print(cur_mapping)
        op_rewards.append(ep_reward)
        lat_records.append(latencies)
        act_records.append(actions)
    return op_rewards, lat_records, act_records


rewards, lat_records, action_records = train(env, agent, mapping, 100,  max_iter=50, update_op_net=True, update_dev_net=False, greedy_dev_selection=True)

