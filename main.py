import torch

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

def train(env, agent, init_mapping, episodes, max_iter=100):
    rewards = []
    mask = torch.zeros(env.n_devices).to(device)
    for i in range(episodes):
        cur_mapping = init_mapping.copy()
        ep_reward = 0
        for t in range(max_iter):
            graphs = []
            temp_mapping = cur_mapping.copy()
            g = env.get_placement_graph(temp_mapping).to(device)
            s = agent.op_selection(g)
            parallel = env.op_parallel[s]
            constraints = env.program.placement_constraints[s]
            mask[:] = 0
            mask[constraints] = 1
            for d in range(n_devices):
                if d != cur_mapping[s]:
                    temp_mapping[s] = d
                    t_g = env.get_placement_graph(temp_mapping).to(device)
                    graphs.append(t_g)
                else:
                    graphs.append(g)
            action = agent.dev_selection(graphs, s, parallel, mask=mask)
            cur_mapping[s] = action
            reward, _ = env.evaluate(cur_mapping)
            reward = -reward / 500
            agent.saved_rewards.append(reward)
            ep_reward += reward
        agent.finish_episode()
        rewards.append(ep_reward)
    return rewards

mapping = program.random_mapping()
rewards = train(env, agent, 2, mapping)





# agent = PlacementAgent(env.get_state_dim(), n_devices)
# episode_rewards, reward_trace = agent.train(env, 300)