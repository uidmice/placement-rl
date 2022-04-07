import numpy as np
import torch
import matplotlib.pyplot as plt
import random

from baseline import exhaustive, random_placement, heft, random_op_greedy_dev, random_op_est_dev
from env.utils import generate_program, generate_network
from env.network import StarNetwork
from env.program import Program
from env.latency import evaluate

from placement_rl.placement_env import PlacementEnv
from placement_rl.placement_agent import PlacementAgent

import os
import pickle

from re import A
from placement_rl.gnn import OpNet, reset
from placement_rl.rl_agent import SoftmaxActor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


n_devices = [10, 20, 30]
n_operators = [10, 20, 30]
networks = {n: StarNetwork(*generate_network(n, seed=0)) for n in n_devices}
programs = {}
for n in n_devices:
    programs[n] = {}
    for m in n_operators:
        programs[n][m] = []
        for seed in range(5):
            DAG, constraints = generate_program(m, n, seed=seed)
            programs[n][m].append(Program(DAG, constraints))

output_dim = 10

program = programs[10][10][0]
network = networks[10]
env = PlacementEnv([network], [program])

device_network = OpNet(env.get_node_feature_dim(), env.get_edge_feature_dim(), output_dim).to(device)

dev_policy = SoftmaxActor(output_dim).to(device)

dev_optim = torch.optim.Adam(list(device_network.parameters())+list(dev_policy.parameters()), lr=0.03)

num_episodes = 20
num_iterations = 50

mapping = program.random_mapping()
print(program.placement_constraints)

last_latency, path, G_stats = env.simulate(0, 0, mapping, 0)
env.get_full_graph(0, 0, mapping, path, G_stats)

def train_one(env,
          init_mapping,
          episodes,
          max_iter=50,
          use_baseline=True,
          noise=0,
          gamma=0.95):
    op_rewards = []

    map_records = [init_mapping]
    lat_records = []
    act_records = []

    program = env.programs[0]
    network = env.networks[0]

    mask_dev = torch.zeros(network.n_devices).to(device)

    currrent_stop_iter = max_iter

    node_dict = env.full_graph_node_dict[0][0]
    action_dict = {}
    for o in node_dict:
        for d in node_dict[o]:
            action_dict[node_dict[o][d]] = (o, d)

    for i in range(episodes):
        cur_mapping = init_mapping.copy()
        last_latency, path, G_stats = env.simulate(0, 0, init_mapping, noise)

        print(f'=== Episode {i} ===')
        latencies = [last_latency]
        actions = []
        ep_reward = 0

        mask_op = torch.ones(program.n_operators).to(device)
        mask_op[0] = mask_op[-1] = 0

        saved_rewards = []
        dev_log_probs = []

        for t in range(currrent_stop_iter):
            temp_mapping = cur_mapping.copy()
            g1 = env.get_full_graph(0, 0, temp_mapping, path, G_stats).to(device)
            dev_embedding = device_network(g1).to(device)
            probs = dev_policy(dev_embedding).to(device)
            m = torch.distributions.Categorical(probs=probs)
            # action = list(node_dict[s].keys())[m.sample().item()]

            a = m.sample()
            dev_log_probs.append(m.log_prob(a))
            s, action = action_dict[a.item()]
            cur_mapping[s] = action
            latency, path, G_stats = env.simulate(0, 0, cur_mapping, noise)
            reward = (last_latency - latency) / 10
            last_latency = latency

            latencies.append(latency)
            actions.append([s, action])
            saved_rewards.append(reward)
            ep_reward = reward + ep_reward * gamma

        R = 0
        dev_policy_loss = 0
        returns = []
        for r in saved_rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        if use_baseline:
            for i in range(len(saved_rewards)):
                if i == 0:
                    bk = saved_rewards[0]
                else:
                    try:
                        bk = sum(saved_rewards[:i + 1]) / len(saved_rewards[:i + 1])
                    except:
                        bk = sum(saved_rewards) / len(saved_rewards)
                returns[i] -= bk

        returns = torch.tensor(returns).to(device)
        # returns = (returns - returns.mean()) / (returns.std() + epsilon)

        dev_optim.zero_grad()
        for log_prob, R in zip(dev_log_probs, returns):
            dev_policy_loss = dev_policy_loss - log_prob * R
        dev_policy_loss.backward()
        dev_optim.step()
        op_rewards.append(ep_reward)
        map_records.append(cur_mapping)
        lat_records.append(latencies)
        act_records.append(actions)

    return op_rewards, lat_records, act_records, map_records

op_rewards, lat_records, act_records, map_records = train_one(env, mapping, 100, max_iter=num_iterations, use_baseline=True, noise=0,  gamma=0.95)

def train(env,
          agent,
          init_mapping,
          episodes,
          max_iter=50,
          update_op_net=True,
          update_dev_net=True,
          greedy_dev_selection=True,
          use_baseline=True,
          noise=0,
          short_earlier_ep=False,
          early_stop = 5):
    op_rewards = []

    map_records = [init_mapping]
    lat_records = []
    act_records = []

    program = env.programs[0]
    network = env.networks[0]

    mask_dev = torch.zeros(network.n_devices).to(device)

    currrent_stop_iter = max_iter
    if short_earlier_ep:
        currrent_stop_iter = early_stop



    for i in range(episodes):
        cur_mapping = init_mapping.copy()
        last_latency, path, G_stats = env.simulate(0, 0, init_mapping, noise)

        print(f'=== Episode {i} ===')
        latencies = [last_latency]
        actions = []
        ep_reward = 0

        mask_op = torch.ones(program.n_operators).to(device)
        mask_op[0] = mask_op[-1] = 0

        for t in range(currrent_stop_iter):
            graphs = []
            temp_mapping = cur_mapping.copy()
            g = env.get_cardinal_graph(0, 0, temp_mapping, path, G_stats).to(device)
            s = agent.op_selection(g, mask_op)

            if greedy_dev_selection:
                action = agent.dev_selection_est(program, network, cur_mapping, G_stats, s, program.placement_constraints[s])
            else:
                parallel = program.op_parallel[s]
                constraints = program.placement_constraints[s]
                mask_dev[:] = 0
                mask_dev[constraints] = 1
                for d in range(n_devices):
                    temp_mapping[s] = d
                    t_g = env.get_cardinal_graph(temp_mapping).to(device)
                    graphs.append(t_g)
                action = agent.dev_selection(graphs, s, parallel, mask=mask_dev)

            cur_mapping[s] = action
            latency, path, G_stats = env.simulate(0, 0, cur_mapping, noise)
            # reward = -latency/10
            # reward = -np.sqrt(latency)/10
            reward = (last_latency - latency)/10
            last_latency = latency

            latencies.append(latency)
            actions.append([s, action])
            agent.saved_rewards.append(reward)
            ep_reward = reward + ep_reward * agent.gamma

        agent.finish_episode(update_op_net, update_dev_net, use_baseline)
        # agent.finish_episode_REINFORCE(update_op_net, update_dev_net)
        # last_latency, _ = env.evaluate(init_mapping)
        # agent.finish_episode_REINFORCE_latency(last_latency, update_op_net,update_dev_net)
        # agent.finish_episode_REINFORCE_latency_sqrt(last_latency,update_op_net,update_dev_net)
        op_rewards.append(ep_reward)
        map_records.append(cur_mapping)
        lat_records.append(latencies)
        act_records.append(actions)


        if short_earlier_ep and len(lat_records) >= 2:
            last_lat = lat_records[-2]
            incr_flag = 0
            for i in range(len(last_lat)):
                if last_lat[i] < latencies[i]:
                    incr_flag = 0
                    break
                incr_flag = 1
            if incr_flag:
                currrent_stop_iter += 1


    return op_rewards, lat_records, act_records, map_records


# test_name = "10episode_REINFORCE_original"
# plt.plot(range(len(rewards)), rewards)
# plt.savefig("./test_imgs/reward_{}.png".format(test_name))
# plt.clf()
#
# for i in range(0, len(lat_records),2):
#     plt.plot(range(len(lat_records[0])), lat_records[i], label = "episode: {}".format(i))
# plt.legend()
# plt.savefig("./test_imgs/latency_{}.png".format(test_name))
# plt.clf()
#
# length = len(lat_records)
# plt.plot(range(len(lat_records[0])), lat_records[0], label = "episode:{}".format(1))
# plt.plot(range(len(lat_records[0])), lat_records[length//4], label = "episode:{}".format(length//4))
# plt.plot(range(len(lat_records[0])), lat_records[length*2//4], label = "episode:{}".format(length*2//4))
# plt.plot(range(len(lat_records[0])), lat_records[length*3//4], label = "episode:{}".format(length*3//4))
# plt.plot(range(len(lat_records[0])), lat_records[-1], label = "episode:{}".format(length))
# plt.legend()
# plt.savefig("./test_imgs/latency_{}_5_line.png".format(test_name))
# plt.clf()
#
# rewards_arr = np.array(rewards)
# lat_arr = np.array(lat_records)
# np.save("./test_imgs/rewards_{}.npy".format(test_name), rewards_arr)
# np.save("./test_imgs/latency_{}.npy".format(test_name), lat_arr)

#
#
# print("rewards: {}".format(len(rewards)))
# print(rewards)
#
# print("lat_records: {}".format(len(lat_records)))
# print(lat_records)
#
# print("action: {}".format(len(action_records)))
# print(action_records)

