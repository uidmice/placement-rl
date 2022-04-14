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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


n_devices = [10, 20, 30]
n_operators = [10, 20, 30]
networks = {n: StarNetwork(*generate_network(n, seed=0)) for n in n_devices}
num_programs = 10
programs = {}
for n in n_devices:
    programs[n] = {}
    for m in n_operators:
        programs[n][m] = []
        for seed in range(num_programs):
            DAG, constraints = generate_program(m, n, seed=seed)
            programs[n][m].append(Program(DAG, constraints))

output_dim = 10

env = PlacementEnv([networks[20]], programs[20][10])

agent = PlacementAgent(env.get_node_feature_dim(), env.get_edge_feature_dim(), output_dim, output_dim)
num_iterations = 50
episode_per_program=20


def run_episodes(env,
                 agent,
                 program_id,
                 network_id,
                 seeds,
                 episode_per_seed=1,
                 shuffle_episodes=False,
                 explore=True,
                 max_iter=50,
                 use_baseline=True,
                 update_op_policy=False,
                 update_policy=True,
                 noise=0):

    if isinstance(seeds, int):
        seeds = [seeds]
    assert isinstance(seeds, list)

    seeds = seeds * episode_per_seed
    if shuffle_episodes:
        np.random.shuffle(seeds)

    program = env.programs[program_id]

    currrent_stop_iter = max_iter

    env.init_full_graph(program_id, network_id)

    action_dict = env.full_graph_action_dict[program_id][network_id]
    node_dict = env.full_graph_node_dict[program_id][network_id]
    mask = torch.ones(len(action_dict)).to(device)


    map_records = []
    return_records = []
    reward_records = []
    lat_records = []
    act_records = []

    for seed in seeds:
        total_return = 0
        cur_mapping = program.random_mapping(seed)

        last_latency, path, G_stats = env.simulate(program_id, network_id, cur_mapping, noise)

        ep_latencies = [last_latency]
        ep_actions = []
        ep_rewards = []


        for t in range(currrent_stop_iter):
            g = env.get_full_graph(program_id, network_id, cur_mapping, G_stats, path).to(device)

            if explore:
                cur_nodes = [node_dict[o][cur_mapping[o]] for o in node_dict]
                mask[:] = 1
                mask[cur_nodes]=0

            s, action = agent.op_dev_selection(g, action_dict, mask)
            cur_mapping[s] = action
            latency, path, G_stats = env.simulate(program_id, network_id, cur_mapping, noise)
            reward = (last_latency - latency) / 10
            last_latency = latency

            ep_latencies.append(latency)
            ep_actions.append([s, action])
            agent.saved_rewards.append(reward)
            ep_rewards.append(reward)
            total_return = reward + total_return * agent.gamma
        map_records.append(cur_mapping)
        return_records.append(total_return)
        lat_records.append(ep_latencies)
        act_records.append(ep_actions)
        reward_records.append(ep_rewards)

        agent.finish_episode(update_op_network=update_op_policy, update_full_network=update_policy, use_baseline=use_baseline)

    return lat_records, reward_records, act_records, map_records, return_records

data = list(range(num_programs)) * episode_per_program
np.random.shuffle(data)

episode_records = {}
for i in range(num_programs):
    episode_records[i] = []
for i, prog_id in enumerate(data):
    print([i, prog_id])
    lat_records, reward_records, act_records, map_records, return_records = run_episodes(env, agent, prog_id, 0, 0, max_iter=num_iterations)
    episode_records[prog_id].append([lat_records[0], reward_records[0], act_records[0], map_records, return_records])

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

