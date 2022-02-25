import numpy as np
import torch
import matplotlib.pyplot as plt
import random

from baseline import exhaustive, random_placement, heft, random_op_greedy_dev
from env.utils import generate_program, generate_network
from env.network import StarNetwork
from env.program import Program

from placement_rl.placement_env import PlacementEnv
from placement_rl.placement_agent import PlacementAgent

import os
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


n_devices = 30
n_operators = 20
seed = 0
network = StarNetwork(*generate_network(n_devices, seed=seed))
DAG, constraints = generate_program(n_operators, n_devices, seed=seed)
program = Program(DAG, constraints)
env = PlacementEnv(network, program)
agent = PlacementAgent(env.get_node_feature_dim(),
                       env.get_edge_feature_dim(),
                       10, 10)
mapping = program.random_mapping()
print(program.placement_constraints)

# m_matrix = np.zeros((n_operators, n_devices))
# m_matrix[0, program.pinned[0]] = 1
# m_matrix[-1, program.pinned[1]] = 1
# min_mapping, min_L, solution = exhaustive(m_matrix, program, network)

num_iter = 50
num_epo = 20



def train(env,
          agent,
          init_mapping,
          episodes,
          place_once=True,
          max_iter=50,
          update_op_net=True,
          update_dev_net=True,
          greedy_dev_selection=False,
          use_baseline=True,
          noise=0,
          early_stop = 5):
    op_rewards = []

    map_records = [init_mapping]
    lat_records = []
    act_records = []

    mask_dev = torch.zeros(env.n_devices).to(device)

    currrent_stop_iter = early_stop

    if place_once:
        max_iter = env.n_operators - 2

    for i in range(episodes):
        cur_mapping = init_mapping.copy()
        last_latency, _ = env.evaluate(init_mapping, noise)

        print(f'=== Episode {i} ===')
        latencies = [last_latency]
        actions = []
        ep_reward = 0

        mask_op = torch.ones(env.n_operators).to(device)
        mask_op[0] = mask_op[-1] = 0

        for t in range(currrent_stop_iter):
            graphs = []
            temp_mapping = cur_mapping.copy()
            g = env.get_placement_graph(temp_mapping).to(device)
            s = agent.op_selection(g, mask_op)
            if place_once:
                mask_op[s] = 0

            if greedy_dev_selection:
                action = agent.dev_selection_greedy(cur_mapping, s, env.program.placement_constraints[s], env, noise)
                action = random.choice(action)
            else:
                parallel = env.op_parallel[s]
                constraints = env.program.placement_constraints[s]
                mask_dev[:] = 0
                mask_dev[constraints] = 1
                for d in range(n_devices):
                    temp_mapping[s] = d
                    t_g = env.get_placement_graph(temp_mapping).to(device)
                    graphs.append(t_g)
                action = agent.dev_selection(graphs, s, parallel, mask=mask_dev)

            cur_mapping[s] = action
            latency, _ = env.evaluate(cur_mapping, noise)
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

        if len(lat_records) >= 2:
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


fig, axs = plt.subplots(2, 2, sharey=True, sharex=True)
noises = [0, 5, 10, 20]
for j in range(len(noises)):
    noise = noises[j]

    #calculating baselines

    # random samples
    sample_map, sample_latency, sample_latencies = random_placement(env, num_iter * 3, noise)
    print(f"Random Sample:\t\t\t {sample_latency:.2f}, mapping: {sample_map}, evaluate latency: {env.evaluate(sample_map, noise)}")

    # HEFT
    heft_map, heft_lat = heft(env, noise)
    print(f"HEFT:\t\t\t\t\t {heft_lat:.2f}, mapping: {heft_map}, evaluate latency: {env.evaluate(heft_map, noise)}")

    # Random Op Selection
    random_traces = []
    random_lat_best = np.Inf
    random_map_best = None
    random_trace_best = None
    for i in range(num_epo):
        random_mapping, random_latency, random_latency_trace = random_op_greedy_dev(env, mapping, num_iter, noise)
        random_traces.append(random_latency_trace)
        if random_latency < random_lat_best:
            random_lat_best = random_latency
            random_map_best = random_mapping
            random_trace_best = random_latency_trace
    print(f"Random Op Selection:\t {random_lat_best:.2f}, mapping: {random_map_best}, evaluate latency: {env.evaluate(random_map_best, noise)}")

    rewards, lat_records, action_records, map_records = train(env, agent, mapping, num_epo,  place_once=False, max_iter=num_iter, update_op_net=True, update_dev_net=False, greedy_dev_selection=True, use_baseline=True, noise=noise)
    best_records = np.argmin([lat[-1] for lat in lat_records])

    print(f"GNN+RL:\t\t\t\t\t {lat_records[best_records][-1]:.2f}, mapping: {map_records[best_records+1]}, evaluate latency: {env.evaluate(map_records[best_records+1], noise)}")

    x = j //2
    y = j % 2
    axs[x, y].plot(range(num_iter+1), np.average(np.array(lat_records), axis=0), label = "GNN+RL - best".format(num_epo), color='blue', linewidth=2)
    axs[x, y].fill_between(range(num_iter+1), np.average(np.array(lat_records), axis=0) + np.std(np.array(lat_records), axis=0), np.average(np.array(lat_records), axis=0) - np.std(np.array(lat_records), axis=0), alpha=.25, color='blue')

    axs[x, y].plot(range(num_iter+1), np.average(np.array(random_traces), axis=0), label=f'Random Op Selection - best', color='orange', linewidth=2)
    axs[x, y].fill_between(range(num_iter+1), np.average(np.array(random_traces), axis=0) + np.std(np.array(random_traces), axis=0), np.average(np.array(random_traces), axis=0) - np.std(np.array(random_traces), axis=0), alpha=.25, color='orange')

    axs[x, y].plot(range(num_iter+1), torch.ones(num_iter+1) * sample_latency, label=f'Best Sample', color='green', linewidth=2)

    axs[x, y].plot(range(num_iter+1), torch.ones(num_iter+1) * heft_lat, label='HEFT', color='red', linewidth=2)


    axs[x, y].set_title(f'Noise {noise}')
handles, labels = axs[x, y].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', fontsize=8)
fig.suptitle(f'{n_devices} devices, {n_operators} ops, seed {seed}', fontsize=20)
plt.show()

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

