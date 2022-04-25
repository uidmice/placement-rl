import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import time
import datetime
import itertools


from baseline import exhaustive, random_placement, heft, random_op_greedy_dev, random_op_est_dev
from env.utils import generate_program, generate_network
from env.network import StarNetwork
from env.program import Program
from env.latency import evaluate
from env.utils import *

from placement_rl.placement_env import PlacementEnv
from placement_rl.placement_agent import PlacementAgent

import os
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data_from_dir(network_path, program_path, exp_cfg, train_ratio=0.8):
    network_fns = os.listdir(network_path)
    # program_fns = os.listdir(program_path)

    program_fns = data_fn_filter(program_path,
                                 exp_cfg.v_range,
                                 exp_cfg.alpha_range,
                                 exp_cfg.seed_range,
                                 exp_cfg.ccr_range,
                                 exp_cfg.beta_range,
                                 exp_cfg.comm_range)

    # network_fn = np.random.choice(network_fns[0])
    network_fn = network_fns[0]
    network = load_pickle(os.path.join(network_path, network_fn))
    delay, bw, speed = network["delay"], network["bw"], network["speed"]
    delay = np.mean(delay, axis = 1)
    bw = np.mean(bw, axis = 1)
    network_constraints = network["device_constraints"]
    network = StarNetwork(delay, bw, speed)
    networks = [network]

    for i, li in enumerate(network_constraints):
        for j, element in enumerate(li):
            if isinstance(element, np.ndarray):
                li[j] = element[0]

    programs = []
    for program_fn in program_fns:
        G = load_dag(os.path.join(program_path, program_fn))

        op_constraints = []
        for n in G.nodes:
            G.nodes[n]["compute"] = float(G.nodes[n]["compute"])
            op_constraints.append(G.nodes[n]["op_constraints"])

        for e in G.edges:
            G.edges[e]["bytes"] = float(G.edges[e]["bytes"])

        constraints = []
        for op_constraint in op_constraints:
            constraints.append([])
            for i, device_constraint in enumerate(network_constraints):
                if op_constraint[0] in device_constraint:
                    constraints[-1].append(i)

        program = Program(G, constraints)
        programs.append(program)

    n_train = int(len(programs) * train_ratio)
    n_val = len(programs) - n_train

    env_train = PlacementEnv([network], programs[:n_train])
    env_val = PlacementEnv([network], programs[n_train + 1:])

    return env_train, env_val, networks, programs[:n_train]

def run_episodes(env,
                 agent,
                 program_ids,
                 network_ids,
                 seeds,
                 use_full_graph=True,
                 use_bip_connection=False,
                 multi_selection=False,
                 explore=True,
                 max_iter=50,
                 use_baseline=True,
                 update_policy=True,
                 save_data=False,
                 save_dir='data',
                 save_name='',
                 noise=0):

    assert isinstance(seeds, list)
    assert isinstance(program_ids, list)
    assert isinstance(network_ids, list)
    assert len(seeds) == len(program_ids) and len(seeds) == len(network_ids)


    currrent_stop_iter = max_iter

    if use_full_graph:
        for program_id, network_id in zip(program_ids, network_ids):
            env.init_full_graph(program_id, network_id)

    records = []

    for seed, program_id, network_id in zip(seeds, program_ids, network_ids):
        total_return = 0
        program = env.programs[program_id]
        network = env.networks[network_id]
        cur_mapping = program.random_mapping(seed)

        if use_full_graph:
            action_dict = env.full_graph_action_dict[program_id][network_id]
            node_dict = env.full_graph_node_dict[program_id][network_id]
            mask = torch.ones(len(action_dict)).to(device)
        else:
            mask = torch.ones(program.n_operators).to(device)
            mask[0] = mask[-1] = 0

        last_latency, path, G_stats = env.simulate(program_id, network_id, cur_mapping, noise)

        ep_latencies = [last_latency]
        ep_actions = []
        ep_rewards = []
        ep_data = {
            'network_id': network_id,
            'program_id': program_id,
            'init_seed': seed,
            'noise': noise,
            'n_iters': currrent_stop_iter
        }

        print(ep_data)

        start_time = time.time()
        for t in range(currrent_stop_iter):
            if use_full_graph:
                g = env.get_full_graph(program_id, network_id, cur_mapping, G_stats, path, use_bip_connection).to(device)

                if explore:
                    cur_nodes = [node_dict[o][cur_mapping[o]] for o in node_dict]
                    mask[:] = 1
                    mask[cur_nodes]=0

                if multi_selection:
                    action_map = agent.multi_op_dev_selection(g, node_dict)
                    cur_mapping = action_map
                    ep_actions.append(action_map)
                else:
                    s, action = agent.op_dev_selection(g, action_dict, mask)
                    cur_mapping[s] = action
                    ep_actions.append([s, action])
            else:
                g = env.get_cardinal_graph(program_id, network_id, cur_mapping, G_stats, path).to(device)
                s = agent.op_selection(g, mask)
                action = agent.dev_selection_est(program, network, cur_mapping, G_stats, s,
                                                 program.placement_constraints[s])
                cur_mapping[s] = action
                ep_actions.append([s, action])


            latency, path, G_stats = env.simulate(program_id, network_id, cur_mapping, noise)
            reward = (last_latency - latency) / 10
            last_latency = latency

            ep_latencies.append(latency)
            agent.saved_rewards.append(reward)
            ep_rewards.append(reward)
            total_return = reward + total_return * agent.gamma
        agent.finish_episode(update_network=update_policy, use_baseline=use_baseline)
        ep_data['run_time'] = time.time() - start_time
        ep_data['final_mapping'] = cur_mapping
        ep_data['ep_return'] = total_return
        ep_data['latency_trace'] = ep_latencies
        ep_data['actions'] = ep_actions
        ep_data['rewards'] = ep_rewards

        records.append(ep_data)

    if save_data:
        logname = os.path.join(
            save_dir, '{}.pk'.format(save_name))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pickle.dump(records, open(logname, 'wb'))
    return records

class Experiment:
    def __init__(self, exp_config):

        self.exp_cfg = exp_config

        self.logdir = os.path.join(
            self.exp_cfg.logdir, '{}_{}'.format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                self.exp_cfg.logdir_suffix))
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        print("LOGDIR: ", self.logdir)
        pickle.dump(self.exp_cfg,
                    open(os.path.join(self.logdir, "args.pkl"), "wb"))


        self.train_networks = [StarNetwork(*generate_network(n, seed=0)) for n in self.exp_cfg.num_devices_training]
        train_network_id = {i: {'n_devices': self.train_networks[i].n_devices, 'seed': 0} for i in range(len(self.train_networks))}

        self.train_programs = []
        train_program_id = {}
        idx = 0
        for m in self.exp_cfg.num_operators_training:
            for n in self.exp_cfg.num_devices_training:
                for seed in self.exp_cfg.application_graph_seeds_training:
                    self.train_programs.append(Program(*generate_program(m, n, seed=seed)))
                    train_program_id[idx] = {'n_operators': m, 'n_devices': n, 'seed': seed}
                    idx += 1


        self.train_env = PlacementEnv(self.train_networks, self.train_programs)

        episode_per_setting = self.exp_cfg.num_episodes_per_setting
        if not episode_per_setting:
            episode_per_setting = self.exp_cfg.num_training_episodes // len(self.train_networks) // len(self.train_programs) // len(self.exp_cfg.init_mapping_seeds_training)

        set = list(itertools.product(list(range(len(self.train_env.networks))), list(range(len(self.train_env.programs))), self.exp_cfg.init_mapping_seeds_training)) * episode_per_setting
        self.num_train_episodes = len(set)
        np.random.shuffle(set)

        self.train_program_ids = [l[1] for l in set]
        self.train_network_ids = [l[0] for l in set]
        self.train_init_seeds = [l[2] for l in set]

        self.test_networks = [StarNetwork(*generate_network(n, seed=0)) for n in self.exp_cfg.num_devices_testing]
        test_network_id = {i: {'n_devices': self.test_networks[i].n_devices, 'seed': 0} for i in range(len(self.test_networks))}

        self.test_programs = []
        test_program_id = {}
        idx = 0
        for m in self.exp_cfg.num_operators_testing:
            for n in self.exp_cfg.num_devices_testing:
                for seed in self.exp_cfg.application_graph_seeds_testing:
                    self.test_programs.append(Program(*generate_program(m, n, seed=seed)))
                    test_program_id[idx] = {'n_operators': m, 'n_devices': n, 'seed': seed}
                    idx += 1
        self.test_env = PlacementEnv(self.test_networks, self.test_programs)

        set = list(itertools.product(list(range(len(self.test_env.networks))), list(range(len(self.test_env.programs))), self.exp_cfg.init_mapping_seeds_testing))

        self.test_program_ids = [l[1] for l in set]
        self.test_network_ids = [l[0] for l in set]
        self.test_init_seeds = [l[2] for l in set]

        self.agent = PlacementAgent(self.train_env.get_node_feature_dim(), self.train_env.get_edge_feature_dim(), self.exp_cfg.output_dim,
                                    hidden_dim=self.exp_cfg.hidden_dim, lr=self.exp_cfg.lr, gamma=self.exp_cfg.gamma)

        pickle.dump({'test_program': test_program_id, 'test_network': test_network_id, 'train_program': train_program_id, 'train_network': train_network_id},
                    open(os.path.join(self.logdir, "data_id.pkl"), "wb"))

    def train(self):
        full_graph = not self.exp_cfg.use_op_selection
        if self.exp_cfg.eval:
            record = []
            eval_records = []
            for i in range(self.num_train_episodes//20):
                train_record = run_episodes(self.train_env,
                                  self.agent,
                                  self.train_program_ids[i*20: i*20 + 20],
                                  self.train_network_ids[i*20: i*20 + 20],
                                  self.train_init_seeds[i*20: i*20 + 20],
                                  use_full_graph=full_graph,
                                  max_iter = self.exp_cfg.max_iterations_per_episode,
                                  update_policy=True,
                                  save_data=False,
                                  noise=self.exp_cfg.noise)
                test_record = run_episodes(self.test_env,
                                            self.agent,
                                            self.test_program_ids * self.exp_cfg.testing_episodes,
                                            self.test_network_ids * self.exp_cfg.testing_episodes,
                                            self.test_init_seeds * self.exp_cfg.testing_episodes,
                                            use_full_graph=full_graph,
                                            max_iter=self.exp_cfg.max_iterations_per_episode,
                                            update_policy=False,
                                            save_data=False,
                                            noise=self.exp_cfg.noise)
                record.append(train_record)
                eval_records.append(test_record)

            pickle.dump(record, open(os.path.join(self.logdir, "train.pk"), "wb"))
            pickle.dump(eval_records, open(os.path.join(self.logdir, "eval.pk"), "wb"))

        else:
            record = run_episodes(self.train_env,
                                  self.agent,
                                  self.train_program_ids,
                                  self.train_network_ids,
                                  self.train_init_seeds,
                                  use_full_graph=full_graph,
                                  max_iter=self.exp_cfg.max_iterations_per_episode,
                                  update_policy=True,
                                  save_data=True,
                                  save_dir=self.logdir,
                                  save_name='train',
                                  noise=self.exp_cfg.noise)
        torch.save(self.agent.policy.state_dict(), os.path.join(self.logdir, 'policy.pk'))
        torch.save(self.agent.embedding.state_dict(), os.path.join(self.logdir, 'embedding.pk'))

        return record

    def test(self):
        for seed, program_id, network_id in zip(self.test_init_seeds, self.test_program_ids, self.test_network_ids):
            self.agent.policy.load_state_dict(torch.load(os.path.join(self.logdir, 'policy.pk')))
            self.agent.embedding.load_state_dict(torch.load(os.path.join(self.logdir, 'embedding.pk')))

            if self.exp_cfg.tuning_spisodes:
                run_episodes(self.test_env, self.agent,
                             [program_id] * self.exp_cfg.tuning_spisodes,
                             [network_id] * self.exp_cfg.tuning_spisodes,
                             [seed] * self.exp_cfg.tuning_spisodes,
                             use_full_graph=not self.exp_cfg.use_op_selection,
                             use_bip_connection=False,
                             explore=True,
                             max_iter=self.exp_cfg.max_iterations_per_episode,
                             update_policy=True,
                             save_data=True,
                             save_dir=self.logdir,
                             save_name=f'tune_program_{program_id}_network_{network_id}_seed_{seed}',
                             noise=self.exp_cfg.noise)

            run_episodes(self.test_env, self.agent,
                         [program_id] * self.exp_cfg.testing_episodes,
                         [network_id] * self.exp_cfg.testing_episodes,
                             [seed] * self.exp_cfg.testing_episodes,
                             use_full_graph=not self.exp_cfg.use_op_selection,
                             use_bip_connection=False,
                             explore=True,
                             max_iter=self.exp_cfg.max_iterations_per_episode,
                             update_policy=False,
                             save_data=True,
                             save_dir=self.logdir,
                             save_name=f'test_explore_program_{program_id}_network_{network_id}_seed_{seed}',
                             noise=self.exp_cfg.noise)

            run_episodes(self.test_env, self.agent,
                         [program_id] * self.exp_cfg.testing_episodes,
                         [network_id] * self.exp_cfg.testing_episodes,
                             [seed] * self.exp_cfg.testing_episodes,
                             use_full_graph=not self.exp_cfg.use_op_selection,
                             use_bip_connection=False,
                             explore=False,
                             max_iter=self.exp_cfg.max_iterations_per_episode,
                             update_policy=False,
                             save_data=True,
                             save_dir=self.logdir,
                             save_name=f'test_noexp_program_{program_id}_network_{network_id}_seed_{seed}',
                             noise=self.exp_cfg.noise)



class Experiment_on_data:
    def __init__(self, exp_config):

        self.exp_cfg = exp_config

        self.logdir = os.path.join(
            self.exp_cfg.logdir, '{}_{}'.format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                self.exp_cfg.logdir_suffix))
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        print("LOGDIR: ", self.logdir)
        pickle.dump(self.exp_cfg,
                    open(os.path.join(self.logdir, "args.pkl"), "wb"))


        network_path = self.exp_cfg.device_net_path
        op_path = self.exp_cfg.op_net_path

        env_train, env_val, networks, programs = load_data_from_dir(network_path, op_path, self.exp_cfg, train_ratio=0.9)
        self.train_networks = networks
        self.train_programs = programs

        train_network_id = {i: {'n_devices': self.train_networks[i].n_devices, 'seed': 0} for i in range(len(self.train_networks))}

        train_program_id = {}

        for idx in range(len(self.train_programs)):
            n_operators = self.train_programs[idx].n_operators
            n_devices = self.train_networks[0].n_devices
            seed = 0
            train_program_id[idx] = {"n_operators":n_operators, "n_devices":n_devices, "seed":0}

        self.train_env = env_train

        episode_per_setting = self.exp_cfg.num_episodes_per_setting
        if not episode_per_setting:
            episode_per_setting = self.exp_cfg.num_training_episodes // len(self.train_networks) // len(self.train_programs) // len(self.exp_cfg.init_mapping_seeds_training)

        set = list(itertools.product(list(range(len(self.train_env.networks))), list(range(len(self.train_env.programs))), self.exp_cfg.init_mapping_seeds_training)) * episode_per_setting
        self.num_train_episodes = len(set)
        np.random.shuffle(set)

        self.train_program_ids = [l[1] for l in set]
        self.train_network_ids = [l[0] for l in set]
        self.train_init_seeds = [l[2] for l in set]

        self.test_networks = [StarNetwork(*generate_network(n, seed=0)) for n in self.exp_cfg.num_devices_testing]
        test_network_id = {i: {'n_devices': self.test_networks[i].n_devices, 'seed': 0} for i in range(len(self.test_networks))}

        self.test_programs = []
        test_program_id = {}
        idx = 0
        for m in self.exp_cfg.num_operators_testing:
            for n in self.exp_cfg.num_devices_testing:
                for seed in self.exp_cfg.application_graph_seeds_testing:
                    self.test_programs.append(Program(*generate_program(m, n, seed=seed)))
                    test_program_id[idx] = {'n_operators': m, 'n_devices': n, 'seed': 0}
                    idx += 1
        self.test_env = PlacementEnv(self.test_networks, self.test_programs)

        set = list(itertools.product(list(range(len(self.test_env.networks))), list(range(len(self.test_env.programs))), self.exp_cfg.init_mapping_seeds_testing))

        self.test_program_ids = [l[1] for l in set]
        self.test_network_ids = [l[0] for l in set]
        self.test_init_seeds = [l[2] for l in set]

        self.agent = PlacementAgent(self.train_env.get_node_feature_dim(), self.train_env.get_edge_feature_dim(), self.exp_cfg.output_dim,
                                    hidden_dim=self.exp_cfg.hidden_dim, lr=self.exp_cfg.lr, gamma=self.exp_cfg.gamma)

        pickle.dump({'test_program': test_program_id, 'test_network': test_network_id, 'train_program': train_program_id, 'train_network': train_network_id},
                    open(os.path.join(self.logdir, "data_id.pkl"), "wb"))

    def train(self):
        full_graph = not self.exp_cfg.use_op_selection
        if self.exp_cfg.eval:
            record = []
            eval_records = []
            for i in range(self.num_train_episodes//20):
                train_record = run_episodes(self.train_env,
                                  self.agent,
                                  self.train_program_ids[i*20: i*20 + 20],
                                  self.train_network_ids[i*20: i*20 + 20],
                                  self.train_init_seeds[i*20: i*20 + 20],
                                  use_full_graph=full_graph,
                                  max_iter = self.exp_cfg.max_iterations_per_episode,
                                  update_policy=True,
                                  save_data=False,
                                  noise=self.exp_cfg.noise)
                test_record = run_episodes(self.test_env,
                                            self.agent,
                                            self.test_program_ids * self.exp_cfg.testing_episodes,
                                            self.test_network_ids * self.exp_cfg.testing_episodes,
                                            self.test_init_seeds * self.exp_cfg.testing_episodes,
                                            use_full_graph=full_graph,
                                            max_iter=self.exp_cfg.max_iterations_per_episode,
                                            update_policy=False,
                                            save_data=False,
                                            noise=self.exp_cfg.noise)
                record.append(train_record)
                eval_records.append(test_record)

            pickle.dump(record, open(os.path.join(self.logdir, "train.pk"), "wb"))
            pickle.dump(eval_records, open(os.path.join(self.logdir, "eval.pk"), "wb"))

        else:
            record = run_episodes(self.train_env,
                                  self.agent,
                                  self.train_program_ids,
                                  self.train_network_ids,
                                  self.train_init_seeds,
                                  use_full_graph=full_graph,
                                  max_iter=self.exp_cfg.max_iterations_per_episode,
                                  update_policy=True,
                                  save_data=True,
                                  save_dir=self.logdir,
                                  save_name='train',
                                  noise=self.exp_cfg.noise)
        torch.save(self.agent.policy.state_dict(), os.path.join(self.logdir, 'policy.pk'))
        torch.save(self.agent.embedding.state_dict(), os.path.join(self.logdir, 'embedding.pk'))

        return record

    def test(self):
        for seed, program_id, network_id in zip(self.test_init_seeds, self.test_program_ids, self.test_network_ids):
            self.agent.policy.load_state_dict(torch.load(os.path.join(self.logdir, 'policy.pk')))
            self.agent.embedding.load_state_dict(torch.load(os.path.join(self.logdir, 'embedding.pk')))

            if self.exp_cfg.tuning_spisodes:
                run_episodes(self.test_env, self.agent,
                             [program_id] * self.exp_cfg.tuning_spisodes,
                             [network_id] * self.exp_cfg.tuning_spisodes,
                             [seed] * self.exp_cfg.tuning_spisodes,
                             use_full_graph=not self.exp_cfg.use_op_selection,
                             use_bip_connection=False,
                             explore=True,
                             max_iter=self.exp_cfg.max_iterations_per_episode,
                             update_policy=True,
                             save_data=True,
                             save_dir=self.logdir,
                             save_name=f'tune_program_{program_id}_network_{network_id}_seed_{seed}',
                             noise=self.exp_cfg.noise)

            run_episodes(self.test_env, self.agent,
                         [program_id] * self.exp_cfg.testing_episodes,
                         [network_id] * self.exp_cfg.testing_episodes,
                             [seed] * self.exp_cfg.testing_episodes,
                             use_full_graph=not self.exp_cfg.use_op_selection,
                             use_bip_connection=False,
                             explore=True,
                             max_iter=self.exp_cfg.max_iterations_per_episode,
                             update_policy=False,
                             save_data=True,
                             save_dir=self.logdir,
                             save_name=f'test_explore_program_{program_id}_network_{network_id}_seed_{seed}',
                             noise=self.exp_cfg.noise)

            run_episodes(self.test_env, self.agent,
                         [program_id] * self.exp_cfg.testing_episodes,
                         [network_id] * self.exp_cfg.testing_episodes,
                             [seed] * self.exp_cfg.testing_episodes,
                             use_full_graph=not self.exp_cfg.use_op_selection,
                             use_bip_connection=False,
                             explore=False,
                             max_iter=self.exp_cfg.max_iterations_per_episode,
                             update_policy=False,
                             save_data=True,
                             save_dir=self.logdir,
                             save_name=f'test_noexp_program_{program_id}_network_{network_id}_seed_{seed}',
                             noise=self.exp_cfg.noise)


# def train(env,
#           agent,
#           init_mapping,
#           episodes,
#           max_iter=50,
#           update_op_net=True,
#           update_dev_net=True,
#           greedy_dev_selection=True,
#           use_baseline=True,
#           noise=0,
#           short_earlier_ep=False,
#           early_stop = 5):
#     op_rewards = []
#
#     map_records = [init_mapping]
#     lat_records = []
#     act_records = []
#
#     program = env.programs[0]
#     network = env.networks[0]
#
#     mask_dev = torch.zeros(network.n_devices).to(device)
#
#     currrent_stop_iter = max_iter
#     if short_earlier_ep:
#         currrent_stop_iter = early_stop
#
#
#
#     for i in range(episodes):
#         cur_mapping = init_mapping.copy()
#         last_latency, path, G_stats = env.simulate(0, 0, init_mapping, noise)
#
#         print(f'=== Episode {i} ===')
#         latencies = [last_latency]
#         actions = []
#         ep_reward = 0
#
#         mask_op = torch.ones(program.n_operators).to(device)
#         mask_op[0] = mask_op[-1] = 0
#
#         for t in range(currrent_stop_iter):
#             graphs = []
#             temp_mapping = cur_mapping.copy()
#             g = env.get_cardinal_graph(0, 0, temp_mapping, path, G_stats).to(device)
#             s = agent.op_selection(g, mask_op)
#
#             if greedy_dev_selection:
#                 action = agent.dev_selection_est(program, network, cur_mapping, G_stats, s, program.placement_constraints[s])
#             else:
#                 parallel = program.op_parallel[s]
#                 constraints = program.placement_constraints[s]
#                 mask_dev[:] = 0
#                 mask_dev[constraints] = 1
#                 for d in range(n_devices):
#                     temp_mapping[s] = d
#                     t_g = env.get_cardinal_graph(temp_mapping).to(device)
#                     graphs.append(t_g)
#                 action = agent.dev_selection(graphs, s, parallel, mask=mask_dev)
#
#             cur_mapping[s] = action
#             latency, path, G_stats = env.simulate(0, 0, cur_mapping, noise)
#             # reward = -latency/10
#             # reward = -np.sqrt(latency)/10
#             reward = (last_latency - latency)/10
#             last_latency = latency
#
#             latencies.append(latency)
#             actions.append([s, action])
#             agent.saved_rewards.append(reward)
#             ep_reward = reward + ep_reward * agent.gamma
#
#         agent.finish_episode(update_op_net, update_dev_net, use_baseline)
#         op_rewards.append(ep_reward)
#         map_records.append(cur_mapping)
#         lat_records.append(latencies)
#         act_records.append(actions)
#
#
#         if short_earlier_ep and len(lat_records) >= 2:
#             last_lat = lat_records[-2]
#             incr_flag = 0
#             for i in range(len(last_lat)):
#                 if last_lat[i] < latencies[i]:
#                     incr_flag = 0
#                     break
#                 incr_flag = 1
#             if incr_flag:
#                 currrent_stop_iter += 1
#
#
#     return op_rewards, lat_records, act_records, map_records
