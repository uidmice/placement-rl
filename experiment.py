import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import time
import datetime
import itertools, json
import pdb


from baseline import exhaustive, random_placement, heft, random_op_greedy_dev, random_op_est_dev
from env.network import StarNetwork, FullNetwork
from env.program import Program
from env.latency import evaluate
from env.utils import *

from placement_rl.placement_env import PlacementEnv
from placement_rl.placement_agent import PlacementAgent
from placement_rl.placeto_agent import PlaceToAgent

import os
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data_from_dir(network_path, program_path, exp_cfg):
    network_para_train = exp_cfg.data_parameters['training']['networks']
    program_para_train = exp_cfg.data_parameters['training']['programs']
    network_para_test = exp_cfg.data_parameters['testing']['networks']
    program_para_test = exp_cfg.data_parameters['testing']['programs']

    #load train data
    train_networks = []
    train_network_para = {}
    idx = 0
    for para in network_para_train:
        network_fns_train = network_fn_filter(network_path,
                                    n_devices=para['num_of_devices'],
                                    type_probs=para['constraint_prob'],
                                   avg_speeds=para['compute_speed'],
                                   avg_bws=para['bw'],
                                   avg_delays=para['delay'],
                                   b_bws=para['beta_bw'],
                                   b_speeds=para['beta_speed'],
                                   num_types=[exp_cfg.data_parameters['num_of_types']])
        for fn in network_fns_train:
            networks = load_pickle(os.path.join(network_path, fn))
            for seed in para['seed']:
                net_data = networks[seed]
                train_networks.append(FullNetwork(net_data['delay'], net_data['comm_speed'], net_data['speed'],
                                                  net_data['device_constraints']))
                train_network_para[idx] = net_data['para']
                idx += 1


    train_programs = []
    train_program_para = {}
    idx = 0
    for para in program_para_train:
        program_fns_train = program_fn_filter(program_path,
                                          vs=para['v'],
                                          alphas=para['alpha'],
                                          connect_probs=para['conn_prob'],
                                          avg_computes=para['compute'],
                                          avg_bytes=para['bytes'],
                                          b_comps=para['bete_compute'],
                                          b_comms=para['beta_byte'],
                                          num_types=[exp_cfg.data_parameters['num_of_types']])




        for fn in program_fns_train:
            programs = load_pickle(os.path.join(program_path, fn))
            for seed in para['seed']:
                G = programs[seed]
                train_programs.append(Program(G))
                train_program_para[idx] = G.graph
                idx += 1


    # load test data
    test_networks = []
    test_network_para = {}
    idx = 0
    for para in network_para_test:
        network_fns_train = network_fn_filter(network_path,
                                              n_devices=para['num_of_devices'],
                                              type_probs=para['constraint_prob'],
                                              avg_speeds=para['compute_speed'],
                                              avg_bws=para['bw'],
                                              avg_delays=para['delay'],
                                              b_bws=para['beta_bw'],
                                              b_speeds=para['beta_speed'],
                                              num_types=[exp_cfg.data_parameters['num_of_types']])
        for fn in network_fns_train:
            networks = load_pickle(os.path.join(network_path, fn))
            for seed in para['seed']:
                net_data = networks[seed]
                train_networks.append(FullNetwork(net_data['delay'], net_data['comm_speed'], net_data['speed'],
                                                  net_data['device_constraints']))
                train_network_para[idx] = net_data['para']
                idx += 1

    test_programs = []
    test_program_para = {}
    idx = 0
    for para in program_para_test:
        program_fns_train = program_fn_filter(program_path,
                                              vs=para['v'],
                                              alphas=para['alpha'],
                                              connect_probs=para['conn_prob'],
                                              avg_computes=para['compute'],
                                              avg_bytes=para['bytes'],
                                              b_comps=para['bete_compute'],
                                              b_comms=para['beta_byte'],
                                              num_types=[exp_cfg.data_parameters['num_of_types']])

        for fn in program_fns_train:
            programs = load_pickle(os.path.join(program_path, fn))
            for seed in para['seed']:
                G = programs[seed]
                train_programs.append(Program(G))
                train_program_para[idx] = G.graph
                idx += 1

    para_set = {'test_program': test_program_para,
                'test_network': test_network_para,
                'train_program': train_program_para,
                'train_network': train_network_para}
    return train_networks, train_programs, test_networks, test_programs, para_set

def generate_data(exp_cfg):
    network_para_train = exp_cfg.data_parameters['training']['networks']
    program_para_train = exp_cfg.data_parameters['training']['programs']
    network_para_test = exp_cfg.data_parameters['testing']['networks']
    program_para_test = exp_cfg.data_parameters['testing']['programs']

    #generate train data
    train_networks = []
    train_network_para = {}
    idx = 0
    for para in network_para_train:
        networks = generate_networks(n_devices=para['num_of_devices'],
                                    type_probs=para['constraint_prob'],
                                   avg_speeds=para['compute_speed'],
                                   avg_bws=para['bw'],
                                   avg_delays=para['delay'],
                                   b_bws=para['beta_bw'],
                                   b_speeds=para['beta_speed'],
                                   num_type=exp_cfg.data_parameters['num_of_types'],
                                   seeds=para['seed'])
        for net_set in networks.values():
            for net_data in net_set:
                train_networks.append(FullNetwork(net_data['delay'], net_data['comm_speed'], net_data['speed'],
                                                      net_data['device_constraints']))
                train_network_para[idx] = net_data['para']
                idx += 1


    train_programs = []
    train_program_para = {}
    idx = 0
    for para in program_para_train:
        programs = generate_programs(alphas=para['alpha'],
                                    vs=para['v'],
                                    connect_probs=para['conn_prob'],
                                    avg_computes=para['compute'],
                                    avg_bytes=para['bytes'],
                                    b_comps=para['bete_compute'],
                                    b_comms=para['beta_byte'],
                                    num_types=exp_cfg.data_parameters['num_of_types'],
                                    seeds=para['seed'])

        for prog_set in programs.values():
            for G in prog_set:
                train_programs.append(Program(G))
                train_program_para[idx] = G.graph
                idx += 1


    # generate test data
    test_networks = []
    test_network_para = {}
    idx = 0
    for para in network_para_test:
        networks = generate_networks(n_devices=para['num_of_devices'],
                                     type_probs=para['constraint_prob'],
                                     avg_speeds=para['compute_speed'],
                                     avg_bws=para['bw'],
                                     avg_delays=para['delay'],
                                     b_bws=para['beta_bw'],
                                     b_speeds=para['beta_speed'],
                                     num_type=exp_cfg.data_parameters['num_of_types'],
                                     seeds=para['seed'])
        for net_set in networks.values():
            for net_data in net_set:
                test_networks.append(FullNetwork(net_data['delay'], net_data['comm_speed'], net_data['speed'],
                                                  net_data['device_constraints']))
                test_network_para[idx] = net_data['para']
                idx += 1

    test_programs = []
    test_program_para = {}
    idx = 0
    for para in program_para_test:
        programs = generate_programs(alphas=para['alpha'],
                                     vs=para['v'],
                                     connect_probs=para['conn_prob'],
                                     avg_computes=para['compute'],
                                     avg_bytes=para['bytes'],
                                     b_comps=para['bete_compute'],
                                     b_comms=para['beta_byte'],
                                     num_types=exp_cfg.data_parameters['num_of_types'],
                                     seeds=para['seed'])

        for prog_set in programs.values():
            for G in prog_set:
                test_programs.append(Program(G))
                test_program_para[idx] = G.graph
                idx += 1

    para_set = {'test_program': test_program_para,
                'test_network': test_network_para,
                'train_program': train_program_para,
                'train_network': train_network_para}
    return train_networks, train_programs, test_networks, test_programs, para_set

def run_episodes(env,
                 agent,
                 program_ids,
                 network_ids,
                 seeds,
                 use_full_graph=True,
                 use_bip_connection=False,
                 use_memory_buffer=False,
                 memory_rollback_prob=0.6,
                 burn_in_steps=5,
                 multi_selection=False,
                 explore=True,
                 samples_to_ops_ratio=1.5,
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

    records = []

    for seed, program_id, network_id, i in zip(seeds, program_ids, network_ids, range(len(seeds))):
        if use_full_graph:
            env.init_full_graph(program_id, network_id)
        program = env.programs[program_id]
        network = env.networks[network_id]

        num_of_samples = int( program.n_operators*  samples_to_ops_ratio)

        constraints = env.get_placement_constraints(program_id, network_id)

        if use_full_graph:
            action_dict = env.full_graph_action_dict[program_id][network_id]
            node_dict = env.full_graph_node_dict[program_id][network_id]
            mask = torch.ones(len(action_dict)).to(device)
        else:
            mask = torch.ones(program.n_operators).to(device)
            mask[0] = mask[-1] = 0

        case_data = {
            'network_id': network_id,
            'program_id': program_id,
            'init_seed': seed,
            'noise': noise,
            'num_of_samples': num_of_samples,
            'episodes': [],
            'sampled_placement': [],
            'latency_trace': []
        }

        print(f'{i}th case: {case_data }')

        new_episode = True

        env.clear_buffer(program_id, network_id)

        start_time = time.time()
        for t in range(num_of_samples):
            if new_episode:
                ep_data = {}
                ep_data['actions'] = []
                ep_data['rewards'] = []
                ep_data['ep_return'] = 0

                cur_mapping = None
                if use_memory_buffer:
                    if np.random.random() < memory_rollback_prob:
                        cur_mapping = env.sample_from_buffer(program_id, network_id)
                    count = 0

                if cur_mapping is None:
                    cur_mapping = env.random_mapping(program_id, network_id, seed)
                last_latency, path, G_stats = env.simulate(program_id, network_id, cur_mapping, noise)
                env.push_to_buffer(program_id, network_id, cur_mapping.copy(), last_latency, True)

                new_episode = False

            if use_full_graph:
                g = env.get_full_graph(program_id, network_id, cur_mapping, G_stats, path, use_bip_connection).to(device)
                if explore:
                    cur_nodes = [node_dict[o][cur_mapping[o]] for o in node_dict]
                    mask[:] = 1
                    mask[cur_nodes]=0
                    if len(ep_data['actions']) > 0:
                        last_op = ep_data['actions'][-1][0]
                        mask[list(node_dict[last_op].values())] = 0

                if multi_selection:
                    action_map = agent.multi_op_dev_selection(g, node_dict)
                    cur_mapping = action_map
                    ep_data['actions'].append(action_map)
                else:
                    s, action = agent.op_dev_selection(g, action_dict, mask)
                    cur_mapping[s] = action
                    ep_data['actions'].append([s, action])
            else:
                # pdb.set_trace()
                g = env.get_cardinal_graph(program_id, network_id, cur_mapping, G_stats, path).to(device)
                s = agent.op_selection(g, mask)
                action = agent.dev_selection_est(program, network, cur_mapping, G_stats, s,
                                                 constraints[s])
                cur_mapping[s] = action
                ep_data['actions'].append([s, action])

            latency, path, G_stats = env.simulate(program_id, network_id, cur_mapping, noise)
            reward = (last_latency - latency) / 10
            last_latency = latency

            case_data['sampled_placement'].append(cur_mapping.copy())
            case_data['latency_trace'].append(latency)
            ep_data['rewards'].append(reward)
            agent.saved_rewards.append(reward)
            ep_data['ep_return'] = reward + ep_data['ep_return'] * agent.gamma

            end_episode = (len(case_data['sampled_placement']) == num_of_samples)
            if use_memory_buffer:
                idx = env.push_to_last_buffer(program_id, network_id, cur_mapping.copy(), latency, 0)
                if idx > -1:
                    count = 0
                else:
                    count += 1

                if count >= burn_in_steps:
                    end_episode = True

            if end_episode:
                agent.finish_episode(update_network=update_policy, use_baseline=use_baseline)
                case_data['episodes'].append(ep_data)
                new_episode = True
        case_data ['run_time'] = time.time() - start_time
        records.append(case_data)

    if save_data:
        logname = os.path.join(
            save_dir, '{}.pk'.format(save_name))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pickle.dump(records, open(logname, 'wb'))
    return records


def run_placeto_episodes(env,
                 agent,
                 program_ids,
                 network_ids,
                 seeds,
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

    records = []

    for seed, program_id, network_id, i in zip(seeds, program_ids, network_ids, range(len(seeds))):
        program = env.programs[program_id]
        network = env.networks[network_id]

        case_data = {
            'network_id': network_id,
            'program_id': program_id,
            'init_seed': seed,
            'noise': noise,
            'num_of_samples': program.n_operators,
            'episodes': [],
            'sampled_placement': [],
            'latency_trace': []
        }

        print(f'{i}th case: {case_data }')

        new_episode = True

        start_time = time.time()
        for t in range(program.n_operators):
            if new_episode:
                ep_data = {}
                ep_data['actions'] = []
                ep_data['rewards'] = []
                ep_data['ep_return'] = 0

                cur_mapping = env.random_mapping(program_id, network_id, seed)
                last_latency, path, G_stats = env.simulate(program_id, network_id, cur_mapping, noise)
                new_episode = False



            g = env.get_cardinal_graph(program_id, network_id, cur_mapping, G_stats, path).to(device)
            s, end = agent.op_selection(g)
            action = agent.dev_selection(g,
                                         program,
                                         network,
                                         s)
            cur_mapping[s] = action
            ep_data['actions'].append([s, action])

            latency, path, G_stats = env.simulate(program_id, network_id, cur_mapping, noise)
            reward = (last_latency - latency) / 10
            print("reward at {} is {}".format(t,reward))
            last_latency = latency

            case_data['sampled_placement'].append(cur_mapping.copy())
            case_data['latency_trace'].append(latency)
            ep_data['rewards'].append(reward)
            agent.saved_rewards.append(reward)
            ep_data['ep_return'] = reward + ep_data['ep_return'] * agent.gamma

            end_episode = (t == program.n_operators - 1)
            if end_episode:
                agent.finish_episode(update_network=update_policy, use_baseline=use_baseline)
                case_data['episodes'].append(ep_data)
                new_episode = True

        case_data ['run_time'] = time.time() - start_time
        records.append(case_data)

    if save_data:
        logname = os.path.join(
            save_dir, '{}.pk'.format(save_name))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pickle.dump(records, open(logname, 'wb'))
    return records



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

        if exp_config.load_data:
            network_path = self.exp_cfg.device_net_path
            op_path = self.exp_cfg.op_net_path
            train_networks, train_programs, test_networks, test_programs, para_set = load_data_from_dir(network_path, op_path, exp_config)
        else:
            train_networks, train_programs, test_networks, test_programs, para_set = generate_data(exp_config)

        self.train_networks = train_networks
        self.train_programs = train_programs
        self.test_networks = test_networks
        self.test_programs = test_programs
        pickle.dump(para_set, open(os.path.join(self.logdir, "data_para.pkl"), "wb"))

        self.train_env = PlacementEnv(self.train_networks, self.train_programs, exp_config.memory_capacity)
        self.test_env = PlacementEnv(self.test_networks, self.test_programs, exp_config.memory_capacity)


        episode_per_setting = self.exp_cfg.num_episodes_per_setting
        random_train = self.exp_cfg.random_training_pair
        self.num_train_episodes = self.exp_cfg.num_training_episodes
        if not episode_per_setting:
            episode_per_setting = self.exp_cfg.num_training_episodes // (
                        len(self.train_networks) * len(self.train_programs) * len(
                    self.exp_cfg.data_parameters['training']['init_mapping']))
            if not episode_per_setting:
                random_train = True
            else:
                self.num_train_episodes = episode_per_setting * (len(self.train_networks) * len(self.train_programs) * len(
                    self.exp_cfg.data_parameters['training']['init_mapping']))
        else:
            self.num_train_episodes = episode_per_setting * (len(self.train_networks) * len(self.train_programs) * len(
                    self.exp_cfg.data_parameters['training']['init_mapping']))


        if random_train:
            self.train_network_ids = np.random.choice(len(self.train_networks), self.num_train_episodes).tolist()
            self.train_program_ids = np.random.choice(len(self.train_programs), self.num_train_episodes).tolist()
            self.train_init_seeds = np.random.choice(exp_config.data_parameters['training']['init_mapping'], self.num_train_episodes).tolist()
        else:
            set = list(itertools.product(list(range(len(self.train_networks))), list(range(len(self.train_programs))), self.exp_cfg.data_parameters['training']['init_mapping'])) * episode_per_setting
            np.random.shuffle(set)
            self.train_network_ids = [l[0] for l in set]
            self.train_program_ids = [l[1] for l in set]
            self.train_init_seeds = [l[2] for l in set]


        set = list(itertools.product(list(range(len(self.test_networks))), list(range(len(self.test_programs))), self.exp_cfg.data_parameters['testing']['init_mapping']))
        self.test_program_ids = [l[1] for l in set]
        self.test_network_ids = [l[0] for l in set]
        self.test_init_seeds = [l[2] for l in set]

        self.agent = PlacementAgent(PlacementEnv.get_node_feature_dim(), PlacementEnv.get_edge_feature_dim(), self.exp_cfg.output_dim,
                                    hidden_dim=self.exp_cfg.hidden_dim, lr=self.exp_cfg.lr, gamma=self.exp_cfg.gamma)
        run_data = {
            'num_of_train_networks': len(self.train_networks),
            'num_of_train_programs': len(self.train_programs),
            'num_of_train_episodes': self.num_train_episodes,
            'train_sequence': {
                'networks': self.train_network_ids,
                'programs': self.train_program_ids,
                'initial_mapping': self.train_init_seeds
            },
            'num_of_test_networks': len(self.test_networks),
            'num_of_test_programs': len(self.test_programs),
            'num_of_test_episodes': self.exp_cfg.num_testing_episodes,
            'test_sequence': {
                'networks': self.test_network_ids,
                'programs': self.test_program_ids,
                'initial_mapping': self.test_init_seeds
            },
            'data_para': self.exp_cfg.data_parameters
        }
        json.dump(run_data, open(os.path.join(self.logdir, "run_data.txt"), "w"), indent=4)


    def train(self):
        full_graph = not self.exp_cfg.use_op_selection
        record = []
        eval_records = []
        for i in range(self.num_train_episodes // 20):
            print('===========================================================================')
            print(f"{i}th: RUNNING training batch. Total {self.num_train_episodes // 20} batches. ")
            torch.save(self.agent.policy.state_dict(), os.path.join(self.logdir, f'policy_{i * 20 }.pk'))
            torch.save(self.agent.embedding.state_dict(), os.path.join(self.logdir, f'embedding_{i * 20}.pk'))

            train_record = run_episodes(self.train_env,
                                        self.agent,
                                        self.train_program_ids[i * 20: i * 20 + 20],
                                        self.train_network_ids[i * 20: i * 20 + 20],
                                        self.train_init_seeds[i * 20: i * 20 + 20],
                                        use_full_graph=full_graph,
                                        samples_to_ops_ratio=self.exp_cfg.samples_to_ops_ratio,
                                        update_policy=True,
                                        save_data=False,
                                        noise=self.exp_cfg.noise)
            record.append(train_record)
            if self.exp_cfg.eval:
                print(f"{i}th: Evaluating. ")
                test_record = run_episodes(self.test_env,
                                           self.agent,
                                           self.test_program_ids * self.exp_cfg.num_testing_episodes,
                                           self.test_network_ids * self.exp_cfg.num_testing_episodes,
                                           self.test_init_seeds * self.exp_cfg.num_testing_episodes,
                                           use_full_graph=full_graph,
                                           samples_to_ops_ratio=self.exp_cfg.samples_to_ops_ratio,
                                           update_policy=False,
                                           save_data=False,
                                           noise=self.exp_cfg.noise)
                eval_records.append(test_record)

        pickle.dump(record, open(os.path.join(self.logdir, "train.pk"), "wb"))
        torch.save(self.agent.policy.state_dict(), os.path.join(self.logdir, f'policy.pk'))
        torch.save(self.agent.embedding.state_dict(), os.path.join(self.logdir, f'embedding.pk'))
        if self.exp_cfg.eval:
            pickle.dump(eval_records, open(os.path.join(self.logdir, "eval.pk"), "wb"))
        return record

    def test(self):
        for seed, program_id, network_id in zip(self.test_init_seeds, self.test_program_ids, self.test_network_ids):
            self.agent.policy.load_state_dict(torch.load(os.path.join(self.logdir, 'policy.pk')))
            self.agent.embedding.load_state_dict(torch.load(os.path.join(self.logdir, 'embedding.pk')))

            if self.exp_cfg.num_tuning_episodes:
                print('===========================================================================')
                print(f"RUNNING {self.exp_cfg.num_tuning_episodes} tuning episodes for network {network_id}/program {program_id}.")
                run_episodes(self.test_env, self.agent,
                             [program_id] * self.exp_cfg.num_tuning_episodes,
                             [network_id] * self.exp_cfg.num_tuning_episodes,
                             [seed] * self.exp_cfg.num_tuning_episodes,
                             use_full_graph=not self.exp_cfg.use_op_selection,
                             use_bip_connection=False,
                             explore=True,
                             samples_to_ops_ratio=self.exp_cfg.samples_to_ops_ratio,
                             update_policy=True,
                             save_data=True,
                             save_dir=self.logdir,
                             save_name=f'tune_program_{program_id}_network_{network_id}_seed_{seed}',
                             noise=self.exp_cfg.noise)
            print('===========================================================================')
            print(f"RUNNING {self.exp_cfg.num_testing_episodes} testing episodes for network {network_id}/program {program_id}.")
            run_episodes(self.test_env, self.agent,
                         [program_id] * self.exp_cfg.num_testing_episodes,
                         [network_id] * self.exp_cfg.num_testing_episodes,
                             [seed] * self.exp_cfg.num_testing_episodes,
                             use_full_graph=not self.exp_cfg.use_op_selection,
                             use_bip_connection=False,
                             explore=True,
                             samples_to_ops_ratio=self.exp_cfg.samples_to_ops_ratio,
                             update_policy=False,
                             save_data=True,
                             save_dir=self.logdir,
                             save_name=f'test_explore_program_{program_id}_network_{network_id}_seed_{seed}',
                             noise=self.exp_cfg.noise)

            # run_episodes(self.test_env, self.agent,
            #              [program_id] * self.exp_cfg.num_testing_episodes,
            #              [network_id] * self.exp_cfg.num_testing_episodes,
            #                  [seed] * self.exp_cfg.num_testing_episodes,
            #                  use_full_graph=not self.exp_cfg.use_op_selection,
            #                  use_bip_connection=False,
            #                  explore=False,
            #                  samples_to_ops_ratio=self.exp_cfg.samples_to_ops_ratio,
            #                  update_policy=False,
            #                  save_data=True,
            #                  save_dir=self.logdir,
            #                  save_name=f'test_noexp_program_{program_id}_network_{network_id}_seed_{seed}',
            #                  noise=self.exp_cfg.noise)



class Experiment_placeto:
    def __init__(self, exp_config):
        self.exp_cfg = exp_config
        self.logdir = os.path.join(
            self.exp_cfg.logdir, 'placeto_{}_{}'.format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                self.exp_cfg.logdir_suffix))
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        print("LOGDIR: ", self.logdir)
        pickle.dump(self.exp_cfg,
                    open(os.path.join(self.logdir, "args.pkl"), "wb"))

        if exp_config.load_data:
            network_path = self.exp_cfg.device_net_path
            op_path = self.exp_cfg.op_net_path
            train_networks, train_programs, test_networks, test_programs, para_set = load_data_from_dir(network_path, op_path, exp_config)
        else:
            train_networks, train_programs, test_networks, test_programs, para_set = generate_data(exp_config)

        self.train_networks = train_networks
        self.train_programs = train_programs
        self.test_networks = test_networks
        self.test_programs = test_programs
        pickle.dump(para_set, open(os.path.join(self.logdir, "data_para.pkl"), "wb"))

        self.train_env = PlacementEnv(self.train_networks, self.train_programs, exp_config.memory_capacity)
        self.test_env = PlacementEnv(self.test_networks, self.test_programs, exp_config.memory_capacity)


        episode_per_setting = self.exp_cfg.num_episodes_per_setting
        random_train = self.exp_cfg.random_training_pair
        self.num_train_episodes = self.exp_cfg.num_training_episodes
        if not episode_per_setting:
            episode_per_setting = self.exp_cfg.num_training_episodes // (
                        len(self.train_networks) * len(self.train_programs) * len(
                    self.exp_cfg.data_parameters['training']['init_mapping']))
            if not episode_per_setting:
                random_train = True
            else:
                self.num_train_episodes = episode_per_setting * (len(self.train_networks) * len(self.train_programs) * len(
                    self.exp_cfg.data_parameters['training']['init_mapping']))
        else:
            self.num_train_episodes = episode_per_setting * (len(self.train_networks) * len(self.train_programs) * len(
                    self.exp_cfg.data_parameters['training']['init_mapping']))


        if random_train:
            self.train_network_ids = np.random.choice(len(self.train_networks), self.num_train_episodes).tolist()
            self.train_program_ids = np.random.choice(len(self.train_programs), self.num_train_episodes).tolist()
            self.train_init_seeds = np.random.choice(exp_config.data_parameters['training']['init_mapping'], self.num_train_episodes).tolist()
        else:
            set = list(itertools.product(list(range(len(self.train_networks))), list(range(len(self.train_programs))), self.exp_cfg.data_parameters['training']['init_mapping'])) * episode_per_setting
            np.random.shuffle(set)
            self.train_network_ids = [l[0] for l in set]
            self.train_program_ids = [l[1] for l in set]
            self.train_init_seeds = [l[2] for l in set]


        set = list(itertools.product(list(range(len(self.test_networks))), list(range(len(self.test_programs))), self.exp_cfg.data_parameters['testing']['init_mapping']))
        self.test_program_ids = [l[1] for l in set]
        self.test_network_ids = [l[0] for l in set]
        self.test_init_seeds = [l[2] for l in set]

        self.agent = PlaceToAgent(PlacementEnv.get_node_feature_dim(),
                                  PlacementEnv.get_edge_feature_dim(),
                                  self.exp_cfg.output_dim,
                                  n_device = self.train_networks[0].n_devices,
                                  hidden_dim=self.exp_cfg.hidden_dim,
                                  lr=self.exp_cfg.lr,
                                  gamma=self.exp_cfg.gamma)
        run_data = {
            'num_of_train_networks': len(self.train_networks),
            'num_of_train_programs': len(self.train_programs),
            'num_of_train_episodes': self.num_train_episodes,
            'train_sequence': {
                'networks': self.train_network_ids,
                'programs': self.train_program_ids,
                'initial_mapping': self.train_init_seeds
            },
            'num_of_test_networks': len(self.test_networks),
            'num_of_test_programs': len(self.test_programs),
            'num_of_test_episodes': self.exp_cfg.num_testing_episodes,
            'test_sequence': {
                'networks': self.test_network_ids,
                'programs': self.test_program_ids,
                'initial_mapping': self.test_init_seeds
            },
            'data_para': self.exp_cfg.data_parameters
        }
        json.dump(run_data, open(os.path.join(self.logdir, "run_data.txt"), "w"), indent=4)

    def train(self):
        if self.exp_cfg.eval:
            record = []
            eval_records = []
            for i in range(self.num_train_episodes//20):
                print('===========================================================================')
                print(f"{i}th: RUNNING training batch. Total {self.num_train_episodes//20} batches. ")
                train_record = run_placeto_episodes(self.train_env,
                                  self.agent,
                                  self.train_program_ids[i*20: i*20 + 20],
                                  self.train_network_ids[i*20: i*20 + 20],
                                  self.train_init_seeds[i*20: i*20 + 20],
                                  update_policy=True,
                                  save_data=False,
                                  noise=self.exp_cfg.noise)
                print(f"{i}th: Evaluating. ")
                test_record = run_placeto_episodes(self.test_env,
                                            self.agent,
                                            self.test_program_ids * self.exp_cfg.num_testing_episodes,
                                            self.test_network_ids * self.exp_cfg.num_testing_episodes,
                                            self.test_init_seeds * self.exp_cfg.num_testing_episodes,
                                            update_policy=False,
                                            save_data=False,
                                            noise=self.exp_cfg.noise)
                record.append(train_record)
                eval_records.append(test_record)

            pickle.dump(record, open(os.path.join(self.logdir, "train.pk"), "wb"))
            pickle.dump(eval_records, open(os.path.join(self.logdir, "eval.pk"), "wb"))

        else:
            record = run_placeto_episodes(self.train_env,
                                  self.agent,
                                  self.train_program_ids,
                                  self.train_network_ids,
                                  self.train_init_seeds,
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

            if self.exp_cfg.num_tuning_episodes:
                print('===========================================================================')
                print(f"RUNNING {self.exp_cfg.num_tuning_episodes} tuning episodes for network {network_id}/program {program_id}.")
                run_episodes(self.test_env, self.agent,
                             [program_id] * self.exp_cfg.num_tuning_episodes,
                             [network_id] * self.exp_cfg.num_tuning_episodes,
                             [seed] * self.exp_cfg.num_tuning_episodes,
                             use_full_graph=not self.exp_cfg.use_op_selection,
                             use_bip_connection=False,
                             explore=True,
                             samples_to_ops_ratio=self.exp_cfg.samples_to_ops_ratio,
                             update_policy=True,
                             save_data=True,
                             save_dir=self.logdir,
                             save_name=f'tune_program_{program_id}_network_{network_id}_seed_{seed}',
                             noise=self.exp_cfg.noise)
            print('===========================================================================')
            print(f"RUNNING {self.exp_cfg.num_testing_episodes} testing episodes for network {network_id}/program {program_id}.")
            run_episodes(self.test_env, self.agent,
                         [program_id] * self.exp_cfg.num_testing_episodes,
                         [network_id] * self.exp_cfg.num_testing_episodes,
                             [seed] * self.exp_cfg.num_testing_episodes,
                             use_full_graph=not self.exp_cfg.use_op_selection,
                             use_bip_connection=False,
                             explore=True,
                             samples_to_ops_ratio=self.exp_cfg.samples_to_ops_ratio,
                             update_policy=False,
                             save_data=True,
                             save_dir=self.logdir,
                             save_name=f'test_explore_program_{program_id}_network_{network_id}_seed_{seed}',
                             noise=self.exp_cfg.noise)
