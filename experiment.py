import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import time
import datetime
import itertools, json
import pdb


from env.network import StarNetwork, FullNetwork
from env.program import Program
from env.latency import evaluate
from env.utils import *

from placement_rl.placement_env import PlacementEnv
from placement_rl.placement_agent import PlacementAgent
from placement_rl.placeto_agent import PlaceToAgent

import os
import pickle

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

def networks_from_para(network_para, num_types):
    rt = []
    for para in network_para:
        networks = generate_networks(n_devices=para['num_of_devices'],
                                    type_probs=para['constraint_prob'],
                                   avg_speeds=para['compute_speed'],
                                   avg_bws=para['bw'],
                                   avg_delays=para['delay'],
                                   b_bws=para['beta_bw'],
                                   b_speeds=para['beta_speed'],
                                   num_type=num_types,
                                   seeds=para['seed'])
        for net_set in networks.values():
            for net_data in net_set:
                rt.append(FullNetwork(net_data['delay'], net_data['comm_speed'], net_data['speed'],
                                                      net_data['device_constraints']))
    return rt

def programs_from_para(prog_para, num_types):
    rt = []
    for para in prog_para:
        programs = generate_programs(alphas=para['alpha'],
                                    vs=para['v'],
                                    connect_probs=para['conn_prob'],
                                    avg_computes=para['compute'],
                                    avg_bytes=para['bytes'],
                                    b_comps=para['bete_compute'],
                                    b_comms=para['beta_byte'],
                                    num_types=num_types,
                                    seeds=para['seed'])

        for prog_set in programs.values():
            for G in prog_set:
                rt.append(Program(G))
    return rt

def generate_data(data_parameters):
    network_para_train = data_parameters['training']['networks']
    program_para_train = data_parameters['training']['programs']
    network_para_test = data_parameters['testing']['networks']
    program_para_test = data_parameters['testing']['programs']

    #generate train data
    train_networks = networks_from_para(network_para_train, data_parameters['num_of_types'])
    train_programs = programs_from_para(program_para_train, data_parameters['num_of_types'])

    # generate test data
    test_networks = networks_from_para(network_para_test, data_parameters['num_of_types'])
    test_programs = programs_from_para(program_para_test, data_parameters['num_of_types'])

    return train_networks, train_programs, test_networks, test_programs

def run_episodes(env,
                 agent,
                 program_ids,
                 network_ids,
                 seeds,
                 device,
                 use_full_graph=True,
                 use_placeto=False,
                 explore=True,
                 samples_to_ops_ratio=2,
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
        constraints = env.get_placement_constraints(program_id, network_id)

        num_of_samples = int( program.n_operators*  samples_to_ops_ratio)
        if use_placeto:
            mask = torch.ones(network.n_devices).to(device)

        elif use_full_graph:
            action_dict = env.full_graph_action_dict[program_id][network_id]
            node_dict = env.full_graph_node_dict[program_id][network_id]
            mask = torch.ones(len(action_dict)).to(device)


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

        new_episode = True

        start_time = time.time()
        for t in range(num_of_samples):
            if new_episode:
                ep_data = {}
                ep_data['actions'] = []
                ep_data['rewards'] = []
                ep_data['ep_return'] = 0

                cur_mapping = env.random_mapping(program_id, network_id, seed)

                last_latency, path, G_stats = env.simulate(program_id, network_id, cur_mapping, noise)
                new_episode = False

                g = None
                last_action=None

                if use_placeto:
                    random.seed()
                    placeto_order = list(range(program.n_operators))
                    random.shuffle(placeto_order)

            if use_placeto:
                s = placeto_order[t % program.n_operators]
                g = env.get_placeto_graph(program_id, network_id, cur_mapping, G_stats, s, placeto_order[:t% program.n_operators]).to(device)
                mask[:] = 1
                mask[constraints[s]] = 0
                action = agent.dev_selection(g, program, s, mask)

            elif use_full_graph:
                g = env.get_full_graph(program_id, network_id, cur_mapping, G_stats, device, path, False, last_g=g, last_action=last_action).to(device)
                if explore:
                    cur_nodes = [node_dict[o][cur_mapping[o]] for o in node_dict]
                    mask[:] = 1
                    mask[cur_nodes]=0
                    if len(ep_data['actions']) > 0:
                        last_op = ep_data['actions'][-1][0]
                        mask[list(node_dict[last_op].values())] = 0

                s, action = agent.op_dev_selection(g, action_dict, mask)
                last_action = [s, cur_mapping[s], action]

            else:
                # pdb.set_trace()
                g = env.get_cardinal_graph(program_id, network_id, cur_mapping, G_stats, path).to(device)
                s = agent.op_selection(g)
                action = agent.dev_selection_eft(program, network, cur_mapping, G_stats, s,
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
            if use_placeto:
                end_episode = (len(case_data['sampled_placement'])  == program.n_operators)

            if end_episode:
                agent.finish_episode(update_network=update_policy, use_baseline=True)
                case_data['episodes'].append(ep_data)
                new_episode = True
        case_data ['run_time'] = time.time() - start_time
        print(f'{i}th case: {[case_data["network_id"], case_data["program_id"]]}, {network.n_devices} devices, {program.n_operators} operators')
        print(f'\t\t Running time {case_data ["run_time"]/case_data["num_of_samples"]: .2f} s/sample, best latency: {min(case_data["latency_trace"]):.2f}')
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
        self.seed = exp_config.seed
        self.exp_cfg = exp_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not exp_config.cuda:
            self.device = torch.device('cpu')
        print(f"Running on {self.device}")

        if exp_config.load_dir is not None:
            self.exp_cfg = pickle.load(open(os.path.join(exp_config.load_dir, 'args.pkl'),'rb'))
            self.logdir = exp_config.load_dir
            self.train_networks, self.train_programs = pickle.load(open(os.path.join(self.logdir, "train_data.pkl"), "rb"))
            self.eval_networks, self.eval_programs = pickle.load(open(os.path.join(self.logdir, "eval_data.pkl"), "rb"))

            run_data = json.load(open(os.path.join(self.logdir, "run_data.txt"), "r"))
            self.train_sequence = run_data['train_sequence']
            self.eval_cases_network = [l[0] for l in run_data['eval_sequence']]
            self.eval_cases_program = [l[1] for l in run_data['eval_sequence']]
            self.eval_cases_init_mapping = [l[2] for l in run_data['eval_sequence']]

        else:
            self.logdir = os.path.join(
                self.exp_cfg.logdir, '{}_{}'.format(
                    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    self.exp_cfg.logdir_suffix))
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)
            pickle.dump(self.exp_cfg,
                        open(os.path.join(self.logdir, "args.pkl"), "wb"))

            self.init()

        print("LOGDIR: ", self.logdir)

        if not hasattr(self.exp_cfg, 'use_edgnn'):
            setattr(self.exp_cfg, 'use_edgnn', False)

        if self.exp_cfg.use_placeto:
            self.agent = PlaceToAgent(len(PlacementEnv.PLACETO_FEATURES),
                                  self.exp_cfg.output_dim,
                                   device=self.device,
                                  n_device = self.exp_cfg.data_parameters['training']['networks'][0]['num_of_devices'][0],
                                  k=self.exp_cfg.placeto_k,
                                  hidden_dim=self.exp_cfg.hidden_dim,
                                  lr=self.exp_cfg.lr,
                                  gamma=self.exp_cfg.gamma)

        else:
            self.agent = PlacementAgent(PlacementEnv.get_node_feature_dim(), PlacementEnv.get_edge_feature_dim(),
                                   self.exp_cfg.output_dim,
                                   device=self.device,
                                   hidden_dim=self.exp_cfg.hidden_dim, lr=self.exp_cfg.lr, gamma=self.exp_cfg.gamma,
                                   use_edgnn = self.exp_cfg.use_edgnn)

        if exp_config.load_dir:
            self.exp_cfg.load_dir = exp_config.load_dir
            self.exp_cfg.policy_model=exp_config.policy_model
            self.exp_cfg.embedding_model=exp_config.embedding_model
            self.agent.policy.load_state_dict(torch.load(os.path.join(self.logdir, exp_config.policy_model)))
            self.agent.embedding.load_state_dict(torch.load(os.path.join(self.logdir, exp_config.embedding_model)))

        self.train_env = PlacementEnv(self.train_networks, self.train_programs, self.exp_cfg.memory_capacity)
        self.eval_env = PlacementEnv(self.eval_networks, self.eval_programs, self.exp_cfg.memory_capacity)

        self.max_num_train_episodes = exp_config.max_num_training_episodes
        self.min_num_train_episodes = exp_config.min_num_training_episodes

        self.last_eval_latency = np.array([np.inf] * self.exp_cfg.num_of_eval_cases)



    def init(self):
        random.seed(self.seed)

        if self.exp_cfg.load_data:
            network_path = self.exp_cfg.device_net_path
            op_path = self.exp_cfg.op_net_path
            train_networks, train_programs, eval_networks, eval_programs, para_set = load_data_from_dir(network_path, op_path, self.exp_cfg)
        else:
            train_networks, train_programs, eval_networks, eval_programs = generate_data(self.exp_cfg.data_parameters)

        self.train_networks = train_networks
        self.train_programs = train_programs
        self.eval_networks = eval_networks
        self.eval_programs = eval_programs
        pickle.dump([train_networks, train_programs], open(os.path.join(self.logdir, "train_data.pkl"), "wb"))
        pickle.dump([eval_networks, eval_programs], open(os.path.join(self.logdir, "eval_data.pkl"), "wb"))

        self.train_sequence = []
        eval_cases = random.sample(list(itertools.product(range(len(eval_networks)), range(len(eval_programs)))), self.exp_cfg.num_of_eval_cases)
        self.eval_cases_network = [a[0] for a in eval_cases]
        self.eval_cases_program = [a[1] for a in eval_cases]
        self.eval_cases_init_mapping = np.random.choice(self.exp_cfg.data_parameters['testing']['init_mapping'], self.exp_cfg.num_of_eval_cases).tolist()

    def train(self):
        full_graph = not self.exp_cfg.use_op_selection
        record = []
        eval_records = []
        cnt = 0

        count_down = 5

        while len(self.train_sequence) < self.max_num_train_episodes:
            train_network_id = np.random.choice(len(self.train_networks), self.exp_cfg.eval_frequency).tolist()
            train_program_id = np.random.choice(len(self.train_programs), self.exp_cfg.eval_frequency).tolist()
            train_init_map = np.random.choice(self.exp_cfg.data_parameters['training']['init_mapping'], self.exp_cfg.eval_frequency).tolist()
            if self.exp_cfg.use_placeto:
                train_init_map = [-1] * self.exp_cfg.eval_frequency

            print('===========================================================================')
            print(f"RUNNING {cnt}th training batch. Max {self.max_num_train_episodes // self.exp_cfg.eval_frequency} batches. ")
            torch.save(self.agent.policy.state_dict(), os.path.join(self.logdir, f'policy_{len(self.train_sequence) }.pk'))
            torch.save(self.agent.embedding.state_dict(), os.path.join(self.logdir, f'embedding_{len(self.train_sequence)}.pk'))

            train_record = run_episodes(self.train_env,
                                        self.agent,
                                        train_program_id,
                                        train_network_id,
                                        train_init_map,
                                        device=self.device,
                                        use_placeto=self.exp_cfg.use_placeto,
                                        use_full_graph=full_graph,
                                        samples_to_ops_ratio=self.exp_cfg.samples_to_ops_ratio,
                                        update_policy=True,
                                        save_data=False,
                                        noise=self.exp_cfg.noise)
            record.append(train_record)
            self.train_sequence.extend(zip(train_network_id, train_program_id, train_init_map))
            cnt += 1
            if self.exp_cfg.eval:
                print(f"Evaluating. ")
                test_record = run_episodes(self.eval_env,
                                           self.agent,
                                           self.eval_cases_program,
                                           self.eval_cases_network,
                                           self.eval_cases_init_mapping,
                                           device=self.device,
                                           use_placeto=self.exp_cfg.use_placeto,
                                           use_full_graph=full_graph,
                                           samples_to_ops_ratio=self.exp_cfg.samples_to_ops_ratio,
                                           update_policy=False,
                                           save_data=False,
                                           noise=self.exp_cfg.noise)
                eval_records.append(test_record)
                eval_output = np.array([min(episode['latency_trace']) for episode in test_record])
                num_improved = np.sum(eval_output < self.last_eval_latency)
                print(f'{num_improved} test cases out of {len(eval_output)} are improved.')
                self.last_eval_latency = eval_output
                if len(self.train_sequence) <= self.min_num_train_episodes:
                    continue

                if num_improved < len(eval_output)//2:
                    count_down -= 1

                    if count_down == 0:
                        print("Performance stop being improved. End training.")
                        break


        run_data = {
            'num_of_train_networks': len(self.train_networks),
            'num_of_train_programs': len(self.train_programs),
            'num_of_train_episodes': len(self.train_sequence),
            'train_sequence': self.train_sequence,
            'num_of_eval_cases': self.exp_cfg.num_of_eval_cases,
            'eval_sequence': list(zip(self.eval_cases_network, self.eval_cases_program, self.eval_cases_init_mapping)),
            'data_para': self.exp_cfg.data_parameters
        }
        json.dump(run_data, open(os.path.join(self.logdir, "run_data.txt"), "w"), indent=4)

        torch.save(self.agent.policy.state_dict(),
                   os.path.join(self.logdir, f'policy_{len(self.train_sequence)}.pk'))
        torch.save(self.agent.embedding.state_dict(),
                   os.path.join(self.logdir, f'embedding_{len(self.train_sequence)}.pk'))

        pickle.dump(record, open(os.path.join(self.logdir, "train.pk"), "ab"))
        torch.save(self.agent.policy.state_dict(), os.path.join(self.logdir, f'policy.pk'))
        torch.save(self.agent.embedding.state_dict(), os.path.join(self.logdir, f'embedding.pk'))
        if self.exp_cfg.eval:
            pickle.dump(eval_records, open(os.path.join(self.logdir, "eval.pk"), "ab"))

        return record

    def test(self, para, max_num_of_tests, test_repeat, num_of_tune, noise):
        if para is None:
            try:
                test_networks, test_programs = pickle.load(open(os.path.join(self.logdir, 'eval_data.pkl'), 'rb'))
                para = json.load(open(os.path.join(self.logdir, 'run_data.txt'), 'r'))['data_para']
            except:
                raise ValueError('Nothing to test')
        else:
            test_networks = networks_from_para(para['testing']['networks'], para['num_of_types'])
            test_programs = programs_from_para(para['testing']['programs'], para['num_of_types'])

        logdir = os.path.join(self.logdir, 'test_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        print("TEST LOGDIR: ", logdir)

        pickle.dump([test_networks, test_programs], open(os.path.join(logdir, "test_data.pkl"), "wb"))
        self.eval_env = PlacementEnv(test_networks, test_programs, self.exp_cfg.memory_capacity)

        set = list(itertools.product(list(range(len(test_networks))), list(range(len(test_programs))), para['testing']['init_mapping']))
        np.random.seed(self.seed)
        np.random.shuffle(set)
        if len(set) > max_num_of_tests:
            set = set[:max_num_of_tests]
        self.test_program_ids = [l[1] for l in set]
        self.test_network_ids = [l[0] for l in set]
        self.test_init_seeds = [l[2] for l in set]

        run_data = {
            'policy_para': self.exp_cfg.policy_model,
            'embedding_para': self.exp_cfg.embedding_model,
            'num_of_test_cases': len(set),
            'num_of_repeat': test_repeat,
            'num_of_tune': num_of_tune,
            'noise': noise,
            'test_sequence': set,
            'data_para': para
        }
        json.dump(run_data, open(os.path.join(logdir, "run_data.txt"), "w"), indent=4)

        torch.save(self.agent.policy.state_dict(), os.path.join(logdir, f'policy.pk'))
        torch.save(self.agent.embedding.state_dict(), os.path.join(logdir, f'embedding.pk'))

        for seed, program_id, network_id, i in zip(self.test_init_seeds, self.test_program_ids, self.test_network_ids, range(len(set))):
            self.agent.policy.load_state_dict(torch.load(os.path.join(logdir, 'policy.pk')))
            self.agent.embedding.load_state_dict(torch.load(os.path.join(logdir, 'embedding.pk')))
            print('===========================================================================')
            print(f"RUNNING {self.exp_cfg.num_testing_cases_repeat} testing episodes for network {network_id}/program {program_id} ({i+1}/{len(set)}).")
            run_episodes(self.eval_env, self.agent,
                         [program_id] * test_repeat,
                         [network_id] * test_repeat,
                         [seed] * test_repeat,
                         device=self.device,
                         use_placeto=self.exp_cfg.use_placeto,
                         use_full_graph=not self.exp_cfg.use_op_selection,
                         explore=True,
                         samples_to_ops_ratio=self.exp_cfg.samples_to_ops_ratio,
                         update_policy=False,
                         save_data=True,
                         save_dir=logdir,
                         save_name=f'test_program_{program_id}_network_{network_id}_seed_{seed}',
                         noise=noise)

            if num_of_tune>0:
                print(f"RUNNING {num_of_tune} tuning episodes for network {network_id}/program {program_id}.")
                run_episodes(self.eval_env, self.agent,
                             [program_id] * num_of_tune,
                             [network_id] * num_of_tune,
                             [seed] * num_of_tune,
                             device=self.device,
                             use_placeto=self.exp_cfg.use_placeto,
                             use_full_graph=not self.exp_cfg.use_op_selection,
                             explore=True,
                             samples_to_ops_ratio=self.exp_cfg.samples_to_ops_ratio,
                             update_policy=True,
                             save_data=True,
                             save_dir=logdir,
                             save_name=f'tune_program_{program_id}_network_{network_id}_seed_{seed}',
                             noise=noise)

                print(f"RUNNING {self.exp_cfg.num_testing_cases_repeat} testing episodes for network {network_id}/program {program_id} after tuned.")
                run_episodes(self.eval_env, self.agent,
                             [program_id] * test_repeat,
                             [network_id] * test_repeat,
                             [seed] * test_repeat,
                             device=self.device,
                             use_placeto=self.exp_cfg.use_placeto,
                             use_full_graph=not self.exp_cfg.use_op_selection,
                             explore=True,
                             samples_to_ops_ratio=self.exp_cfg.samples_to_ops_ratio,
                             update_policy=False,
                             save_data=True,
                             save_dir=logdir,
                             save_name=f'test_program_{program_id}_network_{network_id}_seed_{seed}_tuned',
                             noise=noise)
