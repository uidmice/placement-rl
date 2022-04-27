import itertools
from itertools import product
import dgl
import torch

from env.latency import *


class PlacementEnv:
    NODE_FEATURES = ['compute', 'comp_rate', 'criticality', 'start_time_potential']
    EDGE_FEATURES = ['bytes', 'comm_delay', 'comm_rate', 'criticality']

    def __init__(self, networks: list, programs: list, seed=0):

        self.networks = networks
        self.programs = programs

        n_devices = [network.n_devices for network in networks]
        n_operators = [program.n_operators for program in programs]

        self.node_feature_mean = {'compute': torch.mean(torch.cat([p.op_compute for p in programs])),
                                  'comp_rate': torch.mean(torch.cat([n.comp_rate for n in networks]))}
        self.node_feature_std = {'compute': torch.std(torch.cat([p.op_compute for p in programs])),
                                 'comp_rate': torch.std(torch.cat([n.comp_rate for n in networks]))}

        self.edge_feature_mean = {'bytes': torch.mean(torch.cat([torch.flatten(p.data_bytes) for p in programs])),
                                  'comm_delay': torch.mean(torch.cat([torch.flatten(n.comm_delay) for n in networks])),
                                  'comm_rate': torch.mean(torch.cat([torch.flatten(n.comm_rate) for n in networks]))}

        self.edge_feature_std = {'bytes': torch.std(torch.cat([torch.flatten(p.data_bytes) for p in programs])),
                                 'comm_delay': torch.std(torch.cat([torch.flatten(n.comm_delay) for n in networks])),
                                 'comm_rate': torch.std(torch.cat([torch.flatten(n.comm_rate) for n in networks]))}
        self.seed = seed

        self.full_graph_node_dict = {}
        self.full_graph_action_dict = {}
        self.full_graph_node_static_feat = {}

        self.placement_constraints = {}

        self.cardinal_graph_node_static_feat = {}

    def get_parents(self, program_id, node, mapping):
        program = self.programs[program_id]
        device_set = [mapping[n] for n in program.op_parents[node]]
        return program.op_parents[node], device_set

    def get_children(self, program_id, node, mapping):
        program = self.programs[program_id]
        device_set = [mapping[n] for n in program.op_children[node]]
        return program.op_children[node], device_set

    def get_parallel(self, program_id, node, mapping):
        program = self.programs[program_id]
        device_set = [mapping[n] for n in program.op_parallel[node]]
        return program.op_parallel[node], device_set

    def get_placement_constraints(self, program_id, network_id):
        if program_id in self.placement_constraints:
            if network_id in self.placement_constraints[program_id]:
                return self.placement_constraints[program_id][network_id]
            else:
                self.placement_constraints[program_id][network_id] = []
        else:
            self.placement_constraints[program_id] = {}
            self.placement_constraints[program_id][network_id] = []
        constraints = self.placement_constraints[program_id][network_id]
        network = self.networks[network_id]
        program = self.programs[program_id]
        for n in program.P.nodes:
            c = program.P.nodes[n]['h_constraint']
            constraints.append([k[0] for k in filter(lambda elem: c in elem[1], network.device_constraints.items())])

        return constraints

    def random_mapping(self, program_id, network_id, seed=0):
        constraints = self.get_placement_constraints(program_id, network_id)
        np.random.seed(seed)
        mapping = [np.random.choice(constraints[i]) for i in range(self.programs[program_id].n_operators)]
        return mapping

    @staticmethod
    def get_node_feature_dim():
        return len(PlacementEnv.NODE_FEATURES)

    @staticmethod
    def get_edge_feature_dim():
        return len(PlacementEnv.EDGE_FEATURES)

    def get_node_feature(self, program_id, network_id, mapping, G_stats):
        try:
            program = self.programs[program_id]
        except:
            print(f'Program id {program_id} not valid')
            return
        try:
            network = self.networks[network_id]
        except:
            print(f'Network id {network_id} not valid')
            return

        feat = {'compute': program.op_compute,
                'comp_rate': network.comp_rate[mapping],
                # 'comp_time': torch.tensor([torch.mean(G_stats.nodes[o]['comp_time']) for o in range(program.n_operators)]),
                'criticality': torch.tensor([program.P.nodes[o]['criticality'] for o in range(program.n_operators)])}

        def est(op):
            e = {}
            parents = program.op_parents[op]
            if op == 0:
                return 0

            end_time = np.array([np.mean(G_stats.nodes[p]['end_time']) for p in parents])
            for dev in self.placement_constraints[program_id][network_id][op]:
                c_time = np.array([communication_latency(program, network, p, op, mapping[p], dev) for p in parents])
                e[dev] = np.max(c_time + end_time)
            return min(e.values()) - np.average(G_stats.nodes[op]['start_time'])

        feat['start_time_potential'] = torch.tensor([est(op) for op in range(program.n_operators)])

        self.node_feature_mean['start_time_potential'] = torch.mean(feat['start_time_potential'])
        self.node_feature_std['start_time_potential'] = torch.std(feat['start_time_potential'])

        for feature in self.node_feature_mean:
            feat[feature] = (feat[feature] - self.node_feature_mean[feature]) / (self.node_feature_std[feature] + 0.01)
        return feat

    def get_full_node_feature(self, program_id, network_id, mapping, G_stats):
        node_dict = self.full_graph_node_dict[program_id][network_id]
        program = self.programs[program_id]
        network = self.networks[network_id]

        def est(op, dev):
            parents = program.op_parents[op]
            if op == 0:
                return 0
            end_time = np.array([np.mean(G_stats.nodes[p]['end_time']) for p in parents])
            c_time = np.array([communication_latency(program, network, p, op, mapping[p], dev) for p in parents])
            est = np.max(c_time + end_time)
            return est - np.average(G_stats.nodes[op]['start_time'])

        op_comp = torch.zeros(sum([len(node_dict[a]) for a in node_dict]))
        comp_rate = torch.zeros_like(op_comp)
        criticality = torch.zeros_like(op_comp)
        e = torch.zeros_like(op_comp)

        for op in node_dict:
            op_comp[list(node_dict[op].values())] = program.op_compute[op]
            criticality[list(node_dict[op].values())] = program.P.nodes[op]['criticality']
            for d in node_dict[op]:
                comp_rate[node_dict[op][d]] = network.comp_rate[d]
                e[node_dict[op][d]] = est(op, d)

        feat = {'compute': op_comp,
                'comp_rate': comp_rate,
                # 'comp_time': torch.tensor([torch.mean(G_stats.nodes[o]['comp_time']) for o in range(program.n_operators)]),
                'criticality': criticality,
                'start_time_potential': e}

        self.node_feature_mean['start_time_potential'] = torch.mean(feat['start_time_potential'])
        self.node_feature_std['start_time_potential'] = torch.std(feat['start_time_potential'])

        for feature in self.node_feature_mean:
            feat[feature] = (feat[feature] - self.node_feature_mean[feature]) / (self.node_feature_std[feature] + 0.01)
        return feat

    def get_edge_feature(self, program_id, network_id, mapping, G_stats):
        try:
            program = self.programs[program_id]
        except:
            print(f'Program id {program_id} not valid')
            return
        try:
            network = self.networks[network_id]
        except:
            print(f'Network id {network_id} not valid')
            return

        n = program.P.number_of_edges()
        u = torch.zeros(n).int()
        v = torch.zeros(n).int()
        bytes = torch.zeros(n)
        comm_delay = torch.zeros(n)
        comm_rate = torch.zeros(n)
        criticality = torch.zeros(n)

        for i, line in enumerate(nx.generate_edgelist(program.P, data=False)):
            (e1, e2) = [int(s) for s in line.split(' ')]
            u[i] = e1
            v[i] = e2
            bytes[i] = program.get_data_bytes(e1, e2)
            comm_delay[i] = network.comm_delay[mapping[e1], mapping[e2]]
            comm_rate[i] = network.comm_rate[mapping[e1], mapping[e2]]
            criticality[i] = program.P.edges[e1, e2]['criticality']
        feat = {'bytes': bytes, 'comm_delay': comm_delay, 'comm_rate': comm_rate, 'criticality': criticality}
        for feature in self.edge_feature_mean:
            feat[feature] = (feat[feature] - self.edge_feature_mean[feature]) / self.edge_feature_std[feature]
        return u, v, feat

    def get_full_edge_feature(self, program_id, network_id, mapping, G_stats, bip_connection):
        program = self.programs[program_id]
        network = self.networks[network_id]
        node_dict = self.full_graph_node_dict[program_id][network_id]
        action_dict = self.full_graph_action_dict[program_id][network_id]

        u = []
        v = []
        bytes = []
        comm_delay = []
        comm_rate = []
        criticality = []

        if bip_connection:
            for n1, n2 in program.P.edges():
                edges = list(product(node_dict[n1].values(), node_dict[n2].values()))
                u.extend([e[0] for e in edges])
                v.extend([e[1] for e in edges])
                criticality.extend([program.P.edges[n1, n2]['criticality']]*len(edges))
                bytes.extend([program.get_data_bytes(n1, n2)]*len(edges))
                for d in edges:
                    _, d1 = action_dict[d[0]]
                    _, d2 = action_dict[d[1]]
                    comm_delay.append(network.comm_delay[d1, d2])
                    comm_rate.append(network.comm_rate[d1, d2])
        else:
            for op in node_dict:
                for dev in node_dict[op]:
                    node = node_dict[op][dev]
                    for op1, op2 in program.P.in_edges(op):
                        u.append(node_dict[op1][mapping[op1]])
                        v.append(node)
                        bytes.append(program.get_data_bytes(op1, op2))
                        comm_delay.append(network.comm_delay[mapping[op1], dev])
                        comm_rate.append(network.comm_rate[mapping[op1], dev])
                        criticality.append(program.P.edges[op1, op2]['criticality'])
                    for op1, op2 in program.P.out_edges(op):
                        v.append(node_dict[op2][mapping[op2]])
                        u.append(node)
                        bytes.append(program.get_data_bytes(op1, op2))
                        comm_delay.append(network.comm_delay[dev, mapping[op2]])
                        comm_rate.append(network.comm_rate[dev, mapping[op2]])
                        criticality.append(program.P.edges[op1, op2]['criticality'])

        u = torch.tensor(u)
        v = torch.tensor(v)
        bytes = torch.tensor(bytes)
        comm_delay = torch.tensor(comm_delay)
        comm_rate = torch.tensor(comm_rate)
        criticality = torch.tensor(criticality)

        feat = {'bytes': bytes, 'comm_delay': comm_delay, 'comm_rate': comm_rate, 'criticality': criticality}
        for feature in self.edge_feature_mean:
            feat[feature] = (feat[feature] - self.edge_feature_mean[feature]) / self.edge_feature_std[feature]
        return u, v, feat

    def init_full_graph(self, program_id, network_id):
        self.get_placement_constraints(program_id, network_id)
        if program_id in self.full_graph_node_dict:
            if network_id in self.full_graph_node_dict[program_id]:
                return
            else:
                self.full_graph_node_dict[program_id][network_id] = {}
                self.full_graph_action_dict[program_id][network_id] = {}
        else:
            self.full_graph_node_dict[program_id] = {}
            self.full_graph_node_dict[program_id][network_id] = {}
            self.full_graph_action_dict[program_id] = {}
            self.full_graph_action_dict[program_id][network_id] = {}

        node_dict = self.full_graph_node_dict[program_id][network_id]
        action_dict = self.full_graph_action_dict[program_id][network_id]
        program = self.programs[program_id]
        id = 0

        for n in program.P.nodes():
            node_dict[n] = {}
            for d in self.placement_constraints[program_id][network_id][n]:
                node_dict[n][d] = id
                action_dict[id] = (n, d)
                id += 1


    def get_full_graph(self, program_id, network_id, mapping, G_stats, critical_path=None, bip_connection=False):
        self.programs[program_id].update_criticality(critical_path)
        self.init_full_graph(program_id, network_id)

        node_features = self.get_full_node_feature(program_id, network_id, mapping, G_stats)
        u, v, edge_features = self.get_full_edge_feature(program_id, network_id, mapping, G_stats, bip_connection)

        g = dgl.graph((u, v))
        g.edata['x'] = torch.t(torch.stack([edge_features[feat] for feat in PlacementEnv.EDGE_FEATURES])).float()
        g.ndata['x'] = torch.t(torch.stack([node_features[feat] for feat in PlacementEnv.NODE_FEATURES])).float()
        return g

    def get_cardinal_graph(self, program_id, network_id, mapping, G_stats, critical_path=None):
        self.programs[program_id].update_criticality(critical_path)
        node_features = self.get_node_feature(program_id, network_id, mapping, G_stats)
        u, v, edge_features = self.get_edge_feature(program_id, network_id, mapping, G_stats)

        g = dgl.graph((u, v))
        g.edata['x'] = torch.t(torch.stack([edge_features[feat] for feat in PlacementEnv.EDGE_FEATURES])).float()
        g.ndata['x'] = torch.t(torch.stack([node_features[feat] for feat in PlacementEnv.NODE_FEATURES])).float()
        return g

    def evaluate(self, program_id, network_id, mapping, noise=0, repeat=1):
        l, path, G = self.simulate(program_id, network_id, mapping, noise, repeat)
        try:
            average_l = torch.mean(l)
        except:
            average_l = l
        return average_l

    def simulate(self, program_id, network_id, mapping, noise=0, repeat=1):
        program = self.programs[program_id]
        network = self.networks[network_id]
        G, l, path = simulate(mapping, program, network, noise, repeat)
        return l, path, G

    # def get_feature_device(self, node, device):
    #     feature_parent = np.zeros(self.network.get_edge_feature_dim())
    #     op, d = self.get_parents(node)
    #     if len(op)>0:
    #         a = np.array([self.program.get_relative_criticality(node, n) for n in op], dtype=np.float64)
    #         a /= np.sum(a)
    #         for i in range(len(op)):
    #             feature_parent += a[i] * self.network.get_edge_feature(d[i], device)
    #
    #     feature_child = np.zeros(self.network.get_edge_feature_dim())
    #     op, d = self.get_children(node)
    #     if len(op) > 0:
    #         a = np.array([self.program.get_relative_criticality(node, n) for n in op], dtype=np.float64)
    #         a /= np.sum(a)
    #         for i in range(len(op)):
    #             feature_child += a[i] * self.network.get_edge_feature(device, d[i])
    #
    #     feature_para = np.zeros(self.program.get_node_feature_dim())
    #     op, d = self.get_parallel(node)
    #     for i in range(len(d)):
    #         if d[i] == device:
    #             feature_para += self.program.get_node_feature(op[i])
    #
    #     return np.concatenate((feature_parent, feature_child, feature_para, self.network.get_node_feature(device)), axis=None)
    #
    # def get_state(self, node):
    #     state = [self.get_feature_device(node, d) for d in range(self.n_devices)]
    #     state.append(self.program.get_node_feature(node))
    #     return torch.from_numpy(np.concatenate(state)).float()
    #
    # def get_state_dim(self):
    #     return (2 * self.network.get_edge_feature_dim()
    #             + self.program.get_node_feature_dim()
    #             + self.network.get_node_feature_dim()) * self.n_devices + self.program.get_node_feature_dim()
    #
    #
    # def step(self, node, action):
    #     assert action in self.program.placement_constraints[node]
    #     self.mapping[node] = action
    #
    #     latency, cpath = evaluate_maxP(from_mapping_to_matrix(self.mapping, self.n_devices), self.program, self.network)
    #
    #     self.program.update_criticality(cpath)
    #     self.latency = latency
    #
    #     return latency
