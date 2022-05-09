import itertools
from itertools import product
import dgl
import torch

from env.latency import *
from placement_rl.memory_buffer import Buffer


class PlacementEnv:
    NODE_FEATURES = ['compute', 'comp_rate', 'comp_time', 'criticality', 'start_time_potential']
    EDGE_FEATURES = ['bytes', 'comm_delay', 'comm_time', 'comm_rate', 'criticality']
    PLACETO_FEATURES = ['comp_time', 'output_size', 'cur_placement', 'is_cur', 'is_done']

    def __init__(self, networks: list, programs: list, memory_size=5):

        self.networks = networks
        self.programs = programs

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

        self.full_graph_node_dict = {}
        self.full_graph_action_dict = {}
        self.full_graph_node_static_feat = {}

        self.placement_constraints = {}

        self.placement_buffer = {}
        self.memory_size = memory_size

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

    def get_memory_buffer(self, program_id, network_id):
        if program_id not in self.placement_buffer:
            self.placement_buffer[program_id] = {}
            self.placement_buffer[program_id][network_id] = Buffer(capacity=self.memory_size)
        else:
            if network_id not in self.placement_buffer[program_id]:
                self.placement_buffer[program_id][network_id] = Buffer(capacity=self.memory_size)
        return self.placement_buffer[program_id][network_id]

    def push_to_buffer(self, program_id, network_id, mapping, latency, force):
        buffer = self.get_memory_buffer(program_id, network_id)
        return buffer.push(mapping, latency, force)

    def sample_from_buffer(self, program_id, network_id):
        buffer = self.get_memory_buffer(program_id, network_id)
        return buffer.sample()

    def clear_buffer(self, program_id, network_id):
        buffer = self.get_memory_buffer(program_id, network_id)
        buffer.clear()

    def push_to_last_buffer(self, program_id, network_id, mapping, latency, p):
        buffer = self.get_memory_buffer(program_id, network_id)
        return buffer.push_to_last(mapping, latency, p)

    def random_mapping(self, program_id, network_id, seed=-1):
        constraints = self.get_placement_constraints(program_id, network_id)
        if isinstance(seed, list):
            s = np.random.choice(seed)
            np.random.seed(s)
        elif isinstance(seed, int) and seed > -1:
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
                'comp_time': torch.tensor([np.mean(G_stats.nodes[o]['comp_time']) for o in range(program.n_operators)]),
                'criticality': torch.tensor([program.P.nodes[o]['criticality'] for o in range(program.n_operators)])}
        self.node_feature_mean['comp_time'] = torch.mean(feat['comp_time'])
        self.node_feature_std['comp_time'] = torch.std(feat['comp_time'])
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
        comp_time = torch.zeros_like(op_comp)
        criticality = torch.zeros_like(op_comp)
        e = torch.zeros_like(op_comp)

        for op in node_dict:
            op_comp[list(node_dict[op].values())] = program.op_compute[op]
            criticality[list(node_dict[op].values())] = program.P.nodes[op]['criticality']
            for d in node_dict[op]:
                comp_rate[node_dict[op][d]] = network.comp_rate[d]
                comp_time[node_dict[op][d]] = computation_latency(program, network, op, d)
                e[node_dict[op][d]] = est(op, d)

        feat = {'compute': op_comp,
                'comp_rate': comp_rate,
                'comp_time': comp_time,
                'criticality': criticality,
                'start_time_potential': e}

        self.node_feature_mean['comp_time'] = torch.mean(feat['comp_time'])
        self.node_feature_std['comp_time'] = torch.std(feat['comp_time'])
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
        comm_time = torch.zeros(n)
        criticality = torch.zeros(n)

        for i, line in enumerate(nx.generate_edgelist(program.P, data=False)):
            (e1, e2) = [int(s) for s in line.split(' ')]
            u[i] = e1
            v[i] = e2
            bytes[i] = program.get_data_bytes(e1, e2)
            comm_delay[i] = network.comm_delay[mapping[e1], mapping[e2]]
            comm_rate[i] = network.comm_rate[mapping[e1], mapping[e2]]
            criticality[i] = program.P.edges[e1, e2]['criticality']
            comm_time[i] = np.mean(G_stats.edges[e1, e2]['comm_time'])
        feat = {'bytes': bytes,
                'comm_delay': comm_delay,
                'comm_rate': comm_rate,
                'comm_time': comm_time,
                'criticality': criticality}
        self.edge_feature_mean['comm_time'] = torch.mean(feat['comm_time'])
        self.edge_feature_std['comm_time'] = torch.std(feat['comm_time'])
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
        comm_time = []
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
                    comm_time.append(communication_latency(program, network, n1, n2, d1, d2))
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
                        comm_time.append(communication_latency(program, network,  op1, op2, mapping[op1], dev))
                        criticality.append(program.P.edges[op1, op2]['criticality'])
                    for op1, op2 in program.P.out_edges(op):
                        v.append(node_dict[op2][mapping[op2]])
                        u.append(node)
                        bytes.append(program.get_data_bytes(op1, op2))
                        comm_delay.append(network.comm_delay[dev, mapping[op2]])
                        comm_rate.append(network.comm_rate[dev, mapping[op2]])
                        comm_time.append(communication_latency(program, network, op1, op2, dev, mapping[op2]))
                        criticality.append(program.P.edges[op1, op2]['criticality'])

        u = torch.tensor(u)
        v = torch.tensor(v)
        bytes = torch.tensor(bytes)
        comm_delay = torch.tensor(comm_delay)
        comm_rate = torch.tensor(comm_rate)
        criticality = torch.tensor(criticality)

        feat = {'bytes': bytes,
                'comm_delay': comm_delay,
                'comm_rate': comm_rate,
                'comm_time': comm_time,
                'criticality': criticality}
        self.edge_feature_mean['comm_time'] = torch.mean(feat['comm_time'])
        self.edge_feature_std['comm_time'] = torch.std(feat['comm_time'])
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

    def get_placeto_graph(self, program_id, network_id, mapping, G_stats, cur_op, done_nodes):
        program = self.programs[program_id]
        network = self.networks[network_id]


        feat = {'comp_time': torch.tensor(
                    [np.mean(G_stats.nodes[o]['comp_time']) for o in range(program.n_operators)]),
                'output_size': torch.tensor(
                    [max([program.P.edges[e]['bytes'] for e in program.P.out_edges(o)]) for o in range(program.n_operators)]),
                'cur_placement': 2 * torch.tensor(mapping)/network.n_devices - 1}
        feat['comp_time'] = (feat['comp_time'] - torch.mean(feat['comp_time']))/(torch.std(feat['comp_time']) + 0.001)
        feat['is_cur'] = torch.zeros(program.n_operators)
        feat['is_cur'][cur_op] = 1
        feat['is_done'] = torch.zeros(program.n_operators)
        if len(done_nodes):
            feat['is_done'][done_nodes] = 1

        g = dgl.from_networkx(program.P)

        g.ndata['x'] = torch.t(torch.stack([feat[a] for a in PlacementEnv.PLACETO_FEATURES])).float()

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
