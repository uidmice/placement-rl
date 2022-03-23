
import dgl
import torch

from env.latency import *

class PlacementEnv:
    NODE_FEATURES = ['compute', 'comp_rate']
    EDGE_FEATURES = ['bytes', 'comm_delay', 'comm_rate']
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
        # self.graph = self.program.P
        #
        # self.n_operators = self.program.n_operators
        # self.n_devices = self.network.n_devices
        #
        # self.op_features = torch.Tensor([self.program.get_node_compute(i) for i in range(self.n_operators)]).view(self.n_operators, -1)
        # self.dev_features = torch.Tensor([self.network.get_node_compute(i) for i in range(self.n_devices)]).view(self.n_devices, -1)

        self.seed = seed


    def get_parents(self, program_id, node, mapping):
        try:
            program = self.programs[program_id]
        except:
            print(f'Program id {program_id} not valid')
            return

        device_set = [mapping[n] for n in program.op_parents[node]]
        return program.op_parents[node], device_set

    def get_children(self, program_id, node, mapping):
        try:
            program = self.programs[program_id]
        except:
            print(f'Program id {program_id} not valid')
            return
        device_set = [mapping[n] for n in program.op_children[node]]
        return program.op_children[node], device_set

    def get_parallel(self, program_id, node, mapping):
        try:
            program = self.programs[program_id]
        except:
            print(f'Program id {program_id} not valid')
            return
        device_set = [mapping[n] for n in program.op_parallel[node]]
        return program.op_parallel[node], device_set

    def get_node_feature_dim(self):
        return len(PlacementEnv.NODE_FEATURES)

    def get_edge_feature_dim(self):
        return len(PlacementEnv.EDGE_FEATURES)

    def get_node_feature(self, program_id, network_id, mapping):
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

        feat = {'compute': program.op_compute, 'comp_rate': network.comp_rate[mapping]}
        for feature in PlacementEnv.NODE_FEATURES:
            feat[feature] = (feat[feature] - self.node_feature_mean[feature])/self.node_feature_std[feature]
        return feat

    def get_edge_feature(self, program_id, network_id, mapping):
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

        for i, line in enumerate(nx.generate_edgelist(program.P, data=False)):
            (e1, e2) = [int(s) for s in line.split(' ')]
            u[i] = e1
            v[i] = e2
            bytes[i] = program.get_data_bytes(e1, e2)
            comm_delay[i] = network.comm_delay[mapping[e1], mapping[e2]]
            comm_rate[i] = network.comm_rate[mapping[e1], mapping[e2]]
        feat = {'bytes': bytes, 'comm_delay': comm_delay, 'comm_rate': comm_rate}
        for feature in PlacementEnv.EDGE_FEATURES:
            feat[feature] = (feat[feature] - self.edge_feature_mean[feature])/self.edge_feature_std[feature]
        return u, v, feat


    def get_placement_graph(self, program_id, network_id, mapping):
        node_features = self.get_node_feature(program_id, network_id, mapping)
        u, v, edge_features = self.get_edge_feature(program_id, network_id, mapping)

        g = dgl.graph((u, v))
        g.edata['x'] = torch.t(torch.stack(list(edge_features.values()))).float()
        g.ndata['x'] = torch.t(torch.stack(list(node_features.values()))).float()
        return g

    def evaluate(self, program_id, network_id, mapping, noise=0, repeat=1, return_stats=False):
        program = self.programs[program_id]
        network = self.networks[network_id]

        G, l, path = simulate(mapping, program, network, noise, repeat)
        if return_stats:
            return l, path, G
        try:
            average_l = torch.mean(l)
        except:
            average_l = l
        return average_l

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
