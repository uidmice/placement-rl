
import dgl
import torch

from env.latency import *

class PlacementEnv:
    def __init__(self, network, program, seed=0):

        self.network = network
        self.program = program

        self.graph = self.program.P

        self.n_operators = self.program.n_operators
        self.n_devices = self.network.n_devices

        self.op_features = torch.Tensor([self.program.get_node_feature(i) for i in range(self.n_operators)]).view(self.n_operators, -1)
        self.dev_features = torch.Tensor([self.network.get_node_feature(i) for i in range(self.n_devices)]).view(self.n_devices, -1)

        self.seed = seed

        self.op_parents = [list(self.graph.predecessors(n)) for n in range(self.n_operators)]
        self.op_children = [list(self.graph.successors(n)) for n in range(self.n_operators)]
        self.op_parallel = []
        for n in range(self.n_operators):
            parallel_group = []
            for m in range(self.n_operators):
                if m == n:
                    continue
                if not nx.has_path(self.graph, n, m) and not nx.has_path(self.graph,  m, n):
                    parallel_group.append(m)
            self.op_parallel.append(parallel_group)


    def get_parents(self, node, mapping):
        device_set = [mapping[n] for n in self.op_parents[node]]
        return self.op_parents[node], device_set

    def get_children(self, node, mapping):
        device_set = [mapping[n] for n in self.op_children[node]]
        return self.op_children[node], device_set

    def get_parallel(self, node, mapping):
        device_set = [mapping[n] for n in self.op_parallel[node]]
        return self.op_parallel[node], device_set

    def get_placement_graph(self, mapping):
        node_feature = torch.cat([self.op_features, self.dev_features[mapping]], dim=1)
        u = torch.zeros(self.graph.number_of_edges()).int()
        v = torch.zeros(self.graph.number_of_edges()).int()
        edge_feature = torch.zeros((self.graph.number_of_edges(),
                                    self.program.get_edge_feature_dim()+self.network.get_edge_feature_dim()))
        idx = 0

        for line in nx.generate_edgelist(self.graph, data=False):
            (e1, e2) = [int(s) for s in line.split(' ')]
            u[idx] = e1
            v[idx] = e2
            h1 = self.program.get_edge_feature(e1, e2)
            h2 = self.network.get_edge_feature(mapping[e1], mapping[e2])
            edge_feature[idx] = torch.cat((h1, h2))
            idx += 1

        g = dgl.graph((u, v))
        g.edata['x'] = edge_feature
        g.ndata['x'] = node_feature
        return g

    def get_node_feature_dim(self):
        return self.program.get_node_feature_dim() + self.network.get_node_feature_dim()

    def get_edge_feature_dim(self):
        return self.program.get_edge_feature_dim() + self.network.get_edge_feature_dim()

    def evaluate (self, mapping, noise=0, repeat=100, return_values=False):
        mapping = from_mapping_to_matrix(mapping, self.n_devices)
        l, path = evaluate(mapping, self.program, self.network, noise)
        s = 0
        a = np.ones(repeat) * l
        if noise:
            for i in range(repeat):
                a[i], path = evaluate(mapping, self.program, self.network, noise)
            l = np.average(a)
            s = np.std(a)
        if return_values:
            return l, path, a
        return l, path

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
