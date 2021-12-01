import networkx as nx
import numpy as np
import torch

from env.utils import *

class PlacementEnv:
    def __init__(self, network, program, seed=0):

        self.network = network
        self.program = program

        self.graph = self.program.P

        self.n_operators = self.program.n_operators
        self.n_devices = self.network.n_devices

        self.seed = seed
        self.mapping = None
        self.latency = np.inf

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

        self.reset()


    def reset(self, reset_map=True, mapping = None):
        if mapping:
            self.mapping = mapping
        elif reset_map:
            self.mapping = self.program.random_mapping()
        self.latency, _ = evaluate_maxP(from_mapping_to_matrix(self.mapping, self.n_devices), self.program, self.network)

    def get_parents(self, node):
        device_set = [self.mapping[n] for n in self.op_parents[node]]
        return self.op_parents[node], device_set

    def get_children(self, node):
        device_set = [self.mapping[n] for n in self.op_children[node]]
        return self.op_children[node], device_set

    def get_parallel(self, node):
        device_set = [self.mapping[n] for n in self.op_parallel[node]]
        return self.op_parallel[node], device_set

    def get_feature_device(self, node, device):
        feature_parent = np.zeros(self.network.get_edge_feature_dim())
        op, d = self.get_parents(node)
        if len(op)>0:
            a = np.array([self.program.get_relative_criticality(node, n) for n in op], dtype=np.float64)
            a /= np.sum(a)
            for i in range(len(op)):
                feature_parent += a[i] * self.network.get_edge_feature(d[i], device)

        feature_child = np.zeros(self.network.get_edge_feature_dim())
        op, d = self.get_children(node)
        if len(op) > 0:
            a = np.array([self.program.get_relative_criticality(node, n) for n in op], dtype=np.float64)
            a /= np.sum(a)
            for i in range(len(op)):
                feature_child += a[i] * self.network.get_edge_feature(device, d[i])

        feature_para = np.zeros(self.program.get_node_feature_dim())
        op, d = self.get_parallel(node)
        for i in range(len(d)):
            if d[i] == device:
                feature_para += self.program.get_node_feature(op[i])

        return np.concatenate((feature_parent, feature_child, feature_para, self.network.get_node_feature(device)), axis=None)

    def get_state(self, node):
        state = [self.get_feature_device(node, d) for d in range(self.n_devices)]
        state.append(self.program.get_node_feature(node))
        return torch.from_numpy(np.concatenate(state)).float()

    def get_state_dim(self):
        return (2 * self.network.get_edge_feature_dim()
                + self.program.get_node_feature_dim()
                + self.network.get_node_feature_dim()) * self.n_devices + self.program.get_node_feature_dim()


    def step(self, node, action):
        assert action in self.program.placement_constraints[node]
        self.mapping[node] = action

        latency, cpath = evaluate_maxP(from_mapping_to_matrix(self.mapping, self.n_devices), self.program, self.network)

        self.program.update_criticality(cpath)
        self.latency = latency

        return latency
