import networkx as nx
import torch
import numpy as np
import matplotlib.pyplot as plt

class Program:
    def __init__(self, P: nx.DiGraph, constraints):
        self.P = P
        self.n_operators = self.P.number_of_nodes()

        self.op_feature = torch.reshape(torch.tensor([P.nodes[i]['compute'] for i in range(self.n_operators)]), (self.n_operators, 1))
        self.op_feature_norm = (self.op_feature - torch.mean(self.op_feature)) / torch.std(self.op_feature)

        self.data_feature = torch.zeros((self.n_operators, self.n_operators))
        for e in self.P.edges:
            self.data_feature[e] = self.P.edges[e]['bytes']
        self.data_feature_norm = (self.data_feature - torch.mean(self.data_feature))/torch.std(self.data_feature)

        self.data_feature = torch.stack([torch.ones(self.n_operators, self.n_operators), self.data_feature], dim=2)
        self.data_feature_norm = torch.stack([torch.ones(self.n_operators, self.n_operators), self.data_feature_norm], dim=2)


        self.placement_constraints = constraints

        self.pinned = [self.placement_constraints[0][0], self.placement_constraints[self.n_operators-1][0]]

        self.init_criticality()

    def random_mapping(self):
        mapping = [np.random.choice(self.placement_constraints[i]) for i in range(self.n_operators)]
        return mapping

    def draw(self):
        nx.draw_networkx(self.P)
        plt.show()

    def init_criticality (self, deterministic=False):
        for n in nx.topological_sort(self.P):
            data_size = {e : self.P.edges[e]['bytes'] for e in self.P.out_edges(n)}
            if len(data_size):
                if len(data_size) == 1:
                    self.P.edges[list(data_size.keys())[0]]['criticality'] = 1
                    continue

                if deterministic:
                    sorted_data = sorted(data_size.items(), key=lambda item: item[1])
                    max_path = sorted_data[0][0]
                    self.P.edges[max_path]['criticality']=1
                    for edges, v in sorted_data[1:]:
                        self.P.edges[edges]['criticality'] = 0
                else:
                    max_size = max(data_size.values())
                    sum = 0
                    for k in data_size:
                        data_size[k] = np.exp(data_size[k]/max_size * 10) # normalize before exp
                        sum += data_size[k]
                    for k in data_size:
                        self.P.edges[k]['criticality'] = data_size[k]/sum
        self.populate_criticality()

    def populate_criticality(self):
        for n in nx.topological_sort(self.P):
            if self.P.in_degree(n) == 0:
                self.P.nodes[n]['criticality'] = 1
            else:
                self.P.nodes[n]['criticality'] = 0
                for parent in self.P.predecessors(n):
                    self.P.nodes[n]['criticality'] += self.P.nodes[parent]['criticality'] * self.P.edges[(parent, n)]['criticality']
        for e in self.P.edges:
            if self.P.edges[e]['criticality'] == 0:
                self.P.edges[e]['criticality_r'] = 0
            else:
                self.P.edges[e]['criticality_r'] = self.P.edges[e]['criticality'] * self.P.nodes[e[0]]['criticality']/self.P.nodes[e[1]]['criticality']


    def update_criticality(self, critical_path, step_size=0.1):
        for i in range(len(critical_path) - 1):
            n = critical_path[i]
            if self.P.out_degree(n) >1:
                critical_child = critical_path[i+1]
                for e in self.P.out_edges(n):
                    if e[1] != critical_child:
                        self.P.edges[e]['criticality'] -= step_size * self.P.edges[e]['criticality']
                    else:
                        self.P.edges[e]['criticality'] += step_size * (1-self.P.edges[e]['criticality'])
        self.populate_criticality()

    def get_node_feature(self, node):
        return self.op_feature_norm[node]

    def get_node_feature_dim(self):
        return self.op_feature_norm.shape[1]

    def get_edge_feature(self, node1, node2):
        return torch.squeeze(self.data_feature_norm[node1, node2])

    def get_edge_feature_dim(self):
        return self.data_feature_norm.shape[2]

    def get_relative_criticality(self, node1, node2):
        if nx.has_path(self.P, node1, node2):
            rt = 0
            for path in map(nx.utils.pairwise, nx.all_simple_paths(self.P, node1, node2)):
                pp = 1
                for e in path:
                    pp *= self.P.edges[e]['criticality']
                rt += pp
            return rt

        if nx.has_path(self.P, node2, node1):
            rt = 0
            for path in map(nx.utils.pairwise, nx.all_simple_paths(self.P, node2, node1)):
                pp = 1
                for e in path:
                    pp *= self.P.edges[e]['criticality_r']
                rt += pp
            return rt

        return 0

