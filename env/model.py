import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class StarNetwork:
    def __init__(self, n_devices, n_types):
        self.n_types = n_types
        self.n_devices = n_devices
        self.groups = [set() for i in range(n_types)]
        for i in range(n_devices):
            self.groups[np.random.choice(n_types)].add(i)

        self.fast_link = set(np.random.choice(n_devices, n_devices//2, False))
        self.slow_link = set(range(n_devices)) - self.fast_link


        self.delay = np.random.uniform(5, 10, n_devices)
        self.delay[list(self.slow_link)] = self.delay[list(self.slow_link)] + np.random.uniform(10, 20, len(self.slow_link))

        self.bw = np.random.uniform(100, 200, n_devices)     
        self.bw[list(self.slow_link)] = np.random.uniform(20, 50, len(self.slow_link))

        self.L = np.zeros((n_devices, n_devices))
        self.R = np.zeros((n_devices, n_devices))


        for i in range(n_devices):
            self.L[i, i] = 0
            self.R[i, i] = 0
            for j in range(i+1, n_devices):
                self.L[i,j] = self.delay[i] + self.delay[j]
                self.R[i,j] = 1/self.bw[i] + 1/self.bw[j]
                self.L[j, i] = self.L[i, j]
                self.R[j, i] = self.R[i, j]
                
        self.C1 = 0.2
        self.C2 = 30


    def central_delay(self, n_bytes, i, j):
        if i == j:
            return 0
        return n_bytes * self.C1 + self.C2

    def communication_delay(self, n_bytes, i, j):
        return self.L[i,j] + n_bytes * self.R[i,j] + self.central_delay(n_bytes, i, j)
    
    def get_delay_matrix(self):
        B = self.L + self.C2
        for i in range(N):
            B[i,i] = 0
        return B
    
    def get_rate_matrix(self, nodes=None):
        R =  self.R + self.C1
        for i in range(N):
            R[i,i] = 0
        if nodes:
            return R[nodes][:,nodes]
        return R


class Program:
    def __init__(self, n_operators, network):
        self.n_operators = n_operators
        self.P = nx.DiGraph()
        self.P.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4), (1, 6), (2, 4), (2, 3), (3, 5), (4, 5), (5, 6)])
#         self.P.add_edges_from([(i, i+1) for i in range(n_operators-1)])

        self.B = np.zeros((n_operators, n_operators))
        for e in self.P.edges:
            self.P.edges[e]['bytes'] = np.random.uniform(500, 1000)
            self.B[e] = self.P.edges[e]['bytes']

        k = network.n_types
        self.placement_constraints = [np.random.choice(k, k//2 + (np.random.sample()> 0.5) * 1 - (np.random.sample()> 0.5) * 1) for i in range(n_operators)]
        for i in range(n_operators):
            self.placement_constraints[i] = set().union(*[network.groups[j] for j in self.placement_constraints[i]])

        self.T = np.zeros([n_operators, network.n_devices])
        for i in range(n_operators):
            self.T[i, list(self.placement_constraints[i])] = np.random.exponential(50, len(self.placement_constraints[i]))
            self.T[self.T==0] = np.inf

        self.pinned = [np.random.choice(list(self.placement_constraints[0])), np.random.choice(list(self.placement_constraints[-1]))]

    def random_mapping(self):
        mapping = [np.random.choice(list(self.placement_constraints[i])) for i in range(self.n_operators)]
        mapping[0] = self.pinned[0]
        mapping[-1] = self.pinned[1]
        return mapping



