import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class StarNetwork:
    def __init__(self, delay, bw, speed):
        assert len(delay) == len(bw), 'Network delay and bw are for different numbers of devices'
        assert len(delay) == len(speed), 'Confusion on the number of devices in the network'
        devices = list(range(len(delay)))

        self.n_devices = len(delay)

        self.delay = delay
        self.bw = bw
        self.S = speed

        self.C1 = 0.2
        self.C2 = 30

        self.D = np.zeros((self.n_devices, self.n_devices))
        self.R = np.zeros((self.n_devices, self.n_devices))
        for i in range(self.n_devices):
            self.D[i,i] = 0
            self.R[i,i] = 0
            for j in range(i + 1, self.n_devices):
                self.D[i, j] = self.delay[i] + self.delay[j] + self.C2
                self.D[j, i] = self.D[i, j]
                self.R[i, j] = 1/self.bw[i] + 1/self.bw[j] + self.C1
                self.R[j, i] = self.R[i, j]


    def communication_delay(self, n_bytes, i, j):
        return self.D[i,j] + n_bytes * self.R[i,j]


class Program:
    def __init__(self, P, constraints, network):
        self.P = P
        self.n_operators = self.P.number_of_nodes()

        self.A = [P.nodes[i]['compute'] for i in range(self.n_operators)]
        self.B = np.zeros((self.n_operators, self.n_operators))
        for e in self.P.edges:
            self.B[e] = self.P.edges[e]['bytes']

        self.placement_constraints = constraints

        self.T = np.zeros([self.n_operators, network.n_devices])
        for i in range(self.n_operators):
            for j in self.placement_constraints[i]:
                self.T[i,j] = self.A[i]/network.S[j]
            self.T[self.T==0] = np.inf

        self.pinned = [self.placement_constraints[0][0], self.placement_constraints[self.n_operators-1][0]]

    def random_mapping(self):
        mapping = [np.random.choice(self.placement_constraints[i]) for i in range(self.n_operators)]
        return mapping



