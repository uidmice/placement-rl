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
        self.cpath = None
        self.op_idx = None

        self.done = False
        self.max_steps = 15 * self.n_operators
        self.step_count = 0

        self.op_neighbors = []
        self.op_edges = []

        for n in range(self.n_operators):
            self.op_neighbors.append(list(self.graph.predecessors(n)) + list(self.graph.successors(n)))
            self.op_edges.append(list(self.graph.in_edges(n)) + list(self.graph.out_edges(n)))


    def reset(self, mapping = None):
        self.done = False
        self.step_count = 0
        if mapping:
            self.mapping = mapping
        else:
            self.mapping = self.program.random_mapping()

        _, self.cpath = evaluate_maxP(from_mapping_to_matrix(self.mapping, self.n_devices), self.program, self.network)
        self.op_idx = 1
        central_op = self.cpath[1]
        neighbors = self.op_neighbors[central_op]
        edges = self.op_edges[central_op]
        self.state = {
            'op': [central_op, self.program.A[central_op]],
            'constraint': self.program.placement_constraints[central_op],
            'n_op': {n: self.program.A[n] for n in neighbors},
            'n_e': {e: self.program.B[e] for e in edges},
            'map': self.mapping
        }
        return self.state

    def cpc_cost(self, ops, devs):
        assert len(ops) == 3 and len(devs) == 3
        s1, o, s2 = ops
        d1, d, d2 = devs
        return self.network.communication_delay(self.program.B[s1, o], d1, d) + self.network.communication_delay(self.program.B[o, s2], d, d2) + self.program.T[o, d]


    def step(self, action):
        op = self.state['op'][0]
        assert action in self.program.placement_constraints[op]

        pre = self.mapping[op]
        self.mapping[op] = action

        reward, cpath = evaluate_maxP(from_mapping_to_matrix(self.mapping, self.n_devices), self.program, self.network)
        reward = -reward

        if self.op_idx < len(self.cpath) - 2:
            self.op_idx += 1
        else:
            self.cpath = cpath
            self.op_idx = 1
        central_op = self.cpath[self.op_idx]
        neighbors = self.op_neighbors[central_op]
        edges = self.op_edges[central_op]
        next_state = {
            'op': [central_op, self.program.A[central_op]],
            'constraint': self.program.placement_constraints[central_op],
            'n_op': {n: self.program.A[n] for n in neighbors},
            'n_e': {e: self.program.B[e] for e in edges},
            'map': self.mapping
        }
        self.state = next_state
        self.step_count += 1

        done = self.step_count > self.max_steps
        info = {'mapping_change': [op, pre, action]}
        return next_state, reward, done, info
