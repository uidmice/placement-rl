

import gym
from gym import spaces
from gym.utils import seeding


from env.model import StarNetwork, Program
from utils import *

N_OP = 7
N_DEV = 50
class PlacementEnv(gym.Env):

    def __init__(self):

        self.network = StarNetwork(N_DEV, 5)
        self.program = Program(N_OP, self.network)

        self.n_operators = N_OP
        self.n_devices = N_DEV

        self.action_space = spaces.Discrete(self.n_devices)
        self.observation_space = spaces.Dict({
            'op': spaces.Tuple((
                spaces.Discrete(self.n_operators),
                spaces.Discrete(self.n_operators),
                spaces.Discrete(self.n_operators)
            )),
            'device': spaces.Tuple((
                spaces.Discrete(self.n_devices),
                spaces.Discrete(self.n_devices),
                spaces.Discrete(self.n_devices)
            ))
        })

        self.seed()
        self.init_mapping = self.program.random_mapping()

        self.mapping = None
        self.state = None
        self.path = None
        self.idx = None

        self.max_steps = self.n_operators * 50
        self.count = 0


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.mapping = self.init_mapping.copy()
        self.count = 0

        latency, self.path = evaluate_maxP(from_mapping_to_matrix(self.mapping, self.n_devices), self.program, self.network)
        self.idx = 0
        ops = [self.path[0], self.path[1], self.path[2]]
        ds = [self.mapping[i] for i in ops]
        self.state = {
            'op': ops,
            'device': ds
        }
        return self.state

    def cpc_cost(self, ops, devs):
        assert len(ops) == 3 and len(devs) == 3
        s1, o, s2 = ops
        d1, d, d2 = devs
        return self.network.communication_delay(self.program.B[s1, o], d1, d) + self.network.communication_delay(self.program.B[o, s2], d, d2) + self.program.T[o, d]


    def step(self, action):
        op = self.state['op'][1]
        assert action in self.program.placement_constraints[op]

        pre = self.mapping[op]
        self.mapping[op] = action
        self.count += 1
        reward = -self.cpc_cost(self.state['op'], [self.mapping[i] for i in self.state['op']])


        if self.idx < len(self.path) - 3:
            self.idx += 1
            ops = [self.path[self.idx], self.path[self.idx+1], self.path[self.idx+2]]
            ds = [self.mapping[i] for i in ops]
            next_state = {
                'op': ops,
                'device': ds
            }
        else:
            latency, self.path = evaluate_maxP(from_mapping_to_matrix(self.mapping, self.n_devices), self.program, self.network)
            self.idx = 0
            ops = [self.path[0], self.path[1], self.path[2]]
            ds = [self.mapping[i] for i in ops]
            next_state = {
                'op': ops,
                'device': ds
            }

        self.state = next_state

        done = self.count > self.max_steps
        info = {'mapping_change': [op, pre, action]}
        return next_state, reward, done, info

