import numpy as np

from env.utils import generate_program, generate_network
from env.model import Program, StarNetwork
from env.placement_env import PlacementEnv
from placement_rl.agent import PlacementAgent

import os
import pickle


def get_config(n_devices, n_programs, s1=0, s2=[0]):
    p = f'data/N_{n_devices}'
    delay, bw, speed = pickle.load(open(os.path.join(p, f'network_{n_devices}_seed_{s1}.pk'), 'rb'))

    network = StarNetwork(delay, bw, speed)
    programs = []
    for seed in s2:
        DAG, constraints = pickle.load(open(os.path.join(p, f'program_{n_programs}_seed_{seed}.pk'), 'rb'))
        programs.append(Program(DAG, constraints, network))
    return network, programs


n_devices = 20
n_operators = 8
network = StarNetwork(*generate_network(n_devices, seed=1))
DAG, constraints = generate_program(n_operators, n_devices, seed=2)
program = Program(DAG, constraints, network)
env = PlacementEnv(network, program)
agent = PlacementAgent(env.get_state_dim(), n_devices)
episode_rewards, reward_trace = agent.train(env, 10)