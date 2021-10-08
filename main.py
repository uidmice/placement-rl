import numpy as np

from env.model import Program, StarNetwork
from env.placement_env import PlacementEnv
import os
import pickle

def get_config(n_devices, n_programs, s1=0, s2=3):
    p = f'data/N_{n_devices}'
    delay, bw, speed = pickle.load(open(os.path.join(p, f'network_{n_devices}_seed_{s1}.pk'), 'rb'))
    DAG, constraints = pickle.load(open(os.path.join(p, f'program_{n_programs}_seed_{s2}.pk'), 'rb'))
    network = StarNetwork(delay, bw, speed)
    program = Program(DAG, constraints, network)
    return network, program

n = 40
m = 8

network, program = get_config(n, m)
env = PlacementEnv(network, program)
