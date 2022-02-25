import numpy as np
from env.utils import *
import pyomo.environ as pyo
from env.latency import evaluate, computation_latency, communication_latency
from heft.core import schedule


def random_placement(env, number_mappings=100, noise=0):
    latencies = np.zeros(number_mappings)
    min_lat = np.Inf

    for i in range(number_mappings):
        mapping = [np.random.choice(env.program.placement_constraints[i]) for i in range(env.program.n_operators)]
        latencies[i], _ = env.evaluate(mapping, noise)
        if latencies[i] < min_lat:
            min_lat = latencies[i]
            map = mapping
    return map, min_lat, latencies

def heft(env):
    program, network = env.program, env.network
    dag = {}
    for n in program.P.nodes:
        dag[n] = list(program.P.neighbors(n))

    compcost = lambda op, dev: computation_latency(program, network, op, dev)
    commcost = lambda op1, op2, d1, d2: communication_latency(program, network, op1, op2, d1, d2)
    orders, jobson = schedule(dag, program.placement_constraints, compcost, commcost)
    return [jobson[i] for i in range(program.n_operators)], max(e.end for e in orders[program.pinned[1]])


def random_op_greedy_dev(env, init_mapping, iter_num=100, noise=0):
    program, network = env.program, env.network
    map = np.copy(init_mapping)
    last_latency, critical_path = env.evaluate(map, noise)
    latencies = [last_latency]
    for i in range(iter_num):
        o = np.random.choice(program.n_operators)
        lat = {}
        mapping = map.copy()
        for d in program.placement_constraints[o]:
            mapping[o] = d
            latency, _ = env.evaluate(mapping, noise)
            lat[d] = latency
        best = min(lat.values())
        map[o] =np.random.choice([d for d in program.placement_constraints[o] if lat[d] == best])
        latencies.append(best)
    last_latency, critical_path = env.evaluate(map, noise)
    return map, last_latency, latencies


def exhaustive(mapping, program, network):
    to_be_mapped = []
    constraints = program.placement_constraints
    for i in nx.topological_sort(program.P):
        if np.sum(mapping[i]) < 1:
            to_be_mapped.append(i)
    l = len(to_be_mapped)

    def helper(to_be, idx, constraints):
        if idx == len(to_be) - 1:
            for node in constraints[to_be[idx]]:
                yield [node]
        else:
            for node in constraints[to_be[idx]]:
                partial_mapping = helper(to_be, idx + 1, constraints)
                for p_m in partial_mapping:
                    p_m.append(node)
                    yield p_m

    min_L = np.inf
    min_mapping = []
    mapp = np.copy(mapping)
    solution = []

    for mapped in helper(to_be_mapped, 0, constraints):
        for i in range(l):
            mapp[to_be_mapped[i]] = 0
            mapp[to_be_mapped[i], mapped[-1 - i]] = 1
        latency_i, _ = evaluate(mapp, program, network)
        solution.append(latency_i)
        if latency_i < min_L:
            min_L = latency_i
            min_mapping = []
            min_mapping.append(mapp)
        elif latency_i == min_L:
            min_mapping.append(mapp)
    return min_mapping, min_L, solution

