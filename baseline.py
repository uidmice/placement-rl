import numpy as np
from env.utils import *
from env.latency import evaluate, computation_latency, communication_latency, simulate
from placement_rl.placement_env import PlacementEnv
from heft.core import schedule

def get_placement_constraints(program, network):
    constraints = []
    for n in program.P.nodes:
        c = program.P.nodes[n]['h_constraint']
        constraints.append([k[0] for k in filter(lambda elem: c in elem[1], network.device_constraints.items())])
    return constraints

def random_placement(program, network, number_mappings=100, noise=0):
    latencies = np.zeros(number_mappings)
    min_lat = np.Inf

    constraints = get_placement_constraints(program, network)

    for i in range(number_mappings):
        mapping = [np.random.choice(constraints[i]) for i in range(program.n_operators)]
        latencies[i] = evaluate(mapping, program, network, noise)
        if latencies[i] < min_lat:
            min_lat = latencies[i]
            map = mapping
    return map, min_lat, latencies

def heft(program, network):
    dag = {}
    for n in program.P.nodes:
        dag[n] = list(program.P.neighbors(n))

    constraints = get_placement_constraints(program, network)
    compcost = lambda op, dev: computation_latency(program, network, op, dev)
    commcost = lambda op1, op2, d1, d2: communication_latency(program, network, op1, op2, d1, d2)
    orders, jobson = schedule(dag, constraints, compcost, commcost)
    return [jobson[i] for i in range(program.n_operators)], max(e.end for e in orders[program.pinned[1]])

def random_op_est_dev(program, network, init_mapping, iter_num=50, noise=0):
    map = init_mapping.copy()
    last_latency = evaluate(map, program, network, noise)
    latencies = [last_latency]
    for i in range(iter_num):
        o = np.random.choice(program.n_operators)
        if o==0:
            latencies.append(latencies[-1])
            continue
        est = {}
        parents = program.op_parents[o]
        G, _, _ = simulate(map, program, network, noise)
        end_time = np.array([np.average(G.nodes[p]['end_time']) for p in parents])
        constraints = get_placement_constraints(program, network)
        for d in constraints[o]:
            c_time = np.array([communication_latency(program, network, p, o, map[p], d) for p in parents])
            est[d] = np.max(c_time + end_time)
        map[o] = min(est, key=est.get)
        latencies.append(evaluate(map, program, network, noise, repeat=3))
    return map, latencies[-1], latencies


def random_op_greedy_dev(program, network, init_mapping, iter_num=50, noise=0):
    map = init_mapping.copy()
    last_latency = evaluate(map, program, network, noise)
    latencies = [last_latency]
    for i in range(iter_num):
        o = np.random.choice(program.n_operators)
        lat = {}
        mapping = map.copy()
        constraints = get_placement_constraints(program, network)
        for d in constraints[o]:
            mapping[o] = d
            latency = evaluate(mapping, program, network, noise)
            lat[d] = latency
        best = min(lat.values())
        map[o] =np.random.choice([d for d in constraints[o] if lat[d] == best])
        latencies.append(best)
    last_latency = evaluate(map, program, network, noise)
    return map, last_latency, latencies


def exhaustive(mapping, program, network):
    to_be_mapped = []
    constraints = get_placement_constraints(program, network)
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

