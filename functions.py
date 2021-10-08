import numpy as np
from env.utils import *


def iterative(mapping, program, network):
    map = np.copy(mapping)
    constraints = program.placement_constraints
    to_be_mapped = [i for i in range(program.n_operators) if np.sum(mapping[i]) < 1]

    for o in to_be_mapped:
        map[o, np.random.choice(list(constraints[o]))] = 1

    last_latency, critical_path = evaluate(map, program, network)

    count = 100
    while True:
        order = list(range(1, len(critical_path) - 1))
        np.random.shuffle(order)
        for i in order:
            o = critical_path[i]
            s1 = critical_path[i - 1]
            s2 = critical_path[i + 1]
            d = get_mapped_node(map, o)
            d1 = get_mapped_node(map, s1)
            d2 = get_mapped_node(map, s2)
            c_p_c = network.communication_delay(program.B[s1, o], d1, d) + network.communication_delay(program.B[o, s2],
                                                                                                       d, d2) + \
                    program.T[o, d]
            choices = list(constraints[o])
            for n in choices:
                new_cpc = network.communication_delay(program.B[s1, o], d1, n) + network.communication_delay(
                    program.B[o, s2], n, d2) + program.T[o, n]
                if new_cpc < c_p_c:
                    c_p_c = new_cpc
                    d = n
            map[o] = 0
            map[o, d] = 1
        cur_latency, critical_path = evaluate(map, program, network)
        if cur_latency < last_latency:
            last_latency = cur_latency
        elif count:
            count -= 1
        else:
            break
    return map, last_latency


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
    min_mapping = None
    mapp = np.copy(mapping)

    for mapped in helper(to_be_mapped, 0, constraints):
        for i in range(l):
            mapp[to_be_mapped[i]] = 0
            mapp[to_be_mapped[i], mapped[-1 - i]] = 1
        latency_i, _ = evaluate(mapp, program, network)
        if latency_i < min_L:
            min_L = latency_i
            min_mapping = np.copy(mapp)
    return min_mapping, min_L