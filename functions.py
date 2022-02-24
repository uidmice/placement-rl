import numpy as np
from env.utils import *
import pyomo.environ as pyo
from env.latency import evaluate


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
            c_p_c = network.communication_delay(program.data_feature[s1, o], d1, d) + network.communication_delay(program.data_feature[o, s2],
                                                                                                                  d, d2) + \
                    program.T[o, d]
            choices = list(constraints[o])
            for n in choices:
                new_cpc = network.communication_delay(program.data_feature[s1, o], d1, n) + network.communication_delay(
                    program.data_feature[o, s2], n, d2) + program.T[o, n]
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


def qlp_linear(pinned, placement_constraints, B, T, D, R):
    M = T.shape[0]
    N = T.shape[1]
    model = pyo.ConcreteModel(name='Linear constraint')
    model.x = pyo.Var(range(M), range(N), domain=pyo.Binary)

    def _obj(m):
        summ = 0
        for i in range(M):
            for j in range(N):
                if T[i, j] < np.inf:
                    summ += T[i, j] * m.x[i, j]
        for i in range(M):
            for j in range(M):
                if B[i, j] > 0:
                    for k in range(N):
                        for l in range(N):
                            summ += m.x[i, k] * m.x[j, l] * (B[i, j] * R[l, k] + D[l, k])
        return summ

    model.obj = pyo.Objective(rule=_obj, sense=pyo.minimize)

    model.c = pyo.ConstraintList()
    model.c.add(model.x[0, pinned[0]] == 1)
    model.c.add(model.x[M - 1, pinned[1]] == 1)
    for i in range(M):
        model.c.add(sum(model.x[i, j] for j in range(N)) == 1)
        model.c.add(sum(model.x[i, j] for j in list(placement_constraints[i])) == 1)

    # model.pprint()

    baron = pyo.SolverFactory('baron', executable='~/baron-osx64/baron')
    result_obj = baron.solve(model, tee=False)
    solution = [j for i in range(M) for j in range(N) if pyo.value(model.x[i, j]) > 0]
    return solution


def iterative_qlp(mapping, program, network):
    mapp = np.copy(mapping)
    constraints = program.placement_constraints
    to_be_mapped = [i for i in range(program.n_operators) if np.sum(mapping[i]) < 1]

    for o in to_be_mapped:
        mapp[o, np.random.choice(list(constraints[o]))] = 1

    last_latency, critical_path = evaluate(mapp, program, network)
    best_mapp = np.copy(mapp)

    count = 7
    while True:
        p_constraints = [constraints[i] for i in critical_path]
        B = program.data_feature[critical_path][:, critical_path]
        T = program.T[critical_path]
        D = network.D
        R = network.R
        sol = qlp_linear(program.pinned, p_constraints, B, T, D, R)
        for i in range(1, len(critical_path) - 1):
            mapp[critical_path[i]] = 0
            mapp[critical_path[i], sol[i]] = 1
        cur_latency, cur_path = evaluate(mapp, program, network)
        if set(critical_path) == set(cur_path):
            break
        if not count:
            break
        count -= 1

        critical_path = cur_path
        if cur_latency < last_latency:
            last_latency = cur_latency
            best_mapp = np.copy(mapp)

    return best_mapp, last_latency


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

