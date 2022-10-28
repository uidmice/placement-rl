import numpy as np
from env.utils import *
from env.latency import evaluate, computation_latency, communication_latency, simulate
from placement_rl.placement_env import PlacementEnv

def get_placement_constraints(program, network):
    constraints = []
    for n in range(program.n_operators):
        c = program.P.nodes[n]['h_constraint']
        constraints.append([k[0] for k in filter(lambda elem: c in elem[1], network.device_constraints.items())])
    return constraints

def random_placement(program, network, constraints, number_mappings=100, noise=0):
    latencies = np.zeros(number_mappings)
    for i in range(number_mappings):
        mapping = [np.random.choice(constraints[i]) for i in range(program.n_operators)]
        latencies[i] = evaluate(mapping, program, network, noise, repeat=3)
    return mapping, latencies


def heft(program, network, constraints):
    G = nx.DiGraph()
    G.add_edges_from(program.P.edges())
    for n in G.nodes():
        G.nodes[n]['w'] = np.mean([computation_latency(program, network, n, d) for d in constraints[n]])

    for e in G.edges():
        d1 = constraints[e[0]]
        d2 = constraints[e[1]]
        comm_t = [communication_latency(program, network, e[0], e[1], dev1, dev2) for dev1 in d1 for dev2 in d2]
        G.edges[e]['w'] = np.mean(comm_t)

    for n in reversed(list(nx.topological_sort(G))):
        r = G.nodes[n]['w']
        m = 0
        for v in G.successors(n):
            m = max(m, G.nodes[v]['ranku'] + G.edges[n, v]['w'])
        G.nodes[n]['ranku'] = r + m

    dev_schedule = {}


    for n in reversed(sorted(G.nodes(), key=lambda n: G.nodes[n]['ranku'])):
        eft = {}
        for dev in constraints[n]:
            parent_ft = np.array([G.nodes[v]['ft'] for v in G.predecessors(n)])
            comm_time = np.array([communication_latency(program, network, v, n, G.nodes[v]['dev'], dev) for v in G.predecessors(n)])
            if len(parent_ft) > 0:
                est = np.max(parent_ft + comm_time)
            else:
                est = 0

            comp_time = computation_latency(program, network, n, dev)

            if dev not in dev_schedule:
                eft[dev] = [est, est+ comp_time]
            else:
                schedule = dev_schedule[dev]
                schedule.sort(key=lambda n: n[0])
                slots = [[-np.Inf, schedule[0][0]]]
                slots += [[schedule[i][1], schedule[i+1][0]] for i in range(len(slots)-1)]
                slots += [[schedule[-1][1], np.Inf]]

                for slot in slots:
                    if est >= slot[0] and est+comp_time<= slot[1]:
                        eft[dev] = [est, est + comp_time]
                        break
                    if est <=slot[0] and (slot[0] + comp_time < slot[1]):
                        eft[dev] = [slot[0], slot[0] + comp_time]
                        break

        selected_dev = min(eft, key=lambda d: eft[d][1])
        G.nodes[n]['dev'] = selected_dev
        G.nodes[n]['ft'] = eft[selected_dev][1]
        if selected_dev in dev_schedule:
            dev_schedule[selected_dev].append(eft[selected_dev])
        else:
            dev_schedule[selected_dev] = [eft[selected_dev]]
    return [G.nodes[n]['dev'] for n in range(program.n_operators)]



def random_op_eft_dev(program, network, init_mapping, constraints, iter_num=50, noise=0):
    map = init_mapping.copy()
    latencies = []
    for i in range(iter_num):
        o = np.random.choice(program.n_operators)
        est = {}
        parents = program.op_parents[o]
        G, _, _ = simulate(map, program, network, noise)
        end_time = np.array([np.average(G.nodes[p]['end_time']) for p in parents])
        for d in constraints[o]:
            c_time = np.array([communication_latency(program, network, p, o, map[p], d) for p in parents])
            comp_time = computation_latency(program, network, o, d)
            if len(c_time) > 0:
                est[d] = np.max(c_time + end_time) + comp_time
            else:
                est[d] = comp_time
        map[o] = min(est, key=est.get)
        latencies.append(evaluate(map, program, network, noise, repeat=3))
    return map, latencies


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

