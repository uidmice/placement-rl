import networkx as nx
import numpy as np
import pickle
import os

def get_mapped_node(map, i):
    return np.where(map[i] == 1)[0][0]

def from_mapping_to_matrix(mapping, n_devices):
    m = np.zeros((len(mapping), n_devices))
    for i in range(len(mapping)):
        m[i, mapping[i]] = 1
    return m

def from_matrix_to_mapping(m):
    return [get_mapped_node(m, i) for i in range(m.shape[0])]

def evaluate(mapping, program, network):
    for o in nx.topological_sort(program.P):
        if program.P.in_degree(o) == 0:
            program.P.nodes[o]['t'] = program.T[o, get_mapped_node(mapping, o)]
            program.P.nodes[o]['critical'] = None
        else:
            max_time = 0
            critical_node = None
            des = get_mapped_node(mapping, o)
            for in_edge in program.P.in_edges(o):
                bytes = program.B[in_edge]
                s = get_mapped_node(mapping, in_edge[0])
                c = network.communication_delay(bytes, s, des)
                dl = c + program.P.nodes[in_edge[0]]['t']
                if dl > max_time:
                    max_time = dl
                    critical_node = in_edge[0]
            program.P.nodes[o]['t'] = max_time + program.T[o, des]
            program.P.nodes[o]['critical'] = critical_node
    o = list(nx.topological_sort(program.P))[-1]
    latency = program.P.nodes[o]['t'] 
    critical_path = [o]

    n = program.P.nodes[o]['critical']
    while n is not None:
        critical_path.append(n)
        n = program.P.nodes[n]['critical']
    critical_path.reverse()
    return latency, critical_path

def evaluate_maxP(mapping, program, network):
    for o in program.P.nodes:
        des = get_mapped_node(mapping, o)
        program.P.nodes[o]['c'] = program.T[o, des]
            
    for e in program.P.edges:
        d1 = get_mapped_node(mapping, e[0])
        d2 = get_mapped_node(mapping, e[1])
        b = program.B[e]
        program.P.edges[e]['c'] = network.communication_delay(b, d1, d2)
    
    for o in program.P.nodes:
        if program.P.in_degree(o) == 0:
            for e in program.P.out_edges(o):
                program.P.edges[e]['c'] += program.P.nodes[o]['c']
            continue
        if program.P.out_degree(o) == 0:
            for e in program.P.in_edges(o):
                program.P.edges[e]['c'] += program.P.nodes[o]['c']
            continue
        for e in program.P.in_edges(o):
            program.P.edges[e]['c'] += program.P.nodes[o]['c']/2
        for e in program.P.out_edges(o):
            program.P.edges[e]['c'] += program.P.nodes[o]['c']/2
    
    critical_path = nx.dag_longest_path(program.P, 'c')
    latency = 0
    for i in range(len(critical_path)-1):
        latency += program.P.edges[critical_path[i], critical_path[i+1]]['c']
    
    return latency, critical_path


def generate_network(n_devices, seed):
    np.random.seed(seed)

    fast_link = set(np.random.choice(n_devices, n_devices // 2, False))
    slow_link = set(range(n_devices)) - fast_link

    delay = np.random.uniform(5, 10, n_devices)
    delay[list(slow_link)] = delay[list(slow_link)] + np.random.uniform(10, 20, len(slow_link))

    bw = np.random.uniform(100, 200, n_devices)
    bw[list(slow_link)] = np.random.uniform(20, 50, len(slow_link))

    speed = np.random.uniform(1, 3, n_devices)
    return delay, bw, speed


def generate_program(n_operators, n_devices, seed, B=1000, l=100):
    np.random.seed(seed)
    G = nx.gnp_random_graph(n_operators - 2, 0.8, seed=seed, directed=True)
    DAG = nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v])
    DAG = nx.relabel.convert_node_labels_to_integers(DAG, first_label=1)
    heads = [node for node in DAG.nodes() if DAG.in_degree(node) == 0]
    tails = [node for node in DAG.nodes() if DAG.out_degree(node) == 0]

    for n in heads:
        DAG.add_edge(0, n)
    for n in tails:
        DAG.add_edge(n, n_operators - 1)

    constraints = {}
    n_types = n_devices // 5
    groups = [set() for i in range(n_types)]
    for i in range(n_devices):
        groups[np.random.choice(n_types)].add(i)
    k = len(groups)
    for e in DAG.edges:
        DAG.edges[e]['bytes'] = np.random.uniform(B/2, B)
    for n in DAG.nodes:
        DAG.nodes[n]['compute'] = np.random.exponential(l)
        group_ids = np.random.choice(k, k // 2 + (np.random.sample() > 0.5) * 1 - (np.random.sample() > 0.5) * 1)
        constraints[n] = list(set().union(*[groups[j] for j in group_ids]))
        if not len(constraints[n]):
            constraints[n] = np.random.choice(n_devices, n_devices//2).tolist()
    constraints[0] = [np.random.choice(constraints[0])]
    constraints[n_operators - 1] = [np.random.choice(constraints[n_operators - 1])]
    return DAG, constraints

