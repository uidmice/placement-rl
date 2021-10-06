import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

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


def iterative(mapping, program, network):
    map = np.copy(mapping)
    constraints = program.placement_constraints
    to_be_mapped = [i for i in range(program.n_operators) if np.sum(mapping[i]) < 1]

    for o in to_be_mapped:
        map[o, np.random.choice(list(constraints[o]))] = 1
  
    last_latency, critical_path = evaluate(map, program, network)
    
    count = 100
    while True:
        order = list(range(1, len(critical_path)-1))
        np.random.shuffle(order)
        for i in order:
            o = critical_path[i]
            s1 = critical_path[i-1]
            s2 = critical_path[i+1]
            d = d1 = get_mapped_node(map, o)
            d1 = get_mapped_node(map, s1)
            d2 = get_mapped_node(map, s2)
            c_p_c =  network.communication_delay(program.B[s1, o], d1, d) + network.communication_delay(program.B[o, s2], d, d2) + program.T[o, d]
            choices = list(constraints[o]) 
            for n in choices:
                new_cpc = network.communication_delay(program.B[s1, o], d1, n) + network.communication_delay(program.B[o, s2], n, d2) + program.T[o, n]
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
                partial_mapping = helper(to_be, idx+1, constraints)
                for p_m in partial_mapping:
                    p_m.append(node)
                    yield p_m 
  
    min_L = np.inf
    min_mapping = None
    mapp = np.copy(mapping)
  
    for mapped in helper(to_be_mapped, 0, constraints):
        for i in range(l):
            mapp[to_be_mapped[i]] = 0
            mapp[to_be_mapped[i], mapped[-1-i]] = 1
        latency_i, _ = evaluate(mapp, program, network)
        if latency_i < min_L:
            min_L = latency_i
            min_mapping = np.copy(mapp)
    return min_mapping, min_L