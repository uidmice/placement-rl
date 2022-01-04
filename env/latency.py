import torch
import networkx as nx

from env.utils import *

def computation_latency(program, network, op, dev, noise=0):
    a = program.op_feature[op]
    b = network.dev_feature[dev]
    if not a.shape:
        r = a * b
    else:
        r = torch.dot(a.double(), b.double())
    return max(0, torch.normal(mean = r, std=noise))

def all_computation_latency(program, network, noise=0):
    return program.op_feature @ torch.t(network.dev_feature) + torch.normal(std=noise, size=(program.n_operators, network.n_devices))


def communication_latency(program, network, op1, op2, dev1, dev2, noise=0.0):
    a = program.data_feature[op1, op2]
    b = network.net_feature[dev1, dev2]
    if not a.shape:
        r = a * b
    else:
        r = torch.dot(a.double(), b.double())
    return max(0, torch.normal(mean = r, std=noise))


# def evaluate(mapping, program, network):
#     for o in nx.topological_sort(program.P):
#         if program.P.in_degree(o) == 0:
#             program.P.nodes[o]['t'] = program.T[o, get_mapped_node(mapping, o)]
#             program.P.nodes[o]['critical'] = None
#         else:
#             max_time = 0
#             critical_node = None
#             des = get_mapped_node(mapping, o)
#             for in_edge in program.P.in_edges(o):
#                 bytes = program.data_feature[in_edge]
#                 s = get_mapped_node(mapping, in_edge[0])
#                 c = network.communication_delay(bytes, s, des)
#                 dl = c + program.P.nodes[in_edge[0]]['t']
#                 if dl > max_time:
#                     max_time = dl
#                     critical_node = in_edge[0]
#             program.P.nodes[o]['t'] = max_time + program.T[o, des]
#             program.P.nodes[o]['critical'] = critical_node
#     o = list(nx.topological_sort(program.P))[-1]
#     latency = program.P.nodes[o]['t']
#     critical_path = [o]
#
#     n = program.P.nodes[o]['critical']
#     while n is not None:
#         critical_path.append(n)
#         n = program.P.nodes[n]['critical']
#     critical_path.reverse()
#     return latency, critical_path


def evaluate(mapping, program, network):
    for o in program.P.nodes:
        des = get_mapped_node(mapping, o)
        program.P.nodes[o]['c'] = computation_latency(program, network, o, des)

    for e in program.P.edges:
        d1 = get_mapped_node(mapping, e[0])
        d2 = get_mapped_node(mapping, e[1])
        program.P.edges[e]['c'] = communication_latency(program, network, e[0], e[1], d1, d2)

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
            program.P.edges[e]['c'] += program.P.nodes[o]['c'] / 2
        for e in program.P.out_edges(o):
            program.P.edges[e]['c'] += program.P.nodes[o]['c'] / 2

    critical_path = nx.dag_longest_path(program.P, 'c')
    latency = 0
    for i in range(len(critical_path) - 1):
        latency += program.P.edges[critical_path[i], critical_path[i + 1]]['c']

    return latency, critical_path
