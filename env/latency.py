import torch
import networkx as nx
from collections import namedtuple
import simpy
from env.utils import *

def computation_latency(program, network, op, dev, noise=0):
    a = program.op_compute[op]
    b = network.comp_rate[dev]
    r = a * b
    if not noise:
        return r

    if noise <= 1:
        # return torch.clamp(torch.normal(mean = r, std = noise * r), min=r*(1-noise), max=r*(1+noise))
        return r * (1-noise) + torch.rand([])* 2 * r * noise
    return max(torch.tensor(0).float(), torch.normal(mean = r, std=noise))

def all_computation_latency(program, network, noise=0):
    return program.op_compute @ torch.t(network.comp_rate) + torch.normal(std=noise, size=(program.n_operators, network.n_devices))


def communication_latency(program, network, op1, op2, dev1, dev2, noise=0.0):
    if dev1 == dev2:
        return 0
    a = program.data_bytes[op1, op2]
    b = network.comm_delay[dev1, dev2]
    r = network.comm_rate[dev1, dev2]
    r = b + a * r

    if not noise:
        return r

    if noise <= 1:
        return torch.clamp(torch.normal(mean=r, std=noise * r), min=r * (1 - noise), max=r * (1 + noise))

    return max(torch.tensor(0).float(), torch.normal(mean = r, std=noise))


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


def simulate (mapping, program, network, noise=0, repeat=1, objective='slr'):
    if not isinstance(mapping, list):
        map = from_matrix_to_mapping(mapping)
    else:
        map = mapping



    G = nx.DiGraph()
    G.add_edges_from(program.P.edges)
    nx.set_node_attributes(G, {i: {'start_time': [], 'end_time': [], 'comp_time': [] } for i in G.nodes})
    nx.set_edge_attributes(G, {i: {'arrive_time': [], 'comm_time': []} for i in G.edges})

    Op = namedtuple('Op', 'id inputs outputs comp_time dev')


    def communicate(env, finish_event, e, time):
        # print(f"{env.now: .2f}: Op {e1} --> Op {e2} for {time}")
        yield finish_event
        yield env.timeout(time)
        G.edges[e]['arrive_time'].append(env.now)
        G.edges[e]['comm_time'].append(time)
        return env.now

    def compute(env, op, critical_path_record):
        times = (yield simpy.events.AllOf(env, op.inputs.values())).values()
        times = [A.item() for A in times]
        times = dict(zip(op.inputs.keys(), times))
        if len(times):
            critical_path_record[op.id] = max(times, key=times.get)[0]

        with op.dev.request() as req:  # Generate a request event
            yield req                    # Wait for access
            G.nodes[op.id]['start_time'].append(env.now)
            yield env.timeout(op.comp_time)
            G.nodes[op.id]['end_time'].append(env.now)
            G.nodes[op.id]['comp_time'].append(op.comp_time)
        # print(f"{self.env.now: .2f}: Op {o} --> Dev {des} finished. Running {self.comp_time: .2f}")

        op.outputs.succeed()

    times = []
    paths = []
    repeat_n = repeat
    if noise == 0:
        repeat_n = 1

    for i in range(repeat_n):
        env = simpy.Environment()
        comm_events = {}
        finish_event = {}

        for o in program.P.nodes:
            finish_event[o] = env.event()

        for e in program.P.edges:
            d1 = map[e[0]]
            d2 = map[e[1]]
            comm_events[e] = env.process(communicate(env, finish_event[e[0]], e, communication_latency(program, network, e[0], e[1], d1, d2, noise)))

        ops = {}
        processes = {}
        record = {}
        devices = {d: simpy.Resource(env) for d in list(set(map))}


        for o in program.P.nodes:
            inputs = {e: comm_events[e] for e in program.P.in_edges(o)}
            outputs = finish_event[o]
            des = map[o]
            ops[o] = Op(o, inputs, outputs, computation_latency(program, network, o, des, noise), devices[des])
            processes[o] = env.process(compute(env, ops[o], record))

        while env.peek() < float('inf'):
            env.step()

        c, p = list(record.items())[-1]
        critical_path = [p, c]
        while p in record:
            p = record[p]
            critical_path.insert(0, p)
        times.append(env.now)
        paths.append(critical_path)
    if repeat == 1:
        return G, times[0], paths[0]
    if repeat_n == 1:
        return G, times * repeat, paths
    return G, times,  paths

def evaluate(mapping, program, network, noise=0, repeat=1):
    G, l, path = simulate(mapping, program, network, noise, repeat)
    try:
        average_l = np.mean(l)
    except:
        average_l = l
    return average_l

