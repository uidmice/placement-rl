import torch
import networkx as nx
import simpy
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
    if dev1 == dev2:
        return 0
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
class op:
    def __init__(self, env, inputs, outputs, comp_time, resource):
        # inputs: dict of in-egde comm event {e: event}
        # outputs: dict of out-edge comm event {event: out_comm_time}
        self.env = env
        self.inputs = inputs
        self.outputs = outputs
        self.comp_time = comp_time
        self.resource = resource


    def run(self, o, des, record):
        times = (yield simpy.events.AllOf(self.env, self.inputs.values())).values()
        times = [A.item() for A in times]
        # print(f"{self.env.now: .2f}: Op {o} --> Dev {des} got all inputs at {times}")
        with self.resource.request() as req:  # Generate a request event
            yield req                    # Wait for access
            yield self.env.timeout(self.comp_time)
        # print(f"{self.env.now: .2f}: Op {o} --> Dev {des} finished. Running {self.comp_time: .2f}")

        self.outputs.succeed()
        times = dict(zip(self.inputs.keys(), times))
        if len(times) > 0 :
            record[o] = max(times, key=times.get)[0]
        # for e, t in self.outputs.items():
        #     print(f"{self.env.now}: Op {o} sending for {t}")
        #     yield self.env.timeout(t)
        #     e.succeed(value = self.env.now )
        #
        # return max(times, key=times.get)


def evaluate(mapping, program, network, noise=0):
    env = simpy.Environment()
    map = from_matrix_to_mapping(mapping)

    devices = {d: simpy.Resource(env) for d in list(set(map))}

    def send_output(finish_event, e1, e2, time):
        # print(f"{env.now: .2f}: Op {e1} --> Op {e2} for {time}")
        yield finish_event
        yield env.timeout(time)
        return env.now

    comm_events = {}
    comm_time = {}
    finish_event = {}

    for o in program.P.nodes:
        finish_event[o] = env.event()

    for e in program.P.edges:
        d1 = map[e[0]]
        d2 = map[e[1]]
        comm_time[e] = communication_latency(program, network, e[0], e[1], d1, d2, noise)
        comm_events[e] = env.process(send_output(finish_event[e[0]], e[0], e[1], comm_time[e]))

    ops = {}
    processes = {}
    record = {}
    for o in program.P.nodes:
        inputs = {e: comm_events[e] for e in program.P.in_edges(o)}
        outputs = finish_event[o]
        des = map[o]
        ops[o] = op(env, inputs, outputs, computation_latency(program, network, o, des, noise), devices[des])
        processes[o] = env.process(ops[o].run(o, des, record))

    while env.peek() < float('inf'):
        env.step()

    c, p = list(record.items())[-1]
    critical_path = [p, c]
    while p in record:
        p = record[p]
        critical_path.insert(0, p)
    return env.now, critical_path





# def evaluate(mapping, program, network):
#     for o in program.P.nodes:
#         des = get_mapped_node(mapping, o)
#         program.P.nodes[o]['c'] = computation_latency(program, network, o, des)
#
#     for e in program.P.edges:
#         d1 = get_mapped_node(mapping, e[0])
#         d2 = get_mapped_node(mapping, e[1])
#         program.P.edges[e]['c'] = communication_latency(program, network, e[0], e[1], d1, d2)
#
#     for o in program.P.nodes:
#         if program.P.in_degree(o) == 0:
#             for e in program.P.out_edges(o):
#                 program.P.edges[e]['c'] += program.P.nodes[o]['c']
#             continue
#         if program.P.out_degree(o) == 0:
#             for e in program.P.in_edges(o):
#                 program.P.edges[e]['c'] += program.P.nodes[o]['c']
#             continue
#         for e in program.P.in_edges(o):
#             program.P.edges[e]['c'] += program.P.nodes[o]['c'] / 2
#         for e in program.P.out_edges(o):
#             program.P.edges[e]['c'] += program.P.nodes[o]['c'] / 2
#
#     critical_path = nx.dag_longest_path(program.P, 'c')
#     latency = 0
#     for i in range(len(critical_path) - 1):
#         latency += program.P.edges[critical_path[i], critical_path[i + 1]]['c']
#
#     return latency, critical_path
