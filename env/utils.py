import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import random

def get_mapped_node(map, i):
    return np.where(map[i] == 1)[0][0]

def from_mapping_to_matrix(mapping, n_devices):
    m = np.zeros((len(mapping), n_devices))
    for i in range(len(mapping)):
        m[i, mapping[i]] = 1
    return m

def from_matrix_to_mapping(m):
    return [get_mapped_node(m, i) for i in range(m.shape[0])]


def graph_dag_structure(v,
                        alpha,
                        seed,
                        conn_prob=0.1,
                        visualize=False):
    np.random.seed(seed)

    height_mean = np.sqrt(v) / alpha
    height = int(np.ceil(np.random.uniform(height_mean * 0.8, height_mean * 1.2)))

    width_mean = alpha * np.sqrt(v)
    widths = []

    for i in range(height):
        widths.append(int(np.ceil(np.random.uniform(0, 2 * width_mean))))

    total_operator = sum(widths)

    G = nx.DiGraph()
    G.add_node(0)
    cnt = 1
    nodes = [[] for i in range(height + 2)]
    nodes[0].append(0)
    for i in range(height):
        for j in range(widths[i]):
            nodes[i + 1].append(cnt)
            G.add_node(cnt)
            for k in set().union(*nodes[:i + 1]):
                if np.random.rand() < conn_prob:
                    G.add_edge(k, cnt)
            cnt += 1

    nodes[-1].append(total_operator + 1)
    G.add_node(total_operator + 1)

    # make sure the depth equals height
    critical_path = [random.choice(n) for n in nodes]
    for x, y in zip(critical_path, critical_path[1:]):
        G.add_edge(x, y)

    # Validity checking, if any node in the middle has 0 indegree or 0 outdegree,
    # randomly connect
    for i, layer in enumerate(nodes):
        if i > 0 and i < height + 1:
            for node in layer:
                if G.out_degree(node) == 0:
                    G.add_edge(node, random.choice([a for n in nodes[i + 1:] for a in n]))
                if G.in_degree(node) == 0:
                    G.add_edge(random.choice([a for n in nodes[:i] for a in n]), node)

    if visualize:
        visualize_dag(G, widths, height)

    widths.insert(0, 1)
    widths.append(1)
    height += 2
    return G, widths, height


def generate_graph(alpha,
                   v,
                   connect_prob,
                   seed,
                   num_types,
                   avg_compute,
                   avg_bytes,
                   b_comp=0.2,
                   b_comm=0.2):
    G, widths, height = graph_dag_structure(v, alpha, seed, connect_prob)
    np.random.seed(seed)

    # compute requirement for each dag node
    for n in G.nodes:
        G.nodes[n]['compute'] = np.random.uniform(avg_compute * (1 - b_comp / 2), avg_compute * (1 + b_comp / 2))
        G.nodes[n]['h_constraint'] = np.random.choice(range(num_types))

    # Communication requirement (bytes) for each dag edge
    for edge in G.edges:
        G.edges[edge]['bytes'] = np.random.uniform(avg_bytes * (1 - b_comm / 2), avg_bytes * (1 + b_comm / 2))

    G.graph['alpha'] = alpha
    G.graph['v'] = v
    G.graph['connect_prob'] = connect_prob
    G.graph['seed'] = seed
    G.graph['num_types'] = num_types
    G.graph['avg_compute'] = avg_compute
    G.graph['avg_bytes'] = avg_bytes
    G.graph['b_comp'] = b_comp
    G.graph['b_comm'] = b_comm

    return G


def generate_network(n_devices,
                     seed,
                     num_types=5,
                     type_prob=0.3,
                     avg_speed=3,
                     avg_bw=200,
                     avg_delay=10,
                     b_bw=0.2,
                     b_speed=0.2
                     ):
    delay = np.random.uniform(0, 2 * avg_delay, (n_devices, n_devices))
    avg_comm = 1 / avg_bw
    comm_speed = np.random.uniform(avg_comm * (1 - b_bw / 2), avg_comm * (1 + b_bw / 2), (n_devices, n_devices))
    for i in range(n_devices):
        for j in range(i, n_devices):
            if i == j:
                delay[i][j] = 0
                comm_speed[i][j] = 0
            else:
                delay[i][j] = delay[j][i]
                comm_speed[i][j] = comm_speed[j][i]

    # speed for each device
    speed = np.random.uniform(avg_speed * (1 - b_speed / 2), avg_speed * (1 + b_speed / 2), n_devices)

    device_constraints = {}

    for i in range(n_devices):
        device_constraints[i] = []
        for j in range(num_types):
            if np.random.rand() < type_prob:
                device_constraints[i].append(j)
        if len(device_constraints[i]) == 0:
            device_constraints[i].append(np.random.choice(range(num_types)))

    network = {}
    network["delay"] = delay
    network["comm_speed"] = comm_speed
    network["speed"] = speed
    network["device_constraints"] = device_constraints
    network['para'] = {}

    network['para']['n_devices'] = n_devices
    network['para']["seed"] = seed
    network['para']["num_types"] = num_types
    network['para']["type_prob"] = type_prob
    network['para']["avg_speed"] = avg_speed
    network['para']['avg_bw'] = avg_bw
    network['para']['avg_delay'] = avg_delay
    network['para']["b_bw"] = b_bw
    network['para']['b_speed'] = b_speed

    return network

#
# def generate_network(n_devices, seed):
#     np.random.seed(seed)
#
#     fast_link = set(np.random.choice(n_devices, n_devices // 2, False))
#     slow_link = set(range(n_devices)) - fast_link
#
#     delay = np.random.uniform(5, 10, n_devices)
#     delay[list(slow_link)] = delay[list(slow_link)] + np.random.uniform(10, 20, len(slow_link))
#
#     bw = np.random.uniform(100, 200, n_devices)
#     bw[list(slow_link)] = np.random.uniform(20, 50, len(slow_link))
#
#     speed = np.random.uniform(1, 3, n_devices)
#     return delay, bw, speed
#
#
# def generate_program(n_operators, n_devices, seed, B=1000, l=100):
#     np.random.seed(seed)
#     G = nx.gnp_random_graph(n_operators - 2, 0.8, seed=seed, directed=True)
#     DAG = nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v])
#     DAG = nx.relabel.convert_node_labels_to_integers(DAG, first_label=1)
#     heads = [node for node in DAG.nodes() if DAG.in_degree(node) == 0]
#     tails = [node for node in DAG.nodes() if DAG.out_degree(node) == 0]
#
#     for n in heads:
#         DAG.add_edge(0, n)
#     for n in tails:
#         DAG.add_edge(n, n_operators - 1)
#
#     constraints = {}
#     n_types = n_devices // 5
#     groups = [set() for i in range(n_types)]
#     for i in range(n_devices):
#         groups[np.random.choice(n_types)].add(i)
#     k = len(groups)
#     for e in DAG.edges:
#         DAG.edges[e]['bytes'] = np.random.uniform(B/2, B)
#     for n in DAG.nodes:
#         DAG.nodes[n]['compute'] = np.random.exponential(l)
#         group_ids = np.random.choice(k, k // 2 + (np.random.sample() > 0.5) * 1 - (np.random.sample() > 0.5) * 1)
#         constraints[n] = list(set().union(*[groups[j] for j in group_ids]))
#         if not len(constraints[n]):
#             constraints[n] = np.random.choice(n_devices, n_devices//2, replace=False).tolist()
#     constraints[0] = [np.random.choice(constraints[0])]
#     constraints[n_operators - 1] = [np.random.choice(constraints[n_operators - 1])]
#     return DAG, constraints


def to_pickle(save_path, res):
    with open(save_path, 'wb') as handle:
        pickle.dump(res, handle, protocol = pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
    with open(path, 'rb') as handle:
        res = pickle.load(handle)
    return res

def load_weights(path):
    with open(path, 'rb') as handle:
        res = pickle.load(handle)
    comms, comps = res["comms"], res["comps"]
    return comms, comps

def load_dag_params(path):
    with open(path, 'rb') as handle:
        res = pickle.load(handle)
    widths, height = res["widths"], res['height']
    return widths, height

def save_dag(save_path, G):
    # pickle.dump(G, open(save_path, 'wb'))
    nx.write_gpickle(G, save_path)
    return

def load_dag(path):
    G = nx.read_gpickle(path)
    return G


def load_training_instance_for(v, seed, alpha, ccr, beta, n_devices):
    graph_path = os.path.join("./data", "dag_structure_v_{}".format(v),
                              "dag_structure_v_{}_seed_{}_alpha_{}.pkl".format(v, seed, alpha))
    param_path = os.path.join("./data", "dag_structure_v_{}".format(v),
                              "dag_params_{}_seed_{}_alpha_{}.pkl".format(v, seed, alpha))
    weight_path = os.path.join("./data", "weights_v_{}".format(v),
                               "weights_heterogeneous_v_{}_seed_{}_alpha_{}_ccr_{}_beta_{}_ndevices_{}.pkl".format(v,
                                                                                                                   seed,
                                                                                                                   alpha,
                                                                                                                   ccr,
                                                                                                                   beta,
                                                                                                                   n_devices))

    G = load_dag(graph_path)
    widths, height = load_dag_params(param_path)

    weights = load_pickle(weight_path)
    delay, bw, speed, compute, byte = weights['delay'], weights['bw'], weights['speed'], weights['compute'], weights[
        'byte']

    cnt = 1

    for i in range(height):
        for j in range(widths[i]):
            G.nodes[cnt]["compute"] = compute[i][j]
            cnt += 1

    total_operator = sum(widths)
    G.nodes[0]["compute"] = 0
    G.nodes[total_operator + 1]["compute"] = 0

    for e in G.edges:
        G.edges[e]['bytes'] = byte[e]

    return delay, bw, speed, G


def load_training_instance_for_v_ndevice(v, n_devices):
    delays = []
    bws = []
    speeds = []
    Gs = []

    ccr = [0.1, 0.5, 1.0, 5.0, 10.0]
    alpha = [0.5, 1.0, 2.0]
    beta = [0.1, 0.25, 0.5, 0.75, 1.0]

    for talpha in alpha:
        for tccr in ccr:
            for tbeta in beta:
                for seed in range(10):
                    delay, bw, speed, G = load_training_instance_for(v, seed, talpha, tccr, tbeta, n_devices)
                    delays.append(delay)
                    bws.append(bw)
                    speeds.append(speed)
                    Gs.append(G)
    return delays, bws, speeds, Gs


def visualize_dag(G, widths, height):
    pos = {}
    max_width = max(widths)
    max_height = height

    height_incr = 2 / (max_height + 1)
    width_incr = 2 / max_width

    total_operator = sum(widths)

    pos[0] = np.array([-1, -1])
    pos[total_operator + 1] = np.array([1, -1])

    cnt = 1
    cur_height = -1 + height_incr
    cur_width = -1
    for i in range(height):
        for j in range(widths[i]):
            pos[cnt] = np.array([cur_height, cur_width])
            cur_width += width_incr
            cnt += 1
        cur_width = -1
        cur_height += height_incr

    nx.draw(G, pos)
    plt.show()

    return


def network_fn_filter(network_path,
                   n_devices=[20],
                   type_probs=[0.2],
                   avg_speeds=[5],
                   avg_bws=[100],
                   avg_delays=[10],
                   b_bws=[0.2],
                   b_speeds=[0.2],
                   num_types=[5]):

    fns = os.listdir(network_path)
    res = []
    for fn in fns:
        if '.pkl' not in fn:
            continue
        token = fn.split("_")
        ndevice = int(token[1])
        ntype = int(token[3])
        speed = int(token[5])
        bw = int(token[7])
        delay = int(token[9])
        tprob = float(token[11])
        bbw = float(token[13])
        bspeed = float(token[15][:-4])
        if ndevice not in n_devices:
            continue
        elif ntype not in num_types:
            continue
        elif tprob not in type_probs:
            continue
        elif speed not in avg_speeds:
            continue
        elif bw not in avg_bws:
            continue
        elif delay not in avg_delays:
            continue
        elif bbw not in b_bws:
            continue
        elif bspeed not in b_speeds:
            continue
        else:
            res.append(fn)

    return res


def program_fn_filter(op_path,
                   vs=[20],
                   alphas=[0.2],
                   connect_probs=[0.2],
                   avg_computes=[100],
                   avg_bytes=[10],
                   b_comps=[0.2],
                   b_comms=[0.2],
                   num_types=[5]):

    fns = os.listdir(op_path)
    res = []
    for fn in fns:
        if '.pkl' not in fn:
            continue
        token = fn.split("_")
        v = int(token[1])
        alpha = float(token[3])
        connp = float(token[5])
        ntype = int(token[7])
        compute = int(token[9])
        byte = int(token[11])
        bcomp = float(token[13])
        bcomm = float(token[15][:-4])
        if v not in vs:
            continue
        elif alpha not in alphas:
            continue
        elif connp not in connect_probs:
            continue
        elif ntype not in num_types:
            continue
        elif compute not in avg_computes:
            continue
        elif byte not in avg_bytes:
            continue
        elif bcomp not in b_comps:
            continue
        elif bcomm not in b_comms:
            continue
        else:
            res.append(fn)

    return res

