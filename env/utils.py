import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
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
            constraints[n] = np.random.choice(n_devices, n_devices//2, replace=False).tolist()
    constraints[0] = [np.random.choice(constraints[0])]
    constraints[n_operators - 1] = [np.random.choice(constraints[n_operators - 1])]
    return DAG, constraints


# Generator based on https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=993206
# v: number of tasks
# alpha: shape of graph
# out_degree: out degree of a node
# ccr: communication to computation ratio
# beta: range percentage of computation costs on processors
# avg_comm: average communication cost
# seed: random seed
def graph_dag_structure(v,
                      alpha,
                      seed,
                      save_path = None,
                      visualize = False):
    np.random.seed(seed)

    height_mean = np.sqrt(v) / alpha
    height = int(np.ceil(np.random.uniform(0, 2 * height_mean)))

    width_mean = alpha * np.sqrt(v)
    widths = []

    for i in range(height):
        widths.append(int(np.ceil(np.random.uniform(0, 2 * width_mean))))

    total_operator = sum(widths)

    G = nx.DiGraph()
    G.add_node(0)
    G.add_node(total_operator + 1)
    cnt = 1
    nodes = [[] for i in range(height + 2)]
    nodes[0].append(0)
    for i in range(height):
        for j in range(widths[i]):
            G.add_node(cnt)
            if i == 0:
                G.add_edge(0, cnt)
                nodes[i + 1].append(cnt)
            else:
                start_idx_last_layer = sum(widths[:i - 1]) + 1
                end_idx_last_layer = start_idx_last_layer + widths[i - 1] - 1
                nodes[i + 1].append(cnt)
                for node in range(start_idx_last_layer, end_idx_last_layer + 1):
                    p = np.random.binomial(1, 0.8)
                    if p:
                        G.add_edge(node, cnt)
            cnt += 1

    nodes[-1].append(total_operator + 1)
    end_start_idx = sum(widths[:-1])
    for i in range(widths[-1]):
        node = end_start_idx + i
        G.add_edge(node, total_operator + 1)

    # Valid checking, if any node in the middle is not connected
    # with any node in the following layer, randomly connect
    for i, layer in enumerate(nodes):
        for node in layer:
            if i != len(nodes) - 1:
                if G.out_degree(node) == 0:
                    choice = np.random.choice(nodes[i + 1])
                    G.add_edge(node, choice)

    for i, layer in enumerate(nodes):
        for node in layer:
            if i != 0:
                if G.in_degree(node) == 0:
                    choice = np.random.choice(nodes[i - 1])
                    G.add_edge(choice, node)

    if visualize:
        visualize_dag(G, widths, height)
    if save_path:
        graph_path = os.path.join(save_path,"dag_structure_{}_seed_{}_v_{}_alpha_{}.pkl".format(v,seed,v,alpha))
        params_path = os.path.join(save_path, "dag_params_{}_seed_{}_v_{}_alpha_{}.pkl".format(v, seed,v,alpha))
        params = {"widths": widths,
                  "height": height}
        save_dag(graph_path, G)
        to_pickle(params_path, params)
    return G

def generate_dag_weight(graph_path,
                        params_path,
                        v,
                        ccr,
                        seed,
                        avg_comm = 1000,
                        save_path = None):
    np.random.seed(seed)
    G = load_dag(graph_path)
    widths, height = load_dag_params(params_path)

    avg_comm = np.random.normal(loc=avg_comm, scale=avg_comm / 3)
    avg_comp = avg_comm / ccr

    comps = [[] for i in range(height)]
    for i in range(height):
        for j in range(widths[i]):
            comps[i].append(np.random.uniform(0, 2 * avg_comp))

    comms = {}
    for edge in G.edges:
        comms[edge] = np.random.uniform(0, 2 * avg_comm)

    if save_path:
        weight_path = save_path + "dag_weights_{}_seed_{}.pkl".format(v, seed)
        res = {"comps": comps,
               "comms": comms}
        to_pickle(weight_path, res)
    return comps, comms

def generate_dag_weight_for_heterogeneous_devices(graph_path,
                        params_path,
                        v,
                        ccr,
                        seed,
                        beta,
                        device_list,
                        avg_comm = 1000,
                        save_path = None):
    np.random.seed(seed)
    G = load_dag(graph_path)
    widths, height = load_dag_params(params_path)

    avg_comm = np.random.normal(loc=avg_comm, scale=avg_comm / 3)
    avg_comp = avg_comm / ccr

    comps = [[] for i in range(height)]
    for i in range(height):
        for j in range(widths[i]):
            node_mean = np.random.uniform(0, 2 * avg_comp)
            tmp = []
            for device in range(len(device_list)):
                tmp.append(np.random.uniform(node_mean*(1-beta/2), node_mean*(1+beta/2)))
            comps[i].append(tmp)
    comms = {}
    for edge in G.edges:
        comms[edge] = np.random.uniform(0, 2 * avg_comm)

    if save_path:
        weight_path = save_path + "dag_weights_heterogeneous_{}_seed_{}.pkl".format(v, seed)
        res = {"comps": comps,
               "comms": comms}
        to_pickle(weight_path, res)
    return comps, comms


def generate_weights_for_heterogeneous_devices(graph_path,
                                               params_path,
                                               v,
                                               ccr,
                                               seed,
                                               beta,
                                               n_devices,
                                               avg_comm=1000,
                                               avg_speed=3,
                                               avg_bw=200,
                                               avg_delay=10,
                                               save_path=None):
    np.random.seed(seed)

    G = load_dag(graph_path)
    widths, height = load_dag_params(params_path)

    avg_comm = np.random.normal(loc=avg_comm, scale=avg_comm / 3)
    avg_comp = avg_comm / ccr

    # Copmutation cost for each dag node on each device
    # Structure: list[list[list]]
    comps = [[] for i in range(height)]

    for i in range(height):
        for j in range(widths[i]):
            node_mean = np.random.uniform(0, 2 * avg_comp)
            tmp = []
            for device in range(n_devices):
                tmp.append(np.random.uniform(node_mean * (1 - beta / 2), node_mean * (1 + beta / 2)))
            comps[i].append(tmp)

    # Communication cost for each dag edge
    # Structure: dict, key: edge -> cost
    comms = {}
    for edge in G.edges:
        comms[edge] = np.random.uniform(0, 2 * avg_comm)

    # Delay for each device
    delay = np.random.uniform(0, 2 * avg_delay, n_devices)

    # bw for each device
    bw = np.random.uniform(0, 2 * avg_bw, n_devices)

    # speed for each device
    speed = np.random.uniform(0, 2 * avg_speed, n_devices)

    # Computational amount for each dag node.
    compute = [[] for i in range(height)]
    for i in range(height):
        for j in range(widths[i]):
            accum_compute = 0
            for k in range(n_devices):
                accum_compute += comps[i][j][k] * speed[k]
            avg_compute = accum_compute / n_devices
            compute[i].append(avg_compute)
            assert (compute[i][j] > 0, "Compute should be large than 0")

    # Communicational bytes for each dag link
    byte = {}
    for edge in G.edges:
        byte[edge] = (comms[edge] - avg_delay) * avg_bw
        if byte[edge] < 0:
            byte[edge] = 1
        assert (byte[edge] > 0), "Bytes should be large than 0"

    if save_path:
        weight_path = os.path.join(save_path,
                                   "weights_heterogeneous_v_{}_seed_{}_ccr_{}_beta_{}_ndevices_{}.pkl".format(v, seed,
                                                                                                              ccr, beta,
                                                                                                              n_devices))
        res = {"comps": comps,
               "comms": comms,
               "delay": delay,
               "bw": bw,
               "speed": speed,
               "compute": compute,
               "byte": byte}
        to_pickle(weight_path, res)


def load_graph_with_weights(graph_path, weight_path, params_path):
    G  = load_dag(graph_path)
    comms, comps = load_weights(weight_path)
    widths, height = load_dag_params(params_path)

    cnt = 1
    for i in range(height):
        for j in range(widths[i]):
            G.nodes[cnt]['compute'] = comps[i][j]
            cnt += 1

    for e in G.edges:
        G.edges[e]['bytes'] = comms[e]

    return G


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


def data_fn_filter(op_network_path,
                   v_range=[60, 100],
                   alpha_range=[0.5, 0.5],
                   seed_range=[1, 10],
                   ccr_range=[1.0, 1.0],
                   beta_range=[0.25, 0.25],
                   comm_range=[1000, 1000]):
    eps = 1e-5
    fns = os.listdir(op_network_path)
    res = []
    for fn in fns:
        token = fn.split("_")
        if float(token[1]) < v_range[0] or float(token[1]) > v_range[1]:
            continue
        elif float(token[3]) < alpha_range[0] or float(token[3]) > alpha_range[1]:
            continue
        elif float(token[5]) < seed_range[0] or float(token[5]) > seed_range[1]:
            continue
        elif float(token[7]) < ccr_range[0] or float(token[7]) > ccr_range[1]:
            continue
        elif float(token[9]) < beta_range[0] or float(token[9]) > beta_range[1]:
            continue
        elif float(token[11].split(".")[0]) < comm_range[0] or float(token[11].split(".")[0]) > comm_range[1]:
            continue
        else:
            res.append(fn)

    return res




