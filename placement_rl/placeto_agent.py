import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import networkx as nx
import pdb

# from placement_rl.rl_agent import SoftmaxActor
# from env.latency import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epsilon = 1e-6
torch.autograd.set_detect_anomaly(True)


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class Device_SoftmaxActor(nn.Module):
    def __init__(self,
                 input_dim,
                 n_device,
                 hidden_dim=32,
                 ):
        super(Device_SoftmaxActor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_device = n_device

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, n_device)

        self.apply(weights_init_)

    def forward(self, x, mask=None):
        # pdb.set_trace()
        x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        x = self.linear3(x)

        if mask is None:
            mask = [1 for i in range(x.shape[0])]
        x_masked = x.clone()
        x_masked[mask == 0] = -float('inf')
        return F.softmax(torch.squeeze(x_masked), dim=-1)


class MP(nn.Module):
    def __init__(self,
                 node_dim,
                 edge_dim,
                 out_dim,
                 reverse,
                 first_layer):
        super(MP, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.reverse = reverse
        self.first_layer = first_layer

        if first_layer:
            self.pre_layers = nn.Sequential(
                nn.Linear(node_dim, node_dim),
                nn.ReLU())

            self.update_layers = nn.Sequential(
                nn.Linear(2*node_dim, node_dim),
                nn.ReLU(),
                nn.Linear(node_dim, node_dim))
        else:
            self.pre_layers = nn.Sequential(
                nn.Linear(2*node_dim, 2*node_dim),
                nn.ReLU())

            self.update_layers = nn.Sequential(
                nn.Linear(4*node_dim, node_dim),
                nn.ReLU(),
                nn.Linear(node_dim, node_dim))

        self.apply(weights_init_)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.pre_layers)
        reset(self.update_layers)

    def msg_func(self, edges):
        # pdb.set_trace()
        msg = self.pre_layers(edges.src['x'])
        return {'m': msg}

    def reduce_func(self, nodes):
        # pdb.set_trace()
        z = torch.sum((nodes.mailbox['m']), 1)
        return {'z': z}

    def node_update(self, nodes):
        # pdb.set_trace()
        h = torch.cat([nodes.data['x'],
                       nodes.data['z']],
                    dim=1)
        h = self.update_layers(h)
        return {'h': h}

    def forward(self, g):
        if self.first_layer:
            g.ndata['z'] = torch.randn(g.num_nodes(), self.node_dim).to(device)
        else:
            g.ndata['z'] = torch.randn(g.num_nodes(), self.node_dim * 2).to(device)
        dgl.prop_nodes_topo(g, self.msg_func,
                            self.reduce_func,
                            self.reverse,
                            self.node_update)
        h = g.ndata['h']
        return h



class MP_preprocessing(nn.Module):
    def __init__(self):
        super(MP_preprocessing, self).__init__()

    def reset_parameters(self):
        return

    def msg_func(self, edges):
        # pdb.set_trace()
        msg = edges.data['x']

        # pdb.set_trace()
        if not msg.shape[0] > 0:
            return {'m': torch.zeros(1).to(device)}
        else:
            return {'m': msg[:,0]}

    def reduce_func(self, nodes):
        # pdb.set_trace()
        z = torch.mean((nodes.mailbox['m']).squeeze(), 0).unsqueeze(0).unsqueeze(0)
        # pdb.set_trace()
        return {'z': z}

    def node_update(self, nodes):
        # pdb.set_trace()
        return {'byte': nodes.data['z']}

    def forward(self, g):
        g.ndata['z'] = torch.zeros(g.num_nodes()).to(device)
        g.ndata['byte'] = torch.zeros(g.num_nodes()).to(device)
        dgl.prop_nodes_topo(g, self.msg_func,
                            self.reduce_func,
                            False,
                            self.node_update)
        h = g.ndata['byte']
        return h


class PlaceToAgent:
    def __init__(self,
                 node_dim,
                 edge_dim,
                 out_dim,
                 n_device,
                 hidden_dim=32,
                 lr=0.03,
                 gamma=0.95):

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.n_device = n_device
        self.hidden_dim = hidden_dim

        self.lr = lr
        self.gamma = gamma

        self.mapping_embedding = torch.nn.Embedding(n_device, 1).to(device)
        self.node_cur_embedding = torch.nn.Embedding(2, 1).to(device)
        self.node_done_embedding = torch.nn.Embedding(2, 1).to(device)

        self.MP_forward_1 = MP(node_dim, edge_dim, out_dim // 2, reverse=True, first_layer = True).to(device)
        self.MP_reverse_1 = MP(node_dim, edge_dim, out_dim // 2, reverse=False, first_layer = True).to(device)

        self.MP_forward_2 = MP(node_dim, edge_dim, out_dim // 2, reverse=True, first_layer = False).to(device)
        self.MP_reverse_2 = MP(node_dim, edge_dim, out_dim // 2, reverse=False, first_layer = False).to(device)

        self.MP_forward_3 = MP(node_dim, edge_dim, out_dim // 2, reverse=True, first_layer = False).to(device)
        self.MP_reverse_3 = MP(node_dim, edge_dim, out_dim // 2, reverse=False, first_layer = False).to(device)

        self.mp_preprocess = MP_preprocessing()

        self.pred_net_prev = nn.Sequential(
            nn.Linear(node_dim*2, node_dim),
            nn.ReLU()
        ).to(device)


        self.desc_net_prev = nn.Sequential(
            nn.Linear(node_dim*2, node_dim),
            nn.ReLU()
        ).to(device)


        self.parallel_net_prev = nn.Sequential(
            nn.Linear(node_dim*2, node_dim),
            nn.ReLU()
        ).to(device)

        self.policy = Device_SoftmaxActor(node_dim * 5, n_device, hidden_dim).to(device)
        self.optim = torch.optim.Adam(list(self.MP_forward_1.parameters()) +
                                      list(self.MP_reverse_1.parameters()) +
                                      list(self.MP_forward_2.parameters()) +
                                      list(self.MP_reverse_2.parameters()) +
                                      list(self.MP_forward_3.parameters()) +
                                      list(self.MP_reverse_3.parameters()) +
                                      list(self.pred_net_prev.parameters()) +
                                      list(self.desc_net_prev.parameters()) +
                                      list(self.parallel_net_prev.parameters()) +
                                      list(self.policy.parameters()), lr=lr)

        self.selected_sequence = []
        self.target_device_sequence = []
        self.saved_rewards = []
        self.log_probs = []

        self.available_ops = None
        self.calculated_byte = None
        self.calculated_compute = None

    def reset(self):
        self.selected_sequence = []
        self.available_ops = None
        self.calculated_byte = None
        self.calculated_compute = None

    def op_selection(self, g, program, cur_mapping):
        # The first selection on this op
        # pdb.set_trace()
        if self.available_ops is None:
            self.available_ops = [*range(g.number_of_nodes())]

            self.calculated_compute = []
            for node in program.P.nodes:
                program.P.nodes[node]["byte"] = 0
                program.P.nodes[node]['n_out'] = 0
                self.calculated_compute.append(program.P.nodes[node]["compute"])

            for edge in program.P.edges:
                src = edge[0]
                program.P.nodes[src]['byte'] += program.P.edges[edge]['bytes']
                program.P.nodes[src]["n_out"] += 1

            self.calculated_byte = []
            for node in program.P.nodes:
                if program.P.nodes[node]['n_out'] != 0:
                    program.P.nodes[node]["byte"] /= program.P.nodes[node]['n_out']
                self.calculated_byte.append(program.P.nodes[node]["byte"])

            # bytes = self.mp_preprocess(g)



        selected_op = np.random.choice(self.available_ops)
        self.available_ops.remove(selected_op)
        self.selected_sequence.append(selected_op)

        mapping = torch.tensor(cur_mapping).to(device)
        mapping = self.mapping_embedding(mapping)

        node_cur = torch.zeros([program.n_operators, 1], dtype = torch.long).to(device)
        node_cur[selected_op] = 1
        node_cur = self.node_cur_embedding(node_cur).squeeze().unsqueeze(1)

        node_done = torch.zeros([program.n_operators, 1], dtype = torch.long).to(device)
        for idx in self.selected_sequence:
            # pdb.set_trace()
            if idx != selected_op:
                node_done[idx] = 1
        node_done = self.node_done_embedding(node_done).squeeze().unsqueeze(1)

        bytes = torch.tensor(self.calculated_byte).to(device)
        bytes = bytes.reshape(-1, 1)
        computes = torch.tensor(self.calculated_compute).to(device)
        computes = computes.reshape(-1, 1)

        feats = torch.cat([computes, bytes, mapping, node_cur, node_done], axis = 1)

        g.ndata['x'] = feats

        if len(self.available_ops) > 0:
            end = False
        else:
            end = True

        return selected_op, end

    def dev_selection(self,
                      g,
                      program,
                      network,
                      op):
        op_constraints = program.placement_constraints
        constraint = op_constraints[op]
        dev_constraints = network.device_constraints


        # Mask available device.
        mask = [0 for _ in range(self.n_device)]
        for dev in dev_constraints:
            tmp = dev_constraints[dev]
            if constraint in tmp:
                mask[dev] = 1

        hf_1 = self.MP_forward_1(g)
        hb_1 = self.MP_reverse_1(g)
        h_1 = torch.cat([hf_1, hb_1], dim=1)
        g.ndata['x'] = h_1

        hf_2 = self.MP_forward_2(g)
        hb_2 = self.MP_reverse_2(g)
        h_2 = torch.cat([hf_2,hb_2], dim = 1)
        g.ndata['x'] = h_2

        hf_3 = self.MP_forward_3(g)
        hb_3 = self.MP_reverse_3(g)
        h = torch.cat([hf_3, hb_3], dim=1)
        g.ndata['x'] = h

        all_nodes = [*range(program.n_operators)]

        node_embedding = h[op]
        all_nodes.remove(op)

        # Find predecessors
        op_graph = program.P
        predecessors = list(nx.ancestors(op_graph, op))
        all_nodes = [i for i in all_nodes if i not in predecessors]
        pred_embeddings = self.pred_net_prev(h[predecessors])
        pred_embeddings = torch.sum(pred_embeddings, dim = 0)

        # Find descendants
        descendants = list(nx.descendants(op_graph, op))
        all_nodes = [i for i in all_nodes if i not in descendants]
        desc_embeddings = self.desc_net_prev(h[descendants])
        desc_embeddings = torch.sum(desc_embeddings, dim = 0)

        # Parallels
        if len(all_nodes) == 0:
            parallel_embeddings = torch.zeros(h.shape[1]//2).to(device)
        else:
            parallel_embeddings = self.parallel_net_prev(h[all_nodes])
            parallel_embeddings = torch.sum(parallel_embeddings, dim = 0)

        embedding = torch.cat((node_embedding, pred_embeddings, desc_embeddings, parallel_embeddings), dim = 0)


        probs = self.policy(embedding, mask)

        device_distribution = torch.distributions.Categorical(probs = probs)
        target_device = device_distribution.sample()
        self.log_probs.append(device_distribution.log_prob(target_device))
        self.target_device_sequence.append(target_device)

        return target_device.item()


    def finish_episode(self, update_network=True, use_baseline=True):
        if update_network:
            R = 0
            policy_loss = 0
            returns = []

            for r in self.saved_rewards[::1]:
                R = r + self.gamma * R
                returns.insert(0, R)

            if use_baseline:
                for i in range(len(self.saved_rewards)):
                    if i == 0:
                        bk = self.saved_rewards[0]
                    else:
                        try:
                            bk = sum(self.saved_rewards[:i+1]) / len(self.saved_rewards[:i+1])
                        except:
                            bk = sum(self.saved_rewards) / len(self.saved_rewards)
                    returns[i] -= bk
            returns = torch.tensor(returns).to(device)
            self.optim.zero_grad()
            for log_prob, R in zip(self.log_probs, returns):
                policy_loss = policy_loss - log_prob * R
            policy_loss.backward()
            self.optim.step()

            self.available_ops = None
            del self.saved_rewards[:]
            del self.log_probs[:]
