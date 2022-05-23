import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import dgl
import dgl.function as fn
import math
import pdb


from placement_rl.primative_nn import FNN
from env.latency import *

epsilon = 1e-6
torch.autograd.set_detect_anomaly(True)
# # Soft update of target critic network
# def soft_update(target, source, tau):
#     for target_param, param in zip(target.parameters(), source.parameters()):
#         target_param.data.copy_(target_param.data * (1.0 - tau) +
#                                 param.data * tau)
#
# # Hard update of target critic network
# def hard_update(target, source):
#     for target_param, param in zip(target.parameters(), source.parameters()):
#         target_param.data.copy_(param.data)

GNN_MSG_KEY = 'm'
GNN_NODE_FEAT_IN_KEY = 'hn_in'
GNN_NODE_FEAT_OUT_KEY = 'hn_out'
GNN_EDGE_FEAT_KEY = 'he'
GNN_AGG_MSG_KEY = 'h_msg'
GNN_NODE_NORM = 'norm'
GNN_NODE_LABELS_KEY = 'hnl'
GNN_NODE_ATTS_KEY = 'hna'
GNN_EDGE_LABELS_KEY = 'hel'
GNN_EDGE_NORM = 'norm'

GRAPH_CLASSIFICATION = 'graph_classification'
NODE_CLASSIFICATION = 'node_classification'

AIFB = 'aifb'
MUTAG = 'mutag'
MUTAGENICITY = 'mutagenicity'
PTC_FM = 'ptc_fm'
PTC_FR = 'ptc_fr'
PTC_MM = 'ptc_mm'
PTC_MR = 'ptc_mr'


def reset_graph_features(g):
    keys = [GNN_NODE_FEAT_IN_KEY, GNN_AGG_MSG_KEY, GNN_MSG_KEY, GNN_NODE_FEAT_OUT_KEY]
    for key in keys:
        if key in g.ndata:
            del g.ndata[key]
    if GNN_EDGE_FEAT_KEY in g.edata:
        del g.edata[GNN_EDGE_FEAT_KEY]

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

def init_weights(m):
    if isinstance(m, Linear):
        stdv = 1. / math.sqrt(m.weight.size(1))
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)


class MPLayer(nn.Module):
    def __init__(self,
                 node_dim,
                 edge_dim,
                 out_dim,
                 device):
        super(MPLayer, self).__init__()

        self.pre_layer = FNN(node_dim + edge_dim, [], node_dim + edge_dim).to(device)
        self.update_layer = FNN(node_dim + edge_dim, [], out_dim).to(device)

    def msg_func(self, edges):
        msg = F.relu(self.pre_layer(torch.cat([edges.src['y'], edges.data['x']], dim=1)))
        return {'m': msg}

    def reduce_func(self, nodes):
        z = torch.sum((nodes.mailbox['m']), 1)
        return {'z': z}

    def node_update(self, nodes):
        h = F.relu(self.update_layer(nodes.data['z'])) + nodes.data['y']
        return {'y': h}

    def forward(self, g,  reverse):
        dgl.prop_nodes_topo(g, self.msg_func, fn.mean('m', 'z'), reverse, self.node_update)
        h = g.ndata.pop('y')
        return h

class GiPHEmbedding(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim, device):
        super(GiPHEmbedding, self).__init__()
        self.node_dim = node_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim

        self.fmp = MPLayer(out_dim//2, edge_dim, out_dim//2, device)
        self.bmp = MPLayer(out_dim//2, edge_dim, out_dim//2, device)

        self.node_transform = FNN(node_dim, [node_dim], out_dim//2).to(device)


    def forward(self,  g):
        g.ndata['y'] = self.node_transform(g.ndata['x'].clone())
        hf = self.fmp(g, False)
        g.ndata['y'] = self.node_transform(g.ndata['x'].clone())
        hb = self.bmp(g, True)
        return torch.cat([hf, hb], dim=1)


class edGNNLayer(nn.Module):
    def __init__(self,
                 node_dim,
                 edge_dim,
                 out_feats,
                 activation=None,
                 bias=None):

        super(edGNNLayer, self).__init__()

        # 1. set parameters
        self.node_dim = node_dim
        self.out_feats = out_feats
        self.activation = activation
        self.edge_dim = edge_dim
        self.bias = bias

        # 2. create variables
        self._build_parameters()

        # 3. initialize variables
        self.apply(init_weights)

    def reset_parameters(self):
        reset(self.linear)

    def _build_parameters(self):
        input_dim = 2 * self.node_dim
        if self.edge_dim is not None:
            input_dim = input_dim + self.edge_dim

        self.linear = nn.Linear(input_dim, self.out_feats, bias=self.bias)


    def gnn_msg(self, edges):
        if self.g.edata is not None:
            msg = torch.cat([edges.src[GNN_NODE_FEAT_IN_KEY],
                             edges.data[GNN_EDGE_FEAT_KEY]],
                            dim=1)
        else:
            msg = edges.src[GNN_NODE_FEAT_IN_KEY]
        return {GNN_MSG_KEY: msg}

    def gnn_reduce(self, nodes):
        accum = torch.sum((nodes.mailbox[GNN_MSG_KEY]), 1)
        return {GNN_AGG_MSG_KEY: accum}

    def node_update(self, nodes):
        h = torch.cat([nodes.data[GNN_NODE_FEAT_IN_KEY],
                       nodes.data[GNN_AGG_MSG_KEY]],
                      dim=1)
        h = self.linear(h)

        if self.activation:
            h = self.activation(h)

        return {GNN_NODE_FEAT_OUT_KEY: h}

    def forward(self, node_features, edge_features, g):

        if g is not None:
            self.g = g

        # 1. clean graph features
        reset_graph_features(self.g)

        # 2. set current iteration features
        self.g.ndata[GNN_NODE_FEAT_IN_KEY] = node_features
        self.g.edata[GNN_EDGE_FEAT_KEY] = edge_features

        # 3. aggregate messages
        self.g.update_all(self.gnn_msg,
                          self.gnn_reduce,
                          self.node_update)

        h = self.g.ndata.pop(GNN_NODE_FEAT_OUT_KEY)
        return h



class EdGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim, device, hidden_dim = 128):
        super(EdGNN, self).__init__()
        self.node_dim = node_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.edgnn_1 = edGNNLayer(node_dim, edge_dim, hidden_dim, nn.ReLU()).to(device)
        self.edgnn_2 = edGNNLayer(hidden_dim, edge_dim, hidden_dim, nn.ReLU()).to(device)
        self.edgnn_3 = edGNNLayer(hidden_dim, edge_dim, hidden_dim, nn.ReLU()).to(device)
        self.edgnn_4 = edGNNLayer(hidden_dim, edge_dim, out_dim, None).to(device)


    def forward(self,  g):
        node_feature = g.ndata['x']
        edge_feature = g.edata['x']

        h = self.edgnn_1(node_feature, edge_feature, g)
        h = self.edgnn_2(h, edge_feature, g)
        h = self.edgnn_3(h, edge_feature, g)
        h = self.edgnn_4(h, edge_feature, g)

        return h


class SoftmaxActor(nn.Module):
    def __init__(self,
                 input_dim,
                 device,
                 hidden_dim=32,
                 ):
        super(SoftmaxActor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.nn = FNN(input_dim, [hidden_dim], 1).to(device)


    def forward(self, x, mask=None):
        x = self.nn(x)
        if mask is None:
            mask = [1 for i in range(x.shape[0])]
        x_masked = x.clone()
        x_masked[mask == 0] = -float('inf')
        return F.softmax(torch.squeeze(x_masked), dim=-1)


class PlacementAgent:
    def __init__(self, node_dim, edge_dim, out_dim,
                 device,
                 hidden_dim=32,
                 lr=0.03,
                 gamma=0.95,
                 use_edgnn = False):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.use_edgnn = use_edgnn

        self.lr = lr
        self.gamma = gamma

        self.device = device

        if not use_edgnn:
            self.embedding = GiPHEmbedding(node_dim, edge_dim, out_dim, device=device)
        else:
            self.embedding = EdGNN(node_dim, edge_dim, out_dim, device)

        self.policy = SoftmaxActor(out_dim, device, hidden_dim)
        self.optim = torch.optim.Adam(list(self.embedding.parameters()) + list(self.policy.parameters()), lr=lr)
        self.log_probs = []

        self.saved_rewards = []


    def op_selection(self, g, mask=None):
        u = self.embedding(g)
        probs = self.policy(u, mask)
        m = torch.distributions.Categorical(probs=probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()

    def op_dev_selection(self, g, action_dict, mask=None):
        # pdb.set_trace()
        placement_embedding = self.embedding(g)
        probs = self.policy(placement_embedding, mask)
        m = torch.distributions.Categorical(probs=probs)

        a = m.sample()
        self.log_probs.append(m.log_prob(a))
        return action_dict[a.item()]

    def dev_selection_eft(self, program, network, map:list, G_stats, op, options):
        est = {}
        parents = program.op_parents[op]
        end_time = np.array([np.average(G_stats.nodes[p]['end_time']) for p in parents])
        if len(parents) == 0:
            return random.choice(options)
        for dev in options:
            c_time = np.array([communication_latency(program, network, p, op, map[p], dev) for p in parents])
            est[dev] = np.max(c_time + end_time) + computation_latency(program, network, op, dev)
        return min(est, key=est.get)

    def dev_selection_greedy(self, program, network, map:list, op, options, noise=0):
        lat = {}
        for d in options:
            map[op]=d
            latency = evaluate(map, program, network, noise)
            lat[d] = latency
        best = min(lat.values())
        return [d for d in options if lat[d]==best]

    def finish_episode(self, update_network=True,  use_baseline=True):
        if update_network:
            R = 0
            policy_loss = 0

            returns = []
            for r in self.saved_rewards[::-1]:
                R = r + self.gamma * R
                returns.insert(0, R)

            if use_baseline:
                for i in range(len(self.saved_rewards)):
                    if i == 0:
                        bk = self.saved_rewards[0]
                    else:
                        try:
                            bk = sum(self.saved_rewards[:i + 1]) / len(self.saved_rewards[:i + 1])
                        except:
                            bk = sum(self.saved_rewards) / len(self.saved_rewards)
                    returns[i] -= bk

            returns = torch.tensor(returns, device=self.device)
            # returns = (returns - returns.mean()) / (returns.std() + epsilon)

            self.optim.zero_grad()
            for log_prob, R in zip(self.log_probs, returns):
                policy_loss = policy_loss - log_prob * R
            policy_loss.backward()
            self.optim.step()

        del self.saved_rewards[:]
        del self.log_probs[:]

# class DevNet(nn.Module):
#     def __init__(self,
#                  in_dim,
#                  out_dim,
#                  num_heads=1,
#                  activation=None):
#         super(DevNet, self).__init__()
#
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.num_heads = num_heads
#         self.forward_layer = GATConv (in_dim, out_dim, num_heads=num_heads, activation=activation, allow_zero_in_degree=True)
#         self.backward_layer = GATConv (in_dim, out_dim, num_heads=num_heads,activation=activation, allow_zero_in_degree=True)
#
#
#     def forward(self, graph, feat, op, parallel):
#         n = graph.batch_size
#         if n == 1:
#             fh = torch.mean(self.forward_layer(graph, feat), 1)[op]
#             bh = torch.mean(self.backward_layer(dgl.reverse(graph), feat), 1)[op]
#             para = torch.sum(feat[parallel], 0)
#             return torch.cat([fh, bh, feat[op], para])
#
#         m = graph.num_nodes()//n  # number of nodes per graph
#         idx = list(range(op, graph.num_nodes() , m))
#         fh = torch.mean(self.forward_layer(graph, feat), 1)[idx]
#         bh = torch.mean(self.backward_layer(dgl.reverse(graph), feat), 1)[idx]
#
#         para = [torch.sum(feat[parallel], 0)]
#         for i in range(m, graph.num_nodes(), m):
#             para_idx = [a + i for a in parallel]
#             para.append(torch.sum(feat[para_idx], 0))
#
#         para = torch.stack(para)
#         return torch.cat([fh, bh, feat[idx], para], dim=1)