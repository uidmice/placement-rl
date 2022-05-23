import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import dgl
import math
import dgl.function as fn
import pdb


from placement_rl.primative_nn import FNN
from env.latency import *

epsilon = 1e-6
torch.autograd.set_detect_anomaly(True)


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

class Aggregator(nn.Module):
    def __init__(self,
                 node_dim,
                 edge_dim,
                 out_dim,
                 reverse,
                 device):
        super(Aggregator, self).__init__()

        self.reverse = reverse

        self.pre_layer = FNN(node_dim + edge_dim, [], node_dim + edge_dim).to(device)
        self.update_layer = FNN(node_dim + edge_dim, [], out_dim).to(device)


    def msg_func(self, edges):
        msg = F.relu(self.pre_layer(torch.cat([edges.src['y'], edges.data['x']], dim=1)))
        return {'m': msg}


    def node_update(self, nodes):
        h = F.relu(self.update_layer(nodes.data['z']))
        return {'h': h}

    def forward(self, g):
        if self.reverse:
            g = dgl.reverse(g, copy_edata=True)
        g.update_all(self.msg_func, fn.mean('m', 'z'), self.node_update)
        h = g.ndata.pop('h')
        return h

class GiPHEmbedding_radial(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim, k, device):
        super(GiPHEmbedding_radial, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.k = k

        self.fpa = Aggregator(out_dim, edge_dim, out_dim, False, device).to(device)
        self.bpa = Aggregator(out_dim, edge_dim, out_dim, True, device).to(device)

        self.node_transform = FNN(node_dim, [node_dim], out_dim).to(device)

    def forward(self, g):
        self_trans = self.node_transform(g.ndata['x'].clone())
        def message_pass(agg, sink):
            g.ndata['y'] = self_trans.clone()
            for i in range(self.k):
                h = agg(g).clone()
                h[sink, :] = 0
                h = self_trans + h
                g.ndata['y'] = h
            return g.ndata['y']

        out_fpa = message_pass(self.fpa, len(g.nodes())-1)
        out_bpa = message_pass(self.bpa, 0)
        return torch.cat([out_fpa, out_bpa], dim=1)


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

class PlacementAgent_Radial:
    def __init__(self, node_dim, edge_dim, out_dim,
                 device,
                 hidden_dim=32,
                 lr=0.03,
                 gamma=0.95,
                 k=5):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.lr = lr
        self.gamma = gamma

        self.device = device

        self.embedding = GiPHEmbedding_radial(node_dim, edge_dim, out_dim, k, device=device)


        self.policy = SoftmaxActor(2 * out_dim, device, hidden_dim)
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
        if not parents:
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