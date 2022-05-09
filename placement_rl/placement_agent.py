import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv


from placement_rl.primative_nn import FNN
from env.latency import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

class MPLayer(nn.Module):
    def __init__(self,
                 node_dim,
                 edge_dim,
                 out_dim):
        super(MPLayer, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim

        self.pre_layer = FNN(node_dim + edge_dim, [], node_dim + edge_dim)
        self.update_layer = FNN(2 * node_dim + edge_dim, [], out_dim)

    def msg_func(self, edges):
        msg = self.pre_layer(torch.cat([edges.src['x'], edges.data['x']], dim=1))
        return {'m': msg}

    def reduce_func(self, nodes):
        z = torch.sum((nodes.mailbox['m']), 1)
        return {'z': F.relu(z)}

    def node_update(self, nodes):
        h = torch.cat([nodes.data['x'],
                       nodes.data['z']],
                      dim=1)
        h = self.update_layer(h)
        return {'h': F.relu(h)}

    def forward(self, g,  reverse):
        g.ndata['z'] = torch.rand(g.num_nodes(), self.node_dim + self.edge_dim).to(device)
        dgl.prop_nodes_topo(g, self.msg_func,
                            self.reduce_func,
                            reverse,
                            self.node_update)
        h = g.ndata.pop('h')
        return h

class OpNet(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim):
        super(OpNet, self).__init__()
        self.node_dim = node_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim

        self.fmp = MPLayer(node_dim, edge_dim, out_dim//2)
        self.bmp = MPLayer(node_dim, edge_dim, out_dim//2)

    def forward(self,  g):
        hf = self.fmp(g, False)
        hb = self.bmp(g, True)
        return torch.cat([hf, hb], dim=1)

class SoftmaxActor(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim=32,
                 ):
        super(SoftmaxActor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.nn = FNN(input_dim, [hidden_dim], 1)


    def forward(self, x, mask=None):
        x = self.nn(x)
        if mask is None:
            mask = [1 for i in range(x.shape[0])]
        x_masked = x.clone()
        x_masked[mask == 0] = -float('inf')
        return F.softmax(torch.squeeze(x_masked), dim=-1)

class PlacementAgent:
    def __init__(self, node_dim, edge_dim, out_dim,
                 hidden_dim=32,
                 lr=0.03,
                 gamma=0.95):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.lr = lr
        self.gamma = gamma

        self.embedding = OpNet(node_dim, edge_dim, out_dim).to(device)
        self.policy = SoftmaxActor(out_dim, hidden_dim).to(device)
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

    def dev_selection_est(self, program, network, map:list, G_stats, op, options):
        est = {}
        parents = program.op_parents[op]
        end_time = np.array([np.average(G_stats.nodes[p]['end_time']) for p in parents])
        if len(parents) == 0:
            return random.choice(options)
        for dev in options:
            c_time = np.array([communication_latency(program, network, p, op, map[p], dev) for p in parents])
            est[dev] = np.max(c_time + end_time)
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

            returns = torch.tensor(returns).to(device)
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