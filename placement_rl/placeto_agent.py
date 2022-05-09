import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import networkx as nx

from placement_rl.primative_nn import *
# from placement_rl.rl_agent import SoftmaxActor
# from env.latency import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epsilon = 1e-6
torch.autograd.set_detect_anomaly(True)


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
        self.nn = FNN(input_dim, [hidden_dim], n_device)


    def forward(self, x, mask=None):
        # pdb.set_trace()
        x = self.nn(x)
        if mask is None:
            mask = [1 for i in range(x.shape[0])]
        x_masked = x.clone()
        x_masked[mask == 0] = -float('inf')
        return F.softmax(torch.squeeze(x_masked), dim=-1)


class Aggregator(nn.Module):
    def __init__(self,
                 node_dim,
                 emb_dim,
                 out_dim,
                 reverse):
        super(Aggregator, self).__init__()

        self.reverse = reverse

        self.pre_layers = FNN(node_dim, [node_dim], emb_dim)
        self.update_layers = FNN(emb_dim, [emb_dim], out_dim)


    def msg_func(self, edges):
        msg = F.relu(self.pre_layers(edges.src['y']))
        return {'m': msg}


    def node_update(self, nodes):
        h = F.relu(self.update_layers(nodes.data['z']))
        return {'h': h}

    def forward(self, g):
        if self.reverse:
            g = dgl.reverse(g)
        g.update_all(self.msg_func, fn.sum('m', 'z'), self.node_update)
        h = g.ndata['h']
        return h


class MP(nn.Module):
    def __init__(self, emb_dim,hidden_dim,  k):
        super(MP, self).__init__()

        self.emb_dim = emb_dim
        self.k = k

        self.fpa = Aggregator(emb_dim, hidden_dim, emb_dim, False)
        self.bpa = Aggregator(emb_dim, hidden_dim, emb_dim, True)

        self.node_transform = FNN(emb_dim, [hidden_dim], emb_dim)

    def forward(self, g):
        self_trans = self.node_transform(g.ndata['x'])
        def message_pass(agg, sink):
            g.ndata['y'] = self_trans
            for i in range(self.k):
                h = agg(g)
                h[sink, :] = 0
                h += self_trans
                g.ndata['y'] = h
            return g.ndata['y']

        out_fpa = message_pass(self.fpa, g.num_of_nodes()-1)
        out_bpa = message_pass(self.bpa, 0)
        return torch.cat([out_fpa, out_bpa], dim=1)

class PlaceToEmbedding(nn.Module):
    def __init__(self, emb_size, hidden_dim, k):
        super(PlaceToEmbedding, self).__init__()
        self.mp = MP(emb_size, hidden_dim, k)
        self.agg_p = Aggregator(emb_size * 2, emb_size * 2, emb_size * 2, False)
        self.agg_c = Aggregator(emb_size * 2, emb_size * 2, emb_size * 2, True)
        self.agg_r = Aggregator(emb_size * 2, emb_size * 2, emb_size * 2, False)

    def forward(self, g):
        return self.mp(g)


class PlaceToAgent:
    def __init__(self,
                 node_dim,
                 out_dim,
                 n_device,
                 k=3,
                 hidden_dim=32,
                 lr=0.03,
                 gamma=0.95):

        self.node_dim = node_dim
        self.out_dim = out_dim
        self.n_device = n_device
        self.hidden_dim = hidden_dim
        self.k = k

        self.lr = lr
        self.gamma = gamma

        self.policy = Device_SoftmaxActor(node_dim * 8, n_device, hidden_dim).to(device)
        self.embedding = PlaceToEmbedding(node_dim, hidden_dim, k)
        self.optim = torch.optim.Adam(list(self.embedding.parameters()) +
                                      list(self.policy.parameters()), lr=lr)

        self.saved_rewards = []
        self.log_probs = []

    def dev_selection(self,
                      g,
                      program,
                      op,
                      mask):

        emb = self.embedding(g)
        g.ndata['y'] = emb


        p_g = dgl.node_subgraph(g, [op] + program.op_parents[op])
        pred_embeddings = self.embedding.agg_p(p_g)[0, :]

        # Find descendants
        c_g = dgl.node_subgraph(g, [op] + program.op_children[op])
        desc_embeddings = self.embedding.agg_c(c_g)[0, :]

        # Parallels
        r = program.op_parallel[op]
        r_g = dgl.graph(([0] * len(r), range(len(r))))
        r_g.ndata['y'] = emb[[0]+r, :]
        parallel_embeddings = self.embedding.agg_r(r_g)[0, :]

        node_embedding = emb[op, :]

        embedding = torch.cat((pred_embeddings, desc_embeddings, parallel_embeddings, node_embedding), dim = -1)


        probs = self.policy(embedding, mask)

        device_distribution = torch.distributions.Categorical(probs = probs)
        target_device = device_distribution.sample()
        self.log_probs.append(device_distribution.log_prob(target_device))

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

        del self.saved_rewards[:]
        del self.log_probs[:]
