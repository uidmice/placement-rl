import dgl
from dgl.nn import GATConv

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class MPLayer(nn.Module):
    def __init__(self,
                 node_dim,
                 edge_dim,
                 out_dim):
        super(MPLayer, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim

        self.pre_layer = nn.Linear(node_dim + edge_dim, node_dim + edge_dim)
        self.update_layer = nn.Linear(2 * node_dim + edge_dim, out_dim)

        self.apply(weights_init_)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.pre_layer)
        reset(self.update_layer)

    def msg_func(self, edges):
        msg = torch.cat([edges.src['x'], edges.data['x']], dim=1)
        return {'m': msg}

    def reduce_func(self, nodes):
        z = self.pre_layer(torch.sum((nodes.mailbox['m']), 1))
        return {'z': F.relu(z)}

    def node_update(self, nodes):
        h = torch.cat([nodes.data['x'],
                       nodes.data['z']],
                      dim=1)
        h = self.update_layer(h)
        return {'h': F.relu(h)}

    def forward(self, g,  reverse):
        g.ndata['z'] = torch.randn(g.num_nodes(), self.node_dim + self.edge_dim)
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
        hb = self.fmp(g, True)
        return torch.cat([hf, hb], dim=1)


class DevNet(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_heads=1,
                 activation=None):
        super(DevNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.forward_layer = GATConv (in_dim, out_dim, num_heads=num_heads, activation=activation, allow_zero_in_degree=True)
        self.backward_layer = GATConv (in_dim, out_dim, num_heads=num_heads,activation=activation, allow_zero_in_degree=True)


    def forward(self, graph, feat, op, parallel):
        fh = torch.mean(self.forward_layer(graph, feat), 1)[op]
        bh = torch.mean(self.backward_layer(dgl.reverse(graph), feat), 1)[op]
        para = torch.sum(feat[parallel], 0)
        return torch.cat([fh, bh, feat[op], para])








