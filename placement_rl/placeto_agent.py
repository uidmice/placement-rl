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
                 reverse):
        super(MP, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.reverse = reverse

        self.pre_layers = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim + edge_dim),
            nn.ReLU(),
            nn.Linear(node_dim + edge_dim, node_dim + edge_dim),
            nn.ReLU(),
            nn.Linear(node_dim + edge_dim, node_dim + edge_dim))

        self.update_layers = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim + edge_dim),
            nn.ReLU(),
            nn.Linear(node_dim + edge_dim, node_dim + edge_dim),
            nn.ReLU(),
            nn.Linear(node_dim + edge_dim, out_dim))

        self.apply(weights_init_)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.pre_layers)
        reset(self.update_layers)

    def msg_func(self, edges):
        # pdb.set_trace()
        msg = self.pre_layers(torch.cat([edges.src['x'], edges.data['x']], dim=1))
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
        g.ndata['z'] = torch.randn(g.num_nodes(), self.node_dim + self.edge_dim).to(device)
        dgl.prop_nodes_topo(g, self.msg_func,
                            self.reduce_func,
                            self.reverse,
                            self.node_update)
        h = g.ndata.pop('h')
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

        self.MP_forward = MP(node_dim, edge_dim, out_dim // 2, reverse=True).to(device)
        self.MP_reverse = MP(node_dim, edge_dim, out_dim // 2, reverse=False).to(device)

        self.pred_net_prev = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        ).to(device)

        self.pred_net_later = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        ).to(device)

        self.desc_net_prev = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        ).to(device)

        self.desc_net_later = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        ).to(device)

        self.parallel_net_prev = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        ).to(device)

        self.parallel_net_later = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        ).to(device)



        self.policy = Device_SoftmaxActor(out_dim * 4, n_device, hidden_dim).to(device)
        self.optim = torch.optim.Adam(list(self.MP_forward.parameters()) +
                                      list(self.MP_reverse.parameters()) +
                                      list(self.pred_net_prev.parameters()) +
                                      list(self.pred_net_later.parameters()) +
                                      list(self.desc_net_prev.parameters()) +
                                      list(self.desc_net_later.parameters()) +
                                      list(self.parallel_net_prev.parameters()) +
                                      list(self.parallel_net_later.parameters()) +
                                      list(self.policy.parameters()), lr=lr)
        self.selected_sequence = []
        self.target_device_sequence = []
        self.saved_rewards = []
        self.log_probs = []

        self.available_ops = None

    def reset(self):
        self.available_ops = None

    def op_selection(self, g):
        # The first selection on this op
        if self.available_ops is None:
            self.available_ops = [*range(g.number_of_nodes())]

        selected_op = np.random.choice(self.available_ops)
        self.available_ops.remove(selected_op)
        self.selected_sequence.append(selected_op)

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

        hf = self.MP_forward(g)
        hb = self.MP_reverse(g)
        h = torch.cat([hf, hb], dim=1)

        # pdb.set_trace()

        all_nodes = [*range(program.n_operators)]

        node_embedding = h[op]
        all_nodes.remove(op)

        # Find predecessors
        op_graph = program.P
        predecessors = list(nx.ancestors(op_graph, op))
        all_nodes = [i for i in all_nodes if i not in predecessors]
        pred_embeddings = self.pred_net_prev(h[predecessors])
        pred_embeddings = self.pred_net_later(torch.sum(pred_embeddings, dim = 0))

        # Find descendants
        descendants = list(nx.descendants(op_graph, op))
        all_nodes = [i for i in all_nodes if i not in descendants]
        desc_embeddings = self.desc_net_prev(h[descendants])
        desc_embeddings = self.desc_net_later(torch.sum(desc_embeddings, dim = 0))

        # Parallels
        if len(all_nodes) == 0:
            parallel_embeddings = self.parallel_net_later(torch.zeros(h.shape[1]).to(device))
        else:
            parallel_embeddings = self.parallel_net_prev(h[all_nodes])
            parallel_embeddings = self.parallel_net_later(torch.sum(parallel_embeddings, dim = 0))

        embedding = torch.cat((node_embedding, pred_embeddings, desc_embeddings, parallel_embeddings), dim = 0)
        probs = self.policy(embedding, mask)

        device_distribution = torch.distributions.Categorical(probs = probs)
        target_device = device_distribution.sample()
        self.log_probs.append(device_distribution.log_prob(target_device))
        self.target_device_sequence.append(target_device)

        return target_device.item()


    def finish_episode(self, update_network=True, use_baseline=True):
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
