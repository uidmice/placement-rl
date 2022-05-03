import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from placement_rl.gnn import OpNet, DevNet
from placement_rl.rl_agent import SoftmaxActor
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

        #
        # self.dev_embedding = DevNet(out_op_dim, out_dev_dim).to(device)
        # self.dev_policy = SoftmaxActor(2 * out_dev_dim + 2 * out_op_dim, hidden_dim).to(device)
        # self.dev_network_optim = torch.optim.Adam(list(self.dev_embedding.parameters())+list(self.dev_policy.parameters()), lr=lr)
        # self.dev_log_probs = []

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

    def multi_op_dev_selection(self, g, node_dict):
        placement_embedding = self.embedding(g)
        probs = self.policy(placement_embedding)
        actions = {}
        p = 0
        for node in node_dict:
            prob = probs[list(node_dict[node].values())]
            m = torch.distributions.Categorical(probs=prob/torch.sum(prob))
            a = m.sample()
            d, n_idx = list(node_dict[node].items())[a]
            # n, d = action_dict[a.item()]
            actions[node] = d
            p += m.log_prob(a)
        self.log_probs.append(p)
        return [actions[n] for n in range(len(actions))]

    def dev_selection_est(self, program, network, map:list, G_stats, op, options):
        est = {}
        parents = program.op_parents[op]
        end_time = np.array([np.average(G_stats.nodes[p]['end_time']) for p in parents])
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
