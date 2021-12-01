import torch
import torch.nn as nn
import torch.nn.functional as F

epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# Soft update of target critic network
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) +
                                param.data * tau)


# Hard update of target critic network
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# Q network architecture
class Critic(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs , hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):

        x1 = F.relu(self.linear1(state))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(state))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return torch.min(x1, x2)


class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(Actor, self).__init__()

        self.n_actions = num_actions
        self.l1 = nn.Linear(num_inputs, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state, mask=None):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        if mask is None:
            mask = [1 for i in range(self.n_actions)]
        x_masked = x.clone()
        x_masked[mask == 0] = -float('inf')
        return F.softmax(x_masked, dim=-1)

    def sample(self, state, mask=None):
        probs = self.forward(state, mask)
        dist = torch.distributions.Categorical(probs=probs)
        return dist.sample()