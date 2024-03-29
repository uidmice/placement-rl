import math

import numpy as np
import random

#
# class ReplayBuffer(object):
#     def __init__(self, state_dim, action_dim, max_size=int(1e6)):
#         self.max_size = max_size
#         self.ptr = 0
#         self.size = 0
#
#         self.state = np.zeros((max_size, state_dim))
#         self.action = np.zeros((max_size, action_dim))
#         self.next_state = np.zeros((max_size, state_dim))
#         self.reward = np.zeros((max_size, 1))
#         self.not_done = np.zeros((max_size, 1))
#
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     def add(self, state, action, next_state, reward, done):
#         self.state[self.ptr] = state
#         self.action[self.ptr] = action
#         self.next_state[self.ptr] = next_state
#         self.reward[self.ptr] = reward
#         self.not_done[self.ptr] = 1. - done
#
#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)
#
#     def sample(self, batch_size):
#         ind = np.random.randint(0, self.size, size=batch_size)
#
#         return (
#             torch.FloatTensor(self.state[ind]).to(self.device),
#             torch.FloatTensor(self.action[ind]).to(self.device),
#             torch.FloatTensor(self.next_state[ind]).to(self.device),
#             torch.FloatTensor(self.reward[ind]).to(self.device),
#             torch.FloatTensor(self.not_done[ind]).to(self.device)
#         )
class Buffer:
    def __init__(self, capacity=10):

        if capacity > 0 and capacity < math.inf:
            self.capacity = capacity
        else:
            self.capacity = math.inf

        self.buffer = []
        self.rank = []
        self.last_index = -1

    def push(self, item, rank, force=False):
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
            self.rank.append(rank)
            self.last_index = len(self.rank) - 1
            return self.last_index
        else:
            lowest = max(self.rank)
            if rank < lowest or force:
                idx = self.rank.index(lowest)
                self.buffer[idx] = item
                self.rank[idx] = rank
                self.last_index = idx
                return idx
            else:
                return -1

    def sample(self):
        if len(self.buffer):
            self.last_index = np.random.randint(len(self.buffer))
            return self.buffer[self.last_index]
        else:
            return None

    def clear(self):
        del self.buffer[:]
        del self.rank[:]
        self.last_index = -1

    def push_to_last(self, item, rank, p):
        if self.last_index < 0:
            return self.push(item, rank)
        if rank < self.rank[self.last_index] or np.random.random() < p:
            self.buffer[self.last_index] = item
            self.rank[self.last_index] = rank
            return self.last_index
        return -1


class ReplayBuffer:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)