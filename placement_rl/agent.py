import copy
import os

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from placement_rl.rl_models import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PlacementAgent:
    def __init__(self, state_dim, action_dim,
                 hidden_dim=256,
                 lr=3e-4,
                 gamma=0.99,
                 critic_update_interval=1,
                 actor_update_interval=1):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.critic = Critic(self.state_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.actor = Actor(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.gamma = gamma
        self.critic_update_interval = critic_update_interval
        self.actor_update_interval = actor_update_interval
        self.updates = 0

    # def update_parameters(self, replay_buffer, batch_size=256):
    #     self.updates += 1
    #
    #     state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
    #
    #     with torch.no_grad():
    #         target_Q1, target_Q2 = self.critic_target(next_state)
    #         target_Q = torch.min(target_Q1, target_Q2)
    #         target_Q = reward + not_done * self.gamma * target_Q
    #
    #     current_Q1, current_Q2 = self.critic(state, action)  # Two Q-functions to mitigate positive bias in the policy improvement step
    #
    #     advantage1 = target_Q - current_Q1
    #     advantage2 = target_Q - current_Q2
    #     critic_loss = advantage1.pow(2).mean() + advantage2.pow(2).mean()
    #     self.critic_optimizer.zero_grad()
    #     critic_loss.backward()
    #     self.critic_optimizer.step()
    #
    #     pi, log_pi, _ = self.policy.sample(state_batch)
    #
    #     qf1_pi, qf2_pi = self.critic(state_batch, pi)
    #     min_qf_pi = torch.min(qf1_pi, qf2_pi)
    #
    #     sqf1_pi, sqf2_pi = self.safety_critic(state_batch, pi)
    #     max_sqf_pi = torch.max(sqf1_pi, sqf2_pi)
    #
    #
    #
    #     self.policy_optim.zero_grad()
    #     policy_loss.backward()
    #     self.policy_optim.step()

    def train(self, env, num_episodes):
        episode_rewards = []
        reward_trace = []
        env.reset()
        ops = np.array(list(nx.topological_sort(env.program.P)))
        mask = torch.zeros(self.action_dim).to(device)

        last_latency = env.latency

        for i in range(num_episodes):
            rewards = []
            total_reward = 0
            actions = []
            print(f'====== EPISODE {i} =======')
            for j in range(env.n_operators):
                n = ops[j]
                s = env.get_state(n).to(device)
                action_set = env.program.placement_constraints[n]
                mask[:] = 0
                mask[action_set] = 1
                probs = self.actor(s, mask=mask)
                dist = torch.distributions.Categorical(probs=probs)

                action = dist.sample()
                # print(f'action: device {action}')
                actions.append(action.item())

                latency = env.step(n, action.item())
                reward = - latency/500
                # print(f'reward: {reward}')
                last_latency = latency

                advantage = reward - self.critic(s)
                if j < env.n_operators - 1:
                    ns = env.get_state(ops[j+1]).to(device)
                    advantage += self.gamma * self.critic(ns)


                total_reward += reward
                rewards.append(reward)

                critic_loss = advantage.pow(2).mean()
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                actor_loss = -dist.log_prob(action) * advantage.detach()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
            episode_rewards.append(total_reward)
            reward_trace.append(rewards)
            # print(ops.tolist())
            # print(actions)

        return episode_rewards, reward_trace

    # def save(self, path='/content/gdrive/MyDrive/Placement/checkpoints', exp_name='test', suffix='', info=None):
    #     import datetime
    #     name = f'{exp_name}'
    #     if suffix:
    #         name += f'_{suffix}'
    #     p = os.path.join(path, name)
    #
    #     if not os.path.exists(p):
    #         os.makedirs(p)
    #     print('LOGDIR: ', p)
    #
    #     self.checkpoint += 1
    #
    #     torch.save({
    #         'info': info,
    #         'actor_state_dict': self.actor.state_dict(),
    #         'adam_actor': self.adam_actor.state_dict(),
    #         'critic_state_dict': self.critic.state_dict(),
    #         'adam_critic': self.adam_critic.state_dict()
    #     }, os.path.join(p, f'{exp_name}_{self.checkpoint}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pt'))
