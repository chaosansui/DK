import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
from ddpg_agent import OUNoise
import random

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

Transition = namedtuple('Transition', ('state', 'action', 'log_prob', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

class PPOAgent:
    def __init__(self, state_dim, action_dim, buffer_capacity=100000, batch_size=64, gamma=0.99, lr=3e-4, clip_epsilon=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.policy_net(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze(0).detach().numpy(), action_log_prob.item()

    def store_transition(self, transition):
        self.buffer.add(transition)

    def train(self,batch_size):
        self.batch_size = batch_size
        if self.buffer.size() < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, old_log_probs = zip(*transitions)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        old_log_probs = torch.FloatTensor(old_log_probs)

        # Compute advantages
        values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze()
        advantages = rewards + self.gamma * next_values * (1 - dones) - values

        # Update policy network
        mean, std = self.policy_net(states)
        dist = torch.distributions.Normal(mean, std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        # Update value network
        value_loss = (rewards + self.gamma * next_values * (1 - dones) - values).pow(2).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
    
    def checkpoint_attributes(self):
        # 不保存buffer，太耗内存
        return {
            'actor': self.policy_net.state_dict(),
            'critic': self.value_net.state_dict(),
            'actor_optimizer': self.policy_optimizer.state_dict(),
            'critic_optimizer': self.value_optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        agent_instance = cls(
            checkpoint['state_dim'],
            checkpoint['action_dim']
        )

        agent_instance.policy_net.load_state_dict(checkpoint['actor'])
        agent_instance.value_net.load_state_dict(checkpoint['critic'])
        agent_instance.policy_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        agent_instance.value_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        return agent_instance

    def save_checkpoint(self, path, filename='checkpoint_ppo.pt'):
        torch.save(self.checkpoint_attributes(), os.path.join(path,filename))