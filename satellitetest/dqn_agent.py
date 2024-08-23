import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from ddpg_agent import OUNoise,ReplayBuffer


# Q网络
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN智能体
class DQNAgent:
    def __init__(self, state_dim, action_dim,epsilon=1.0,replay_buffer_capacity=1000000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters())
        self.noise = OUNoise(action_dim)

        self.gamma = 0.99
        self.update_rate = 5
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self.epsilon = epsilon
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995

        self.loss = nn.MSELoss()
        self.cnt = 1
    
    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions).view(-1,1)
        rewards = torch.FloatTensor(rewards).view(-1,1)
        dones = torch.FloatTensor(dones).view(-1,1)

        current_q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).max(1)[0].view(-1, 1)
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss(current_q_values,expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        if self.cnt % 10 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.cnt = 1
        else:
            self.cnt += 1

    def get_action(self, state, noise_scale):
        if random.random() < self.epsilon:
            return np.random.uniform(-1, 1, 3) 
            # return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.q_network(state).detach().numpy()
        # action = q_value.max(1)[1].item()
        noise = noise_scale * self.noise.sample()
        action = np.clip(action + noise, -1, 1)
        return action

    def checkpoint_attributes(self):
        # 不保存buffer，太耗内存
        return {
            'q_net' : self.q_network.state_dict(),
            'tar_net' : self.target_network.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        agent_instance = cls(
            checkpoint['state_dim'],
            checkpoint['action_dim']
        )

        agent_instance.q_network.load_state_dict(checkpoint["q_net"])
        agent_instance.target_network.load_state_dict(checkpoint["tar_net"])
        agent_instance.optimizer.load_state_dict(checkpoint["optimizer"])

        return agent_instance

    def save_checkpoint(self, path, filename='checkpoint_dqn.pt'):
        checkpoint = self.checkpoint_attributes()
        torch.save(checkpoint,os.path.join(path,filename))
