import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from czml3 import Document, Packet, Preamble
from czml3.properties import Position, Path, Material, SolidColorMaterial, Color
from ddpg_agent import DDPGAgent
from dqn_agent import DQNAgent
from ppo_agent import PPOAgent
from satellite_env import SatelliteEnv, plot_orbit_from_vectors
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Args:
    def __init__(self, max_episodes, max_steps, batch_size, algorithm):
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.algorithm = algorithm

# Function to calculate circular orbit velocity
def calculate_circular_velocity(position):
    mu = 398600  # Earth's gravitational constant, km^3/s^2
    r = np.linalg.norm(position)
    v_circular = np.sqrt(mu / r)
    return v_circular

# Function to save trajectories to CZML format

def test(args):
    state_dim = 21
    action_dim = 6  # 2 satellites, each with 3D action (ax, ay, az)
    max_episodes = args.max_episodes
    max_steps = args.max_steps
    batch_size = args.batch_size
    algorithm = args.algorithm

    env = SatelliteEnv()
    if algorithm == "dqn":
        agent_red = DQNAgent(state_dim, action_dim // 2)
        agent_blueacc = DQNAgent(state_dim, action_dim // 2)
    elif algorithm == "ppo":
        agent_red = PPOAgent(state_dim, action_dim // 2)
        agent_blueacc = PPOAgent(state_dim, action_dim // 2)
    elif algorithm == "ddpg":
        agent_red = DDPGAgent(state_dim, action_dim // 2)
        agent_blueacc = DDPGAgent(state_dim, action_dim // 2)
    else:
        raise RuntimeError(f"Invalid Parameter!")

    rewards_red = []
    rewards_blueacc = []
    all_trajectories = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward_red = 0
        episode_reward_blueacc = 0

        epsilon = max(0.1, 1.0 - episode / (max_episodes / 1.5))

        episode_trajectories = []

        for step in range(max_steps):
            action_red = agent_red.get_action(state, noise_scale=epsilon)
            action_blueacc = agent_blueacc.get_action(state, noise_scale=epsilon)

            actions = np.concatenate([action_red, action_blueacc])

            next_state, rewards, done, _ = env.step(actions)
            reward_red, reward_blueacc = rewards
            done = np.any(done)

            episode_trajectories.append([
                np.array(state[:3]),  # Red Satellite
                np.array(state[7:10]),  # Recon Satellite
                np.array(state[14:17])  # Jam Satellite
            ])

            agent_red.train(batch_size)
            agent_blueacc.train(batch_size)

            state = next_state
            episode_reward_red += reward_red
            episode_reward_blueacc += reward_blueacc

            if done:
                break

        logging.info(f"Episode {episode}, Red Reward: {episode_reward_red}, Blueacc Reward: {episode_reward_blueacc}")

        rewards_red.append(episode_reward_red)
        rewards_blueacc.append(episode_reward_blueacc)

        all_trajectories.append(np.array(episode_trajectories))

    os.makedirs('model', exist_ok=True)
    agent_blueacc.save_checkpoint(path='model', filename="agent_blueacc_final.pt")

    plt.figure(figsize=(12, 6))
    plt.plot(rewards_red, label='Red Agent')
    plt.plot(rewards_blueacc, label='Blueacc Agent')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()
    plt.savefig('plots/reward_curve.png')
    plt.show()