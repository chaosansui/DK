import numpy as np
from satellite_env import SatelliteEnv
from agents import Agent, ReplayBuffer

env = SatelliteEnv()
replay_buffer = ReplayBuffer()
agents = [Agent(18, 3) for _ in range(3)]  # 三个智能体，每个有18维状态（3个卫星，每个6维）和3维动作

for episode in range(1000):
    state = env.reset()
    for step in range(100):
        actions = [agent.get_action(state.flatten()) for agent in agents]
        next_state, reward, done, _ = env.step(actions)
        replay_buffer.add((state, actions, reward, next_state, done))
        state = next_state

        if len(replay_buffer.storage) > 64:
            for agent in agents:
                agent.train(replay_buffer)

        if done:
            break
