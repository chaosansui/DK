from satellite_env import SatelliteEnv
from agents import Agent, ReplayBuffer
import numpy as np
from satellite_env import SatelliteEnv

import numpy as np
import matplotlib.pyplot as plt
import torch
def test():


    env = SatelliteEnv()
    replay_buffer = ReplayBuffer()
    agent = Agent(18, 9)  # 一个智能体，输入是18维状态，输出是9维动作（3个卫星，每个3维动作）

    episode_rewards = []

    for episode in range(1000):
        state = env.reset()
        total_reward = 0
        for step in range(100):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(replay_buffer.storage) > 64:
                agent.train(replay_buffer)

            if done:
                break

        episode_rewards.append(total_reward)
        print(f"Episode {episode}, Total Reward: {total_reward}")

    # 保存模型
    torch.save(agent.actor.state_dict(), "actor.pth")
    torch.save(agent.critic.state_dict(), "critic.pth")

    # 绘制奖励变化图表
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward over Time')
    plt.savefig('training_rewards.png')
    plt.show()

