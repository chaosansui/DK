from satellite_env import SatelliteEnv
from agents import Agent, ReplayBuffer
import numpy as np
import matplotlib.pyplot as plt
import torch


def test():
    env = SatelliteEnv()
    replay_buffer = ReplayBuffer(capacity=10000, obs_dim=18, state_dim=18, action_dim=9, batch_size=64)

    agent1 = Agent(state_dim=18, action_dim=9)  # 调整智能体的动作维度为9

    episode_rewards = []

    for episode in range(100):
        state = env.reset()
        total_reward = 0
        for step in range(10):
            action = agent1.get_action(state)

            next_state, reward, done, _ = env.step(action)
            replay_buffer.add((state, action, reward, next_state, done))  # 正确地添加数据到 ReplayBuffer
            state = next_state
            total_reward += reward

            if len(replay_buffer.storage) > 64:
                agent1.train(replay_buffer)

            if done:
                break

        episode_rewards.append(total_reward)
        print(f"Episode {episode}, Total Reward: {total_reward}")

    # 保存模型
    torch.save(agent1.actor.state_dict(), "agent1_actor.pth")
    torch.save(agent1.critic.state_dict(), "agent1_critic.pth")

    # 绘制奖励变化图表
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward over Time')
    plt.savefig('training_rewards.png')
    plt.show()