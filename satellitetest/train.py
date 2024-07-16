import numpy as np
import torch
import matplotlib.pyplot as plt
from agents import Agent, ReplayBuffer
from satellite_env import SatelliteEnv
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test():
    # 参数设置
    state_dim = 21
    action_dim = 6  # 2个卫星，每个3维动作（加速度ax, ay, az）
    max_episodes = 200
    max_steps = 50
    batch_size = 64
    replay_buffer_capacity = 1000000

    # 环境和智能体初始化
    env = SatelliteEnv()
    # 初始化智能体
    agent_blue = Agent(state_dim, action_dim // 2)
    agent_redacc = Agent(state_dim, action_dim // 2)

    # 用于记录每个episode的总奖励
    rewards_blue = []
    rewards_redacc = []

    # 训练循环
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward_blue = 0
        episode_reward_redacc = 0

        epsilon = max(0.1, 1.0 - episode / max_episodes)

        for step in range(max_steps):
            action_blue = agent_blue.get_action(state, noise_scale=epsilon)
            action_redacc = agent_redacc.get_action(state, noise_scale=epsilon)
            actions = np.concatenate([action_blue, action_redacc])

            # 与环境交互
            next_state, rewards, done, _ = env.step(actions)
            reward_blue, reward_redacc = rewards
            done = np.any(done)

            # 将经验添加到重放缓冲区中
            agent_blue.replay_buffer.add(state, action_blue, reward_blue, next_state, done)
            agent_redacc.replay_buffer.add(state, action_redacc, reward_redacc, next_state, done)

            # 训练智能体
            agent_blue.train(batch_size)
            agent_redacc.train(batch_size)

            state = next_state
            episode_reward_blue += reward_blue
            episode_reward_redacc += reward_redacc

            if done:
                break

        logging.info(f"Episode {episode}, Blue Reward: {episode_reward_blue}, Redacc Reward: {episode_reward_redacc}")

        # 记录每个episode的总奖励
        rewards_blue.append(episode_reward_blue)
        rewards_redacc.append(episode_reward_redacc)

    # 在训练结束后保存模型
    torch.save(agent_blue.actor.state_dict(), 'actor_blue_final.pth')
    torch.save(agent_blue.critic.state_dict(), 'critic_blue_final.pth')
    torch.save(agent_redacc.actor.state_dict(), 'actor_redacc_final.pth')
    torch.save(agent_redacc.critic.state_dict(), 'critic_redacc_final.pth')

    # 绘制回报函数曲线图并保存
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_blue, label='Blue Agent')
    plt.plot(rewards_redacc, label='Redacc Agent')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()
    plt.savefig('reward_curve.png')  # 保存图像
    plt.show()
