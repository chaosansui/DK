import numpy as np
import torch
import matplotlib.pyplot as plt
from agents import Agent, ReplayBuffer
from satellite_env import SatelliteEnv, plot_trajectories
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test():
    # 参数设置
    state_dim = 21
    action_dim = 6  # 2个卫星，每个3维动作（加速度ax, ay, az）
    max_episodes = 200
    max_steps = 100
    batch_size = 64
    replay_buffer_capacity = 1000000

    # 环境和智能体初始化
    env = SatelliteEnv()
    agent_red = Agent(state_dim, action_dim // 2)  # 我方卫星
    agent_blueacc = Agent(state_dim, action_dim // 2)  # 敌方干扰卫星
    rewards_red = []
    rewards_blueacc = []

    # 用于存储轨迹的列表
    all_trajectories_red = []
    all_trajectories_blue = []
    all_trajectories_blueacc = []

    # 用于存储前10个回合的轨迹
    first_10_episodes_red = []
    first_10_episodes_blue = []
    first_10_episodes_blueacc = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward_red = 0
        episode_reward_blueacc = 0

        # 存储每个episode的轨迹
        trajectory_red = []
        trajectory_blue = []
        trajectory_blueacc = []

        epsilon = max(0.1, 1.0 - episode / (max_episodes / 1.5))  # 让epsilon下降得更慢

        for step in range(max_steps):
            action_red = agent_red.get_action(state, noise_scale=epsilon)
            action_blueacc = agent_blueacc.get_action(state, noise_scale=epsilon)
            actions = np.concatenate([action_red, action_blueacc])

            # 与环境交互
            next_state, rewards, done, _ = env.step(actions)
            reward_red, reward_blueacc = rewards
            done = np.any(done)

            # 将经验添加到重放缓冲区中
            agent_red.replay_buffer.add(state, action_red, reward_red, next_state, done)
            agent_blueacc.replay_buffer.add(state, action_blueacc, reward_blueacc, next_state, done)

            # 训练智能体
            agent_red.train(batch_size)
            agent_blueacc.train(batch_size)

            state = next_state
            episode_reward_red += reward_red
            episode_reward_blueacc += reward_blueacc

            # 记录轨迹
            trajectory_red.append(state[:3])
            trajectory_blue.append(state[7:10])
            trajectory_blueacc.append(state[14:17])

            if done:
                break

        logging.info(f"Episode {episode}, Red Reward: {episode_reward_red}, Blueacc Reward: {episode_reward_blueacc}")

        rewards_red.append(episode_reward_red)
        rewards_blueacc.append(episode_reward_blueacc)

        all_trajectories_red.extend(trajectory_red)
        all_trajectories_blue.extend(trajectory_blue)
        all_trajectories_blueacc.extend(trajectory_blueacc)

        # 记录前10个回合的轨迹
        if episode < 10:
            first_10_episodes_red.extend(trajectory_red)
            first_10_episodes_blue.extend(trajectory_blue)
            first_10_episodes_blueacc.extend(trajectory_blueacc)

    # 保存训练好的模型
    torch.save(agent_red.actor.state_dict(), 'actor_red_final.pth')
    torch.save(agent_red.critic.state_dict(), 'critic_red_final.pth')
    torch.save(agent_blueacc.actor.state_dict(), 'actor_blueacc_final.pth')
    torch.save(agent_blueacc.critic.state_dict(), 'critic_blueacc_final.pth')

    # 绘制奖励曲线
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_red, label='Red Agent')
    plt.plot(rewards_blueacc, label='Blueacc Agent')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()
    plt.savefig('image/reward_curve.png')
    plt.show()

    # 生成并保存前10个回合的轨迹图
    plot_trajectories([first_10_episodes_red, first_10_episodes_blue, first_10_episodes_blueacc],
                      filename='image/first_10_episodes_trajectories.png')
    # 生成并保存轨迹图
    plot_trajectories([all_trajectories_red, all_trajectories_blue, all_trajectories_blueacc], filename='image/trajectories.png')
