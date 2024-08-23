import numpy as np
import torch
import matplotlib.pyplot as plt
from ddpg_agent import DDPGAgent
from dqn_agent import DQNAgent
from ppo_agent import PPOAgent
from satellite_env import SatelliteEnv, plot_orbit_from_vectors
import logging
import os

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def test(args):
    # 参数设置
    state_dim = 21
    action_dim = 6  # 2个卫星，每个3维动作（加速度ax, ay, az）
    max_episodes = args.max_episodes
    max_steps = args.max_steps
    batch_size = args.batch_size
    algorithm = args.algorithm
    # 环境和智能体初始化
    env = SatelliteEnv()
    if algorithm == "dqn":
        agent_red = DQNAgent(state_dim, action_dim // 2)  # 我方卫星
        agent_blueacc = DQNAgent(state_dim, action_dim // 2)  # 敌方干扰卫星
    elif algorithm == "ppo":
        agent_red = PPOAgent(state_dim, action_dim // 2)  # 我方卫星
        agent_blueacc = PPOAgent(state_dim, action_dim // 2)  # 敌方干扰卫星
    elif algorithm == "ddfg":
        agent_red = DDPGAgent(state_dim, action_dim // 2)  # 我方卫星
        agent_blueacc = DDPGAgent(state_dim, action_dim // 2)  # 敌方干扰卫星
    else:
        raise RuntimeError(f"Invalid Parameter!")

    rewards_red = []
    rewards_blueacc = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward_red = 0
        episode_reward_blueacc = 0

        epsilon = max(0.1, 1.0 - episode / (max_episodes / 1.5))  # 让epsilon下降得更慢

        for step in range(max_steps):
            if algorithm == "ppo":
                action_red, probs_red = agent_red.get_action(state)
                action_blueacc, probs_blueacc = agent_blueacc.get_action(state)
            elif algorithm == "dqn":
                action_red = agent_red.get_action(state, noise_scale=epsilon).reshape(1,3)
                action_blueacc = agent_blueacc.get_action(state, noise_scale=epsilon).reshape(1,3)
            elif algorithm == "ddpg":
                action_red = agent_red.get_action(state, noise_scale=epsilon)
                action_blueacc = agent_blueacc.get_action(state, noise_scale=epsilon)
            else:
                raise RuntimeError(f"not support algorithm : {algorithm}")
            actions = np.concatenate([action_red, action_blueacc])

            # 与环境交互
            next_state, rewards, done, _ = env.step(actions)
            reward_red, reward_blueacc = rewards
            done = np.any(done)

            # 将经验添加到重放缓冲区中
            if algorithm == "dqn":
                action_red_to_buffer = np.argmax(action_red)
                action_blueacc_to_buffer = np.argmax(action_blueacc)
                agent_red.replay_buffer.add(state, action_red_to_buffer, reward_red, next_state, done)
                agent_blueacc.replay_buffer.add(state, action_blueacc_to_buffer, reward_blueacc, next_state, done)
            elif algorithm == "ppo":
                agent_blueacc.store_transition((state, action_blueacc, reward_blueacc, next_state, done, probs_blueacc))
                agent_red.store_transition((state, action_red, reward_red, next_state, done, probs_red))
            elif algorithm == "ddpg":
                action_red_to_buffer = action_red
                action_blueacc_to_buffer = action_blueacc
                agent_red.replay_buffer.add(state, action_red_to_buffer, reward_red, next_state, done)
                agent_blueacc.replay_buffer.add(state, action_blueacc_to_buffer, reward_blueacc, next_state, done)

            # 训练智能体
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

        # 使用最后一个步长的状态计算轨道
        final_trajectories = [
            np.array(state[:3]),     # Red Satellite
            np.array(state[7:10]),   # Recon Satellite
            np.array(state[14:17])   # Jam Satellite
        ]

        final_velocities = [
            np.array(state[3:6]),    # Red Satellite Velocity
            np.array(state[10:13]),  # Recon Satellite Velocity
            np.array(state[17:20])   # Jam Satellite Velocity
        ]

        # 确保每个向量都是一维的
        final_trajectories = [np.atleast_1d(traj) for traj in final_trajectories]
        final_velocities = [np.atleast_1d(vel)   for vel in final_velocities]

        if (episode+1) % 10 == 0:
            # 绘制每个回合结束时的轨道
            plot_orbit_from_vectors(
                trajectories=final_trajectories,
                velocities=final_velocities,
                labels=['Red Satellite', 'Recon Satellite', 'Jam Satellite'],
                colors=['red', 'green', 'blue'],
                filename=f'plots/orbits_episode_{episode}.html'  # 保存每个回合的轨道到不同的文件
            )

    # 保存训练好的模型
    os.makedirs('model', exist_ok=True)
    agent_blueacc.save_checkpoint(path = 'model', filename = "agent_blueacc_final.pt")
    # agent_red.save_checkpoint(path = 'model', filename = "agent_red_final.pt")
    # torch.save(agent_red.actor.state_dict(), 'model/actor_red_final.pth')
    # torch.save(agent_red.critic.state_dict(), 'model/critic_red_final.pth')
    # torch.save(agent_blueacc.actor.state_dict(), 'model/actor_blueacc_final.pth')
    # torch.save(agent_blueacc.critic.state_dict(), 'model/critic_blueacc_final.pth')

    # 绘制奖励曲线
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_red, label='Red Agent')
    plt.plot(rewards_blueacc, label='Blueacc Agent')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()
    plt.savefig('plots/reward_curve.png')
    plt.show()
