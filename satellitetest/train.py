from satellite_env import SatelliteEnv
from agents import Agent, ReplayBuffer
import numpy as np
import matplotlib.pyplot as plt
import torch

def test():
    env = SatelliteEnv()
    replay_buffer = ReplayBuffer(capacity=10000, obs_dim=21, state_dim=21, action_dim=9, batch_size=64)

    agent_our = Agent(state_dim=21, action_dim=3)
    agent_enemy_interference = Agent(state_dim=21, action_dim=3)

    episode_rewards = []

    for episode in range(100):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            state_our = state
            state_enemy_interference = state

            action_our = agent_our.get_action(state_our)
            action_enemy_interference = agent_enemy_interference.get_action(state_enemy_interference)

            actions = np.concatenate([action_our, np.zeros(3), action_enemy_interference])

            next_state, reward, done, _ = env.step(actions)

            reward_our = reward
            reward_enemy_interference = -reward

            replay_buffer.add((state_our, action_our, reward_our, next_state, done))
            replay_buffer.add((state_enemy_interference, action_enemy_interference, reward_enemy_interference, next_state, done))

            state = next_state
            total_reward += reward_our

            if len(replay_buffer.storage) > 64:
                agent_our.train(replay_buffer)
                agent_enemy_interference.train(replay_buffer)

        episode_rewards.append(total_reward)
        print(f"Episode {episode}, Total Reward: {total_reward}")

    # 保存模型
    torch.save(agent_our.actor.state_dict(), "agent_our_actor.pth")
    torch.save(agent_our.critic.state_dict(), "agent_our_critic.pth")
    torch.save(agent_enemy_interference.actor.state_dict(), "agent_enemy_interference_actor.pth")
    torch.save(agent_enemy_interference.critic.state_dict(), "agent_enemy_interference_critic.pth")

    # 绘制奖励变化图表
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward over Time')
    plt.savefig('training_rewards.png')
    plt.show()
