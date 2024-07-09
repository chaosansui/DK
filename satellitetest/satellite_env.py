import numpy as np


class SatelliteEnv:
    def __init__(self):
        self.state = np.zeros((3, 6))  # 每个卫星有位置(x, y, z)和速度(vx, vy, vz)
        self.time = 0

    def reset(self):
        # 初始化卫星位置和速度
        self.state = np.random.randn(3, 6)
        self.time = 0
        return self.state

    def step(self, actions):
        # actions 是一个 (3, 3) 的数组，每个卫星的加速度 (ax, ay, az)
        self.state[:, :3] += self.state[:, 3:]  # 更新位置
        self.state[:, 3:] += actions  # 更新速度
        self.time += 1
        reward = self.calculate_reward()
        done = self.time >= 100  # 设置一个时间上限
        return self.state, reward, done, {}

    def calculate_reward(self):
        # 计算奖励
        distance_to_recon_satellite = np.linalg.norm(self.state[0, :3] - self.state[1, :3])
        distance_to_interference_satellite = np.linalg.norm(self.state[0, :3] - self.state[2, :3])

        reward = -distance_to_recon_satellite  # 目标是靠近侦察卫星
        if distance_to_interference_satellite < 1.0:  # 如果靠近干扰卫星
            reward -= 10.0  # 给予较大的惩罚

        return reward
