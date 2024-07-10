import numpy as np

class SatelliteEnv:
    def __init__(self):
        self.state_dim = 18  # 3个卫星，每个6维
        self.action_dim = 9  # 每个卫星3维动作
        self.reset()

    def reset(self):
        # 初始化卫星位置和速度
        self.state = np.zeros((3, 6))
        self.state[0, :3] = np.array([1000, 0, 0])  # 我方卫星初始位置
        self.state[1, :3] = np.array([0, 0, 0])    # 敌方侦察卫星初始位置
        self.state[2, :3] = np.array([900, 0, 0])  # 敌方防卫卫星初始位置
        self.time = 0
        return self.state.flatten()

    def step(self, actions):
        # actions 是一个 (9,) 的数组，每个卫星的加速度 (ax, ay, az)
        actions = actions.reshape((3, 3))
        self.state[:, :3] += self.state[:, 3:]  # 更新位置
        self.state[:, 3:] += actions  # 更新速度
        self.time += 1

        reward = self.calculate_reward()
        done = self.is_done()

        return self.state.flatten(), reward, done, {}

    def calculate_reward(self):
        distance_to_recon_satellite = np.linalg.norm(self.state[0, :3] - self.state[1, :3])
        distance_to_interference_satellite = np.linalg.norm(self.state[0, :3] - self.state[2, :3])

        if distance_to_recon_satellite <= 20:
            reward = 1000  # 胜利条件
        elif distance_to_recon_satellite <= 100:
            reward = -10 * distance_to_interference_satellite  # 中距离阶段，奖励基于与干扰卫星的距离
        else:
            reward = -distance_to_recon_satellite  # 远程阶段，奖励基于与侦察卫星的距离

        if distance_to_interference_satellite < 10:
            reward -= 100  # 避免过于接近干扰卫星

        return reward

    def is_done(self):
        distance_to_recon_satellite = np.linalg.norm(self.state[0, :3] - self.state[1, :3])
        if distance_to_recon_satellite <= 20:
            return True  # 胜利条件
        if self.time >= 300:
            return True  # 时间限制
        return False
