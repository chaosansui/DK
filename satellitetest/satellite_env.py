import numpy as np
from poliastro.twobody import Orbit, orbit
from astropy.time import Time, TimeDelta
from satellites.satellite_blue import satellite_blue_run
from satellites.satellite_redacc import satellite_redacc_run
from satellites.satellite_red import satellite_red_run
from orbits.orbit_red import create_satellite_red_orbit
from orbits.orbit_blue import generate_blue_orbit
from satellitetest.orbits.orbit_redacc import generate_redacc_orbit
from test import create
from specific_time import specific_time

class SatelliteEnv:
    def __init__(self):
        self.state_dim = 21  # 3个卫星，每个7维（位置3维+速度3维+燃料1维）
        self.action_dim = 9  # 每个卫星3维动作（加速度ax, ay, az）
        self.max_fuel = 100.0  # 最大燃料利用率
        self.reset()

    def reset(self):
        red_orb,blue_orb,redacc_orb,delta_v=create()
        time,blue_location,blue_vector=satellite_blue_run(specific_time, blue_orb)
        time, red_location, red_vector=satellite_red_run(specific_time,red_orb)
        time, redacc_location, redacc_vector = satellite_red_run(specific_time, red_orb)
        # 初始化状态
        self.state = np.zeros((3, 7))
        self.state[0, :3] = blue_location  # 我方卫星初始位置
        self.state[1, :3] = red_location   # 敌方侦察卫星初始位置
        self.state[2, :3] = redacc_location  # 敌方干扰卫星初始位置
        self.state[:, 6] = self.max_fuel  # 燃料初始状态
        self.time = 0
        return self.state.flatten()

    def step(self, actions):
        red_orb, blue_orb, redacc_orb, delta_v = create()
        step=delta_v
        # 动作是一个 (9,) 的数组，每个卫星的加速度 (ax, ay, az)
        actions = actions.reshape((3, 3))
        self.state[:, :3] += self.state[:, 3:6]  # 更新位置
        self.state[:, 3:6] += actions  # 更新速度
        self.state[:, 6] -= np.linalg.norm(actions, axis=1)  # 消耗燃料
        self.state[:, 6] = np.clip(self.state[:, 6], 0, self.max_fuel)  # 燃料不能为负
        self.time += 1

        reward, done = self.calculate_reward()
        return self.state.flatten(), reward, done, {}

    def calculate_reward(self):
        distance_to_recon = np.linalg.norm(self.state[0, :3] - self.state[1, :3])
        distance_to_interference = np.linalg.norm(self.state[0, :3] - self.state[2, :3])
        reward = 0

        # 奖励是负的到敌方侦察卫星的距离
        reward -= distance_to_recon

        # 惩罚是正的到敌方干扰卫星的距离
        reward += distance_to_interference

        # 额外奖励和惩罚
        if distance_to_recon <= 20:
            reward += 1000  # 胜利条件
        if distance_to_interference <= 20:
            reward -= 1000  # 被干扰

        done = distance_to_recon <= 20 or self.time >= 300 or self.state[0, 6] <= 0
        return reward, done

    def is_done(self):
        distance_to_recon = np.linalg.norm(self.state[0, :3] - self.state[1, :3])
        if distance_to_recon <= 20:
            return True  # 胜利条件
        if self.time >= 300 or self.state[0, 6] <= 0:
            return True  # 时间限制或燃料用尽
        return False
