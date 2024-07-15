import numpy as np
from poliastro.twobody import Orbit
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
        self.action_dim = 6  # 2个卫星（我方和敌方干扰卫星），每个3维动作（加速度ax, ay, az）
        self.max_fuel = 100.0  # 最大燃料利用率
        self.max_acceleration = 0.1  # 最大加速度
        self.reset()

    def reset(self):
        # 从外部函数获取卫星轨道、位置、速度等信息
        red_orb, blue_orb, redacc_orb, delta_v = create()
        time, blue_location, blue_vector = satellite_blue_run(specific_time, blue_orb)
        time, red_location, red_vector = satellite_red_run(specific_time, red_orb)
        time, redacc_location, redacc_vector = satellite_redacc_run(specific_time, redacc_orb)

        # 初始化状态
        self.state = np.zeros((3, 7))
        self.state[0, :3] = blue_location  # 我方卫星初始位置
        self.state[1, :3] = red_location  # 敌方侦察卫星初始位置
        self.state[2, :3] = redacc_location  # 敌方干扰卫星初始位置
        self.state[0, 3:6] = blue_vector  # 我方卫星初始速度
        self.state[1, 3:6] = red_vector  # 敌方侦察卫星初始速度
        self.state[2, 3:6] = redacc_vector  # 敌方干扰卫星初始速度
        self.state[0, 6] = self.max_fuel  # 我方卫星燃料初始状态
        self.state[2, 6] = self.max_fuel  # 敌方干扰卫星燃料初始状态
        self.time = 0
        return self.state.flatten()

    def step(self, actions):
        # 动作是一个 (6,) 的数组，我方卫星和敌方干扰卫星的加速度 (ax, ay, az)
        actions = np.array(actions).reshape((2, 3))  # 确保动作数组为 (2, 3)

        # 将动作值裁剪到 [-max_acceleration, max_acceleration] 范围内
        actions = np.clip(actions, -self.max_acceleration, self.max_acceleration)

        # 我方卫星加速度
        blue_acceleration = actions[0]
        # 敌方干扰卫星加速度
        redacc_acceleration = actions[1]

        # 更新位置（使用更精确的物理模型）
        self.state[0, :3] += self.state[0, 3:6] + 0.5 * blue_acceleration  # 我方卫星位置更新
        self.state[1, :3] += self.state[1, 3:6]  # 敌方侦察卫星位置更新
        self.state[2, :3] += self.state[2, 3:6] + 0.5 * redacc_acceleration  # 敌方干扰卫星位置更新

        # 更新速度
        self.state[0, 3:6] += blue_acceleration  # 我方卫星速度更新
        self.state[2, 3:6] += redacc_acceleration  # 敌方干扰卫星速度更新

        # 更新燃料
        self.state[0, 6] -= np.linalg.norm(blue_acceleration)  # 我方卫星燃料消耗
        self.state[2, 6] -= np.linalg.norm(redacc_acceleration)  # 敌方干扰卫星燃料消耗
        self.state[:, 6] = np.clip(self.state[:, 6], 0, self.max_fuel)  # 确保燃料不小于零

        # 增加时间步长
        self.time += 1

        # 计算奖励和是否结束
        reward_blue, reward_redacc, done = self.calculate_reward()

        # 返回新的状态、奖励和是否结束
        return self.state.flatten(), (reward_blue, reward_redacc), done, {}

    def calculate_reward(self):
        distance_to_recon = np.linalg.norm(self.state[0, :3] - self.state[1, :3])
        distance_to_interference = np.linalg.norm(self.state[0, :3] - self.state[2, :3])

        reward_blue = -distance_to_recon + distance_to_interference
        reward_redacc = distance_to_recon - distance_to_interference

        # 额外奖励和惩罚
        if distance_to_recon <= 20:
            reward_blue += 1000  # 蓝方胜利条件
        if distance_to_interference <= 10:
            reward_blue -= 1000  # 蓝方被干扰
            reward_redacc += 1000  # 红方干扰成功

        done = distance_to_recon <= 20 or self.time >= 300 or self.state[0, 6] <= 0
        return reward_blue, reward_redacc, done

    def is_done(self):
        distance_to_recon = np.linalg.norm(self.state[0, :3] - self.state[1, :3])
        if distance_to_recon <= 20:
            return True  # 胜利条件
        if self.time >= 300 or self.state[0, 6] <= 0:
            return True  # 时间限制或燃料用尽
        return False
