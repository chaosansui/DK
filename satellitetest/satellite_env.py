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
        self.perception_delay_steps = 10  # 感知延迟步长
        self.continuous_maneuver_threshold = 3  # 连续变轨阈值
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

        self.previous_blue_acceleration = np.zeros(3)
        self.previous_redacc_acceleration = np.zeros(3)
        self.blue_continuous_maneuver_steps = 0
        self.redacc_continuous_maneuver_steps = 0
        self.blue_delay_steps = self.perception_delay_steps
        self.redacc_delay_steps = self.perception_delay_steps

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

        # 检查我方卫星的连续变轨情况
        if np.array_equal(blue_acceleration, self.previous_blue_acceleration):
            self.blue_continuous_maneuver_steps = 0
        else:
            self.blue_continuous_maneuver_steps += 1
            if self.blue_continuous_maneuver_steps >= self.continuous_maneuver_threshold:
                self.blue_delay_steps = 30

        # 检查敌方干扰卫星的连续变轨情况
        if np.array_equal(redacc_acceleration, self.previous_redacc_acceleration):
            self.redacc_continuous_maneuver_steps = 0
        else:
            self.redacc_continuous_maneuver_steps += 1
            if self.redacc_continuous_maneuver_steps >= self.continuous_maneuver_threshold:
                self.redacc_delay_steps = 30

        # 更新位置和速度
        self.state[0, :3] += self.state[0, 3:6]  # 我方卫星位置更新
        self.state[1, :3] += self.state[1, 3:6]  # 敌方侦察卫星位置更新
        self.state[2, :3] += self.state[2, 3:6]  # 敌方干扰卫星位置更新

        # 使用延迟更新位置和速度
        if self.blue_delay_steps == 0:
            self.state[0, 3:6] += blue_acceleration
        else:
            self.blue_delay_steps -= 1

        if self.redacc_delay_steps == 0:
            self.state[2, 3:6] += redacc_acceleration
        else:
            self.redacc_delay_steps -= 1

        # 更新燃料
        self.state[0, 6] -= np.linalg.norm(blue_acceleration)  # 我方卫星燃料消耗
        self.state[2, 6] -= np.linalg.norm(redacc_acceleration)  # 敌方干扰卫星燃料消耗
        self.state[:, 6] = np.clip(self.state[:, 6], 0, self.max_fuel)  # 确保燃料不小于零

        # 记录当前加速度
        self.previous_blue_acceleration = blue_acceleration
        self.previous_redacc_acceleration = redacc_acceleration

        # 增加时间步长
        self.time += 1

        # 计算奖励和是否结束
        reward_blue, reward_redacc, done = self.calculate_reward()

        # 返回新的状态、奖励和是否结束
        return self.state.flatten(), (reward_blue, reward_redacc), done, {}

    def calculate_reward(self):
        current_distance_to_recon = np.linalg.norm(self.state[0, :3] - self.state[1, :3])
        current_distance_to_interference = np.linalg.norm(self.state[0, :3] - self.state[2, :3])

        if hasattr(self, 'previous_distance_to_recon') and hasattr(self, 'previous_distance_to_interference'):
            previous_distance_to_recon = self.previous_distance_to_recon
            previous_distance_to_interference = self.previous_distance_to_interference
        else:
            previous_distance_to_recon = current_distance_to_recon
            previous_distance_to_interference = current_distance_to_interference

        distance_change_to_recon = current_distance_to_recon - previous_distance_to_recon
        distance_change_to_interference = current_distance_to_interference - previous_distance_to_interference

        # 调整后的奖励函数逻辑
        reward_blue = distance_change_to_recon + distance_change_to_interference
        reward_redacc = distance_change_to_recon - distance_change_to_interference

        fuel_penalty_blue = self.max_fuel - self.state[0, 6]
        fuel_penalty_redacc = self.max_fuel - self.state[2, 6]
        reward_blue -= fuel_penalty_blue * 0.05  # 减少蓝方燃料惩罚
        reward_redacc -= fuel_penalty_redacc * 0.2  # 增加红方燃料惩罚

        # 判断是否形成三点一线
        if self.is_collinear(self.state[0, :3], self.state[1, :3], self.state[2, :3]):
            reward_blue -= 1000  # 蓝方形成三点一线的惩罚

        if current_distance_to_recon <= 20:
            reward_blue += 4000  # 增加蓝方达到目标的奖励
            done = True
        elif current_distance_to_interference == current_distance_to_recon / 2:
            reward_blue -= 500  # 减少蓝方被干扰的惩罚
            reward_redacc += 1000
            done = True
        else:
            done = self.time >= 300 or self.state[0, 6] <= 0

        reward_redacc *= 0.3  # 进一步降低红方的奖励比例

        self.previous_distance_to_recon = current_distance_to_recon
        self.previous_distance_to_interference = current_distance_to_interference

        return reward_blue, reward_redacc, done

    def is_collinear(self, p1, p2, p3, tol=1e-6):
        # 计算向量
        v1 = p2 - p1
        v2 = p3 - p1
        # 计算向量的叉积
        cross_product = np.cross(v1, v2)
        # 叉积的模
        norm_cross_product = np.linalg.norm(cross_product)
        # 如果叉积的模接近于零，则说明共线
        return norm_cross_product < tol
