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
import matplotlib.pyplot as plt

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
        red_orb, blue_orb, redacc_orb, delta_v = create()
        time, blue_location, blue_vector = satellite_blue_run(specific_time, blue_orb)
        time, red_location, red_vector = satellite_red_run(specific_time, red_orb)
        time, redacc_location, redacc_vector = satellite_redacc_run(specific_time, redacc_orb)

        self.state = np.zeros((3, 7))
        self.state[0, :3] = blue_location
        self.state[1, :3] = red_location
        self.state[2, :3] = redacc_location
        self.state[0, 3:6] = blue_vector
        self.state[1, 3:6] = red_vector
        self.state[2, 3:6] = redacc_vector
        self.state[0, 6] = self.max_fuel
        self.state[2, 6] = self.max_fuel
        self.time = 0

        self.previous_blue_acceleration = np.zeros(3)
        self.previous_redacc_acceleration = np.zeros(3)
        self.blue_continuous_maneuver_steps = 0
        self.redacc_continuous_maneuver_steps = 0
        self.blue_delay_steps = self.perception_delay_steps
        self.redacc_delay_steps = self.perception_delay_steps

        self.blue_positions = [self.state[0, :3]]
        self.red_positions = [self.state[1, :3]]
        self.redacc_positions = [self.state[2, :3]]

        return self.state.flatten()

    def step(self, actions):
        actions = np.array(actions).reshape((2, 3))
        actions = np.clip(actions, -self.max_acceleration, self.max_acceleration)

        blue_acceleration = actions[0]
        redacc_acceleration = actions[1]

        if np.array_equal(blue_acceleration, self.previous_blue_acceleration):
            self.blue_continuous_maneuver_steps = 0
        else:
            self.blue_continuous_maneuver_steps += 1
            if self.blue_continuous_maneuver_steps >= self.continuous_maneuver_threshold:
                self.blue_delay_steps = 30

        if np.array_equal(redacc_acceleration, self.previous_redacc_acceleration):
            self.redacc_continuous_maneuver_steps = 0
        else:
            self.redacc_continuous_maneuver_steps += 1
            if self.redacc_continuous_maneuver_steps >= self.continuous_maneuver_threshold:
                self.redacc_delay_steps = 30

        self.state[0, :3] += self.state[0, 3:6]
        self.state[1, :3] += self.state[1, 3:6]
        self.state[2, :3] += self.state[2, 3:6]

        if self.blue_delay_steps == 0:
            self.state[0, 3:6] += blue_acceleration
        else:
            self.blue_delay_steps -= 1

        if self.redacc_delay_steps == 0:
            self.state[2, 3:6] += redacc_acceleration
        else:
            self.redacc_delay_steps -= 1

        self.state[0, 6] -= np.linalg.norm(blue_acceleration)
        self.state[2, 6] -= np.linalg.norm(redacc_acceleration)
        self.state[:, 6] = np.clip(self.state[:, 6], 0, self.max_fuel)

        self.previous_blue_acceleration = blue_acceleration
        self.previous_redacc_acceleration = redacc_acceleration

        self.time += 1

        reward_blue, reward_redacc, done = self.calculate_reward()

        self.blue_positions.append(self.state[0, :3])
        self.red_positions.append(self.state[1, :3])
        self.redacc_positions.append(self.state[2, :3])

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

        reward_blue = distance_change_to_recon + distance_change_to_interference
        reward_redacc = distance_change_to_recon - distance_change_to_interference

        fuel_penalty_blue = self.max_fuel - self.state[0, 6]
        fuel_penalty_redacc = self.max_fuel - self.state[2, 6]
        reward_blue -= fuel_penalty_blue * 0.05
        reward_redacc -= fuel_penalty_redacc * 0.2

        if self.is_collinear(self.state[0, :3], self.state[1, :3], self.state[2, :3]):
            reward_blue -= 1000

        if current_distance_to_recon <= 20:
            reward_blue += 4000
            done = True
        elif current_distance_to_interference == current_distance_to_recon / 2:
            reward_blue -= 500
            reward_redacc += 1000
            done = True
        else:
            done = self.time >= 300 or self.state[0, 6] <= 0

        reward_redacc *= 0.3

        self.previous_distance_to_recon = current_distance_to_recon
        self.previous_distance_to_interference = current_distance_to_interference

        return reward_blue, reward_redacc, done

    def is_collinear(self, p1, p2, p3, tol=1e-6):
        v1 = p2 - p1
        v2 = p3 - p1
        cross_product = np.cross(v1, v2)
        norm_cross_product = np.linalg.norm(cross_product)
        return norm_cross_product < tol


def plot_trajectories(trajectories, filename='trajectories.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['b', 'r', 'g']
    labels = ['Blue Satellite', 'Red Satellite', 'Redacc Satellite']

    for idx, trajectory in enumerate(trajectories):
        positions = np.array(trajectory)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color=colors[idx], label=labels[idx])
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color=colors[idx], marker='o')  # 初始位置标记

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Satellite Trajectories')
    ax.legend()
    plt.savefig(filename)
    plt.show()