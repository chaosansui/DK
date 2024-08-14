import logging
import os
from poliastro.plotting import OrbitPlotter
import numpy as np
from poliastro.twobody import Orbit
from astropy.time import Time, TimeDelta
import plotly.graph_objects as go

from satellites.satellite_blueacc import satellite_blueacc_run
from satellites.satellite_red import satellite_red_run
from satellites.satellite_blue import satellite_blue_run
from orbits.orbit_blue import create_satellite_blue_orbit
from orbits.orbit_blueacc import generate_blueacc_orbit
from satellitetest.orbits.orbit_red import generate_red_orbit
from test import create
from specific_time import specific_time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from poliastro.twobody import Orbit
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from plotly.subplots import make_subplots

class SatelliteEnv:
    def __init__(self):
        self.state_dim = 21  # 3个卫星，每个7维（位置3维+速度3维+燃料1维）
        self.action_dim = 6  # 2个卫星（我方和敌方干扰卫星），每个3维动作（加速度ax, ay, az）
        self.max_fuel = 1000.0  # 最大燃料利用率（单位：kg）
        self.red_max_acceleration = np.array([0.2, 0.2, 0.2])  # 我方卫星最大加速度（3维）
        self.blueacc_max_acceleration = np.array([0.58, 0.58, 0.58])  # 敌方干扰卫星最大加速度（3维）
        self.red_fuel_consumption_per_move = np.array([0.0351, 0.0351, 0.0351])
        self.blueacc_fuel_consumption_per_move = np.array([0.102, 0.102, 0.102])
        self.perception_delay_steps = 10  # 感知延迟步长
        self.continuous_maneuver_threshold = 3  # 连续变轨阈值
        self.reset()

    def reset(self):
        # 生成初始轨道
        blue_orb, blueacc_orb, red_orb, delta_v = create()
        time, blue_location, blue_vector = satellite_blue_run(specific_time, blue_orb)
        time, red_location, red_vector = satellite_red_run(specific_time, red_orb)
        time, blueacc_location, blueacc_vector = satellite_blueacc_run(specific_time, blueacc_orb)

        self.state = np.zeros((3, 7))
        self.state[0, :3] = red_location
        self.state[1, :3] = blue_location
        self.state[2, :3] = blueacc_location
        self.state[0, 3:6] = red_vector
        self.state[1, 3:6] = blue_vector
        self.state[2, 3:6] = blueacc_vector
        self.state[0, 6] = self.max_fuel
        self.state[2, 6] = self.max_fuel
        self.time = 0

        self.previous_red_acceleration = np.zeros(3)
        self.previous_blueacc_acceleration = np.zeros(3)
        self.red_continuous_maneuver_steps = 0
        self.blueacc_continuous_maneuver_steps = 0
        self.red_delay_steps = self.perception_delay_steps
        self.blueacc_delay_steps = self.perception_delay_steps

        # 初始化轨迹记录
        self.red_positions = [self.state[0, :3]]
        self.blue_positions = [self.state[1, :3]]
        self.blueacc_positions = [self.state[2, :3]]

        return self.state.flatten()

    def step(self, actions):
        actions = np.array(actions).reshape((2, 3))

        red_acceleration = np.clip(actions[0], -self.red_max_acceleration, self.red_max_acceleration)
        blueacc_acceleration = np.clip(actions[1], -self.blueacc_max_acceleration, self.blueacc_max_acceleration)

        if np.array_equal(red_acceleration, self.previous_red_acceleration):
            self.red_continuous_maneuver_steps = 0
        else:
            self.red_continuous_maneuver_steps += 1
            if self.red_continuous_maneuver_steps >= self.continuous_maneuver_threshold:
                self.red_delay_steps = 30

        if np.array_equal(blueacc_acceleration, self.previous_blueacc_acceleration):
            self.blueacc_continuous_maneuver_steps = 0
        else:
            self.blueacc_continuous_maneuver_steps += 1
            if self.blueacc_continuous_maneuver_steps >= self.continuous_maneuver_threshold:
                self.blueacc_delay_steps = 30

        self.state[0, :3] += self.state[0, 3:6]
        self.state[1, :3] += self.state[1, 3:6]
        self.state[2, :3] += self.state[2, 3:6]

        if self.red_delay_steps == 0:
            self.state[0, 3:6] += red_acceleration
            self.state[0, 6] -= np.sum(self.red_fuel_consumption_per_move * np.abs(red_acceleration))
        else:
            self.red_delay_steps -= 1

        if self.blueacc_delay_steps == 0:
            self.state[2, 3:6] += blueacc_acceleration
            self.state[2, 6] -= np.sum(self.blueacc_fuel_consumption_per_move * np.abs(blueacc_acceleration))
        else:
            self.blueacc_delay_steps -= 1

        self.state[:, 6] = np.clip(self.state[:, 6], 0, self.max_fuel)

        self.previous_red_acceleration = red_acceleration
        self.previous_blueacc_acceleration = blueacc_acceleration

        self.time += 1

        reward_red, reward_blueacc, done = self.calculate_reward()

        self.red_positions.append(self.state[0, :3])
        self.blue_positions.append(self.state[1, :3])
        self.blueacc_positions.append(self.state[2, :3])

        return self.state.flatten(), (reward_red, reward_blueacc), done, {}

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

        reward_red = distance_change_to_recon + distance_change_to_interference
        reward_blueacc = distance_change_to_recon - distance_change_to_interference

        fuel_used_red = self.max_fuel - self.state[0, 6]
        fuel_used_blueacc = self.max_fuel - self.state[2, 6]

        if fuel_used_red < fuel_used_blueacc:
            reward_red += 10
            reward_blueacc -= 10
        else:
            reward_red -= 10
            reward_blueacc += 10

        if self.is_collinear(self.state[0, :3], self.state[1, :3], self.state[2, :3]):
            reward_red -= 1000

        if current_distance_to_recon <= 20:
            reward_red += 4000
            done = True
        elif current_distance_to_interference == current_distance_to_recon / 2:
            reward_red -= 500
            reward_blueacc += 1000
            done = True
        else:
            done = self.time >= 300 or self.state[0, 6] <= 0

        reward_blueacc *= 0.3

        self.previous_distance_to_recon = current_distance_to_recon
        self.previous_distance_to_interference = current_distance_to_interference

        return reward_red, reward_blueacc, done

    def is_collinear(self, p1, p2, p3, tol=1e-6):
        v1 = p2 - p1
        v2 = p3 - p1
        cross_product = np.cross(v1, v2)
        norm_cross_product = np.linalg.norm(cross_product)
        return norm_cross_product < tol


def plot_orbit_from_vectors(trajectories, velocities, labels, colors, filename):
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])

    for trajectory, velocity, label, color in zip(trajectories, velocities, labels, colors):
        try:
            # 确保每个轨迹和速度是三维的
            r = trajectory * u.km
            v = velocity * u.km / u.s

            print(f"Plotting {label}: Position {r}, Velocity {v}")

            # 从矢量创建轨道
            orbit = Orbit.from_vectors(Earth, r, v)

            # 进行绘图
            fig.add_trace(go.Scatter3d(
                x=orbit.sample().x.value,
                y=orbit.sample().y.value,
                z=orbit.sample().z.value,
                mode='lines',
                name=label,
                line=dict(color=color)
            ))

        except Exception as e:
            logging.error(f"Error processing orbit for {label}: {e}")

    fig.update_layout(scene=dict(
        xaxis_title='X (km)',
        yaxis_title='Y (km)',
        zaxis_title='Z (km)'
    ))
    fig.write_html(filename)