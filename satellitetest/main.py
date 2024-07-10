import numpy as np
from astropy import units as u
from astropy.time import Time
from matplotlib import pyplot as plt
from poliastro.bodies import Earth
from astropy.time import Time, TimeDelta
import numpy as np
from satellite_env import SatelliteEnv

from orbits.orbit_red import create_satellite_red_orbit
from interface.orbits_untils import OrbitalElementsInterface
from satellites.satellite_red import satellite_red_run
from satellites.satellite_redacc import satellite_redacc_run
from orbits.orbit_redacc import generate_redacc_orbit
from orbits.orbit_blue import generate_blue_orbit
from satellites.satellite_blue import satellite_blue_run
from interface.orbits_maneuver import OrbitalManeuverInterface
from orbits.orbital_calculations import calculate_updated_orbit
from train import test


def create():
    # 创建一个红色方固定的轨道
    red_orb = create_satellite_red_orbit()
    print(red_orb)

    # 创建字典 区分轨道
    satellite_orbital_elements = {
        "satellite_redacc": OrbitalElementsInterface(),
        "satellite_blue": OrbitalElementsInterface()
    }

    specific_time = Time("2022-01-01T00:00:00", scale="utc")

    # 设置蓝色卫星的轨道参数
    satellite_orbital_elements["satellite_blue"].set_semi_major_axis((300 * u.km + 1000 * u.km) / 2)
    satellite_orbital_elements["satellite_blue"].set_eccentricity(0.01 * u.one)
    satellite_orbital_elements["satellite_blue"].set_inclination(57 * u.deg)
    satellite_orbital_elements["satellite_blue"].set_raan(0 * u.deg)
    satellite_orbital_elements["satellite_blue"].set_arg_periapsis(0 * u.deg)
    satellite_orbital_elements["satellite_blue"].set_true_anomaly(0 * u.deg)
    OrbitalElementsInterface.set_specific_time(specific_time)

    # 使用 generate_orbit 函数创建轨道
    redacc_orbit = generate_redacc_orbit(
        satellite_orbital_elements["satellite_blue"].get_semi_major_axis(),
        satellite_orbital_elements["satellite_blue"].get_eccentricity(),
        satellite_orbital_elements["satellite_blue"].get_inclination(),
        satellite_orbital_elements["satellite_blue"].get_raan(),
        satellite_orbital_elements["satellite_blue"].get_arg_periapsis(),
        satellite_orbital_elements["satellite_blue"].get_true_anomaly(),
        specific_time
    )
    print(redacc_orbit)

    # 设置红色加速卫星的轨道参数
    satellite_orbital_elements["satellite_redacc"].set_semi_major_axis((350 * u.km + 800 * u.km) / 2)
    satellite_orbital_elements["satellite_redacc"].set_eccentricity(0.01 * u.one)
    satellite_orbital_elements["satellite_redacc"].set_inclination(57 * u.deg)
    satellite_orbital_elements["satellite_redacc"].set_raan(0 * u.deg)
    satellite_orbital_elements["satellite_redacc"].set_arg_periapsis(0 * u.deg)
    satellite_orbital_elements["satellite_redacc"].set_true_anomaly(0 * u.deg)
    OrbitalElementsInterface.set_specific_time(specific_time)

    # 使用 generate_orbit 函数创建轨道
    blue_orbit = generate_blue_orbit(
        satellite_orbital_elements["satellite_redacc"].get_semi_major_axis(),
        satellite_orbital_elements["satellite_redacc"].get_eccentricity(),
        satellite_orbital_elements["satellite_redacc"].get_inclination(),
        satellite_orbital_elements["satellite_redacc"].get_raan(),
        satellite_orbital_elements["satellite_redacc"].get_arg_periapsis(),
        satellite_orbital_elements["satellite_redacc"].get_true_anomaly(),
        specific_time
    )
    print(blue_orbit)

    # 变轨机动
    # 设置机动时间和 delta_v
    maneuver_time = specific_time + 30 * u.s  # 30秒后进行机动
    delta_v = (5 * u.s, [0, 0.5, 0] * u.km / u.s)  # 假设第一个元素是时间增量，第二个元素是速度增量

    # 计算更新后的轨道
    updated_orbit = calculate_updated_orbit(blue_orbit, maneuver_time, delta_v)

    return {
        'red':red_orb,
        'redacc':redacc_orbit,
        'blue':blue_orbit
    }



def main():
    create()
    test()
if __name__ == "__main__":
    main()
