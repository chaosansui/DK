import numpy as np
from astropy import units as u
from astropy.time import Time,TimeDelta
from matplotlib import pyplot as plt
from matplotlib.testing.jpl_units import km
from poliastro.bodies import Earth
from astropy.time import Time, TimeDelta
import numpy as np

from orbits.orbit_blue import create_satellite_blue_orbit
from interface.orbits_untils import OrbitalElementsInterface
from satellites.satellite_blue import satellite_blue_run
from satellites.satellite_red import satellite_red_run
from orbits.orbit_red import generate_red_orbit
from orbits.orbit_blueacc import generate_blueacc_orbit
from satellites.satellite_blueacc import satellite_blueacc_run
from interface.orbits_maneuver import OrbitalManeuverInterface
from orbits.orbital_calculations import calculate_updated_orbit
from specific_time import specific_time

def create():
    # 创建一个红色方固定的轨道
    blue_orb = create_satellite_blue_orbit()


    # 创建轨道要素接口实例
    orbital_elements = OrbitalElementsInterface()

    # 设置轨道要素
    orbital_elements.set_semi_major_axis((200+1600)/2 * u.km)
    orbital_elements.set_eccentricity(0.01 * u.one)
    orbital_elements.set_inclination(57 * u.deg)
    orbital_elements.set_raan(0 * u.deg)
    orbital_elements.set_arg_periapsis(0 * u.deg)
    orbital_elements.set_true_anomaly(0 * u.deg)
    orbital_elements.set_specific_time(specific_time)
    # 创建第二个轨道要素接口实例
    orbital_elements_2 = OrbitalElementsInterface()

    # 设置第二个轨道的要素
    orbital_elements_2.set_semi_major_axis((300+1000)/2 * u.km)  # 更高的LEO轨道
    orbital_elements_2.set_eccentricity(0.02 * u.one)  # 略微增加偏心率
    orbital_elements_2.set_inclination(57 * u.deg)  # 极地轨道
    orbital_elements_2.set_raan(0 * u.deg)
    orbital_elements_2.set_arg_periapsis(0 * u.deg)
    orbital_elements_2.set_true_anomaly(0 * u.deg)
    orbital_elements_2.set_specific_time(specific_time)  # 相同的时间点


    # 生成轨道
    blueacc_orbit = generate_blueacc_orbit(orbital_elements)
    red_orbit= generate_red_orbit(orbital_elements_2)


    # 变轨机动
    # 设置机动时间和 delta_v
    maneuver_time = specific_time + 30 * u.s  # 30秒后进行机动
    delta_v = [0, 3, 0] # 假设第一个元素是时间增量，第二个元素是速度增量

    # 计算更新后的轨道
    #updated_orbit = calculate_updated_orbit(blue_orbit, maneuver_time, delta_v)
    return blue_orb,blueacc_orbit,red_orbit,delta_v
