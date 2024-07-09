from astropy.time import Time
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver

def calculate_updated_orbit(initial_orbit, maneuver_time, delta_v):
    # 确保 delta_v 是一个带有适当单位的列表或数组
    if not isinstance(delta_v, (list, tuple, u.Quantity)):
        raise ValueError("delta_v must be a list, tuple, or Quantity with appropriate units.")

    # 创建 Maneuver 对象
    maneuver = Maneuver(delta_v)

    # 应用机动到轨道
    updated_orbit = initial_orbit.apply_maneuver(maneuver, intermediate=False)

    return updated_orbit