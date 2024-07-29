from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from satellitetest.interface.orbits_untils import OrbitalElementsInterface


def generate_blueacc_orbit(orbital_elements: OrbitalElementsInterface) -> Orbit:
   # 确保必要的轨道元素已经被设置
   if (orbital_elements.get_semi_major_axis().value == 0 or
           orbital_elements.get_inclination().value == 0 or
           orbital_elements.get_specific_time() is None):
      raise ValueError("Semi-major axis, inclination, and specific time must be set.")

   # 获取轨道元素
   semi_major_axis = orbital_elements.get_semi_major_axis()
   eccentricity = orbital_elements.get_eccentricity()
   inclination = orbital_elements.get_inclination()
   raan = orbital_elements.get_raan()
   arg_periapsis = orbital_elements.get_arg_periapsis()
   true_anomaly = orbital_elements.get_true_anomaly()
   epoch = orbital_elements.get_specific_time()

   # 创建轨道
   blueacc_orbit = Orbit.from_classical(Earth, semi_major_axis, eccentricity, inclination, raan, arg_periapsis,
                                     true_anomaly, epoch=epoch)

   return blueacc_orbit

