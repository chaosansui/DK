from astropy.time import Time
from poliastro.maneuver import Maneuver

class OrbitalManeuverInterface:
    def __init__(self, orbit, maneuver_time, delta_v):
        self.orbit = orbit
        self.maneuver_time = maneuver_time
        self.delta_v = delta_v

    def set_delta_v(self, delta_v):
        self.delta_v = delta_v

    def get_delta_v(self):
        return self.delta_v

    def apply_maneuver(self):
        # 创建 Maneuver 对象并应用到轨道
        maneuver = Maneuver(self.delta_v)
        self.orbit = self.orbit.apply_maneuver(maneuver, intermediate=False)

    def check_maneuver_time(self, current_time):
        # 检查当前时间是否已经到达或超过了机动时间
        if current_time >= self.maneuver_time: 
            return True
        return False