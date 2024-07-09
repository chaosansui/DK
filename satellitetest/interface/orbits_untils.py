from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy.time import Time
class OrbitalElementsInterface:
    specific_time = None  # 全局时间变量

    def __init__(self):
        self.semi_major_axis = None
        self.eccentricity = None
        self.inclination = None
        self.raan = None
        self.arg_periapsis = None
        self.true_anomaly = None
    def set_semi_major_axis(self, value: u.Quantity):
        self.semi_major_axis = value

    def get_semi_major_axis(self) -> u.Quantity:
        return self.semi_major_axis

    def set_eccentricity(self, value: u.Quantity):
        self.eccentricity = value

    def get_eccentricity(self) -> u.Quantity:
        return self.eccentricity

    def set_inclination(self, value: u.Quantity):
        self.inclination = value

    def get_inclination(self) -> u.Quantity:
        return self.inclination

    def set_raan(self, value: u.Quantity):
        self.raan = value

    def get_raan(self) -> u.Quantity:
        return self.raan

    def set_arg_periapsis(self, value: u.Quantity):
        self.arg_periapsis = value

    def get_arg_periapsis(self) -> u.Quantity:
        return self.arg_periapsis

    def set_true_anomaly(self, value: u.Quantity):
        self.true_anomaly = value

    def get_true_anomaly(self) -> u.Quantity:
        return self.true_anomaly

    @classmethod
    def set_specific_time(cls, time: Time):
        cls.specific_time = time

    @classmethod
    def get_specific_time(cls) -> Time:
        return cls.specific_time

    def create_orbit(self) -> Orbit:
        if self.semi_major_axis.value != 0 and self.inclination.value != 0 and OrbitalElementsInterface.specific_time is not None:
            return Orbit.from_classical(Earth, self.semi_major_axis, self.eccentricity, self.inclination, self.raan,
                                        self.arg_periapsis, self.true_anomaly, epoch=OrbitalElementsInterface.specific_time)
        else:
            raise ValueError("Semi-major axis, inclination, and specific time must be set to create the orbit.")

