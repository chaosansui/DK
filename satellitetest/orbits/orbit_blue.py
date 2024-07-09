from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

def generate_blue_orbit(semi_major_axis: u.Quantity, eccentricity: u.Quantity, inclination: u.Quantity,
                   raan: u.Quantity, arg_periapsis: u.Quantity, true_anomaly: u.Quantity, epoch: Time) -> Orbit:
    return Orbit.from_classical(Earth, semi_major_axis, eccentricity, inclination, raan, arg_periapsis, true_anomaly, epoch=epoch)


