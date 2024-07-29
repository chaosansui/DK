from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Sun, Earth
from poliastro.twobody import Orbit
from satellitetest.interface.orbits_untils import OrbitalElementsInterface


#该轨道倾角为57度 335km*758km所生成的轨道
def create_satellite_blue_orbit():
    a = 546.5 << u.km
    ecc = 0.01 << u.one
    inc = 57 << u.deg
    raan = 0 << u.deg
    argp = 0 << u.deg
    nu = 0 << u.deg
    # 设置全局时间
    specific_time = Time("2022-01-01T00:00:00", scale="utc")
    OrbitalElementsInterface.set_specific_time(specific_time)
    blue_orb = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu,epoch=specific_time)
    return blue_orb