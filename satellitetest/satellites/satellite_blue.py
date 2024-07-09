from astropy.time import Time, TimeDelta
from astropy import units as u
from poliastro.twobody import Orbit, orbit
from orbits.orbit_redacc import generate_redacc_orbit

def satellite_blue_run(time:Time,orbit:Orbit):
    # 创建轨道对象
    redacc_orbit =orbit

    # 设置初始推进时间间隔（例如，每秒推进一次）
    time_step = 1 * u.second

    # 将时间步长转换为 TimeDelta 对象
    time_step_as_timedelta = TimeDelta(time_step.to(u.s).value, format='sec')

    # 循环推进轨道并输出信息
    current_time = redacc_orbit.epoch
    while True:
        # 推进轨道
        redacc_orbit = redacc_orbit.propagate(current_time + time_step_as_timedelta)

        # 获取位置和速度
        r, v = redacc_orbit.rv()

        # 输出时间、位置和速度
        # print(f"时间：{current_time}")
        # print(f"位置：{r}")
        # print(f"速度：{v}")
        # print("---")

        # 更新时间
        current_time += time_step_as_timedelta
        return current_time,r,v