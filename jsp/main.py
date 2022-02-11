import argparse

from jsp.dataLoader import *
from jsp.schedule import get_schedule


def get_args():
    parser = argparse.ArgumentParser()
    # parameters of trains
    parser.add_argument("--TimeSpan", type=int, default=1080, help="total time of railway operating plan")
    # parameters of modeling
    parser.add_argument("--M", type=int, default=1100, help="M used in safety constraints")
    parser.add_argument("--obj_type", type=str, default='get_fsb_sol', help="type of objective functions")
    parser.add_argument("--obj_num", type=int, default=-1, help="number of trains to assign")
    arguments = parser.parse_args()
    return arguments


def main_jsp():
    # 读取数据
    miles, station_list = read_station('jsp/data/1-station.xlsx')
    g, h = read_station_extra('jsp/data/2-station-extra.xlsx')
    sec_times = read_section('jsp/data/3-section-time.xlsx')
    wait_time_lb, wait_time_ub = read_station_stops('jsp/data/4-station-stops.xlsx')
    aa_speed, dd_speed, pp_speed, ap_speed, pa_speed, dp_speed, pd_speed = read_safe_interval('jsp/data/5-safe-intervals.xlsx')
    train_list = read_train('jsp/data/6-lineplan-down.xlsx', station_list, g, h, miles)
    # train_list = read_train('data/7-lineplan-up.xlsx', station_list, g, h)

    # 设置运行参数
    args = get_args()
    selected_train_list = train_list[:]
    model, theta, x_var, d_var, a_var = get_schedule(selected_train_list, station_list, args, sec_times, wait_time_lb,
                           wait_time_ub, aa_speed, dd_speed, pp_speed, ap_speed, pa_speed, dp_speed, pd_speed)

    return model, theta, x_var, d_var, a_var