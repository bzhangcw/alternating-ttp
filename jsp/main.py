import argparse
from gurobipy import *

from jsp.dataLoader import *
from jsp.model import set_obj_func, add_time_var_and_cons, add_safe_cons
from jsp.plot import *
from jsp.util import *


def get_args():
    parser = argparse.ArgumentParser()
    # parameters of trains
    parser.add_argument("--TimeSpan", type=int, default=1080, help="total time of railway operating plan")
    # parameters of modeling
    parser.add_argument("--M", type=int, default=1100, help="M used in safety constraints")
    parser.add_argument("--force_feasibility", type=int, default=1, help="type of objective functions")
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
    train_list = read_train('jsp/data/6-lineplan-down.xlsx', station_list, g, h)
    # train_list = read_train('data/7-lineplan-up.xlsx', station_list, g, h)

    # 设置运行参数
    args = get_args()
    train_begin = 0
    train_num = 100 # 排标杆车与非标杆车
    # train_num = len([ trn for trn in train_list[train_begin:] if trn.standard == 1 ]) # 只排标杆车
    # selected_train_list = train_list[train_begin : train_begin + train_num] # 排前面的车
    train_split = 50
    # selected_train_list = train_list[:train_split] + train_list[train_split-train_num:] # 前面的标杆车加长程车加后面的短程车
    selected_train_list = train_list[:]  # 排全部的车
    # train_num = len(train_list)

    # greedy地画出标杆车
    # assigned_train_list, D_high, A_high = assign_high_level_train(selected_train_list, sec_times, wait_time_lb,
    #                                                               wait_time_ub, args.TimeSpan)
    # plot_high_level_train(assigned_train_list, station_list, miles, D_high, A_high,
    #                       'output/high_level_train_{}.svg'.format(int(train_list[0].up)), args.TimeSpan)

    # 数据预处理：得到所有车站的停站车辆和通过车辆
    stop_trains, pass_trains = get_stop_pass_trains(station_list, selected_train_list)

    # 建立模型与设置参数
    model = Model('TTP_JSP')
    model.setParam('OutputFlag', 1)
    # 增加各占出发时间变量、到达时间变量及其约束
    model, A_var, D_var, W_var, T_var = add_time_var_and_cons(model, selected_train_list, sec_times, wait_time_lb,
                                                              wait_time_ub, args.TimeSpan, delta=2)
    # 定义目标函数
    model, x_var = set_obj_func(model, selected_train_list, W_var, T_var, args.force_feasibility, args.obj_num)
    # 增加同一站台任意一对车辆的安全间隔约束，以及防止站间越行的约束
    model, theta_aa, theta_ap, theta_pa, theta_pp, theta_dd, theta_dp, theta_pd = \
        add_safe_cons(model, x_var, station_list, stop_trains, pass_trains, A_var, D_var, aa_speed, dd_speed, pp_speed, ap_speed, pa_speed, dp_speed, pd_speed, args.M)
    model.setParam('MIPFocus', 1)  # 专注于可行解

    return model, theta_aa, theta_ap, theta_pa, theta_pp, theta_dd, theta_dp, theta_pd, x_var, \
           D_var, A_var
    # 模型求解
    # model.optimize()
    #
    # # model.computeIIS()
    # # model.write('ttp.ilp')
    #
    # # 输出模型求解结果
    # if model.status != GRB.INFEASIBLE:
    #     print('\n Total running time: {:.4f}'.format(model.runtime))
    #
    #     if model.status != GRB.INTERRUPTED:
    #         model.write("output/ttp_jsp_{0}_{1}.sol".format(len(selected_train_list), int(model.runtime)))
    #
    #     # 画铁路运行时空网络图
    #     plot_fig(selected_train_list, station_list, miles, D_var, A_var,
    #              'output/timetable_{0}_{1}.svg'.format(len(selected_train_list), int(model.runtime)), args.TimeSpan)
