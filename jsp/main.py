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
    parser.add_argument("--obj_type", type=str, default='get_fsb_sol', help="type of objective functions")
    parser.add_argument("--obj_num", type=int, default=-1, help="number of trains to assign")
    parser.add_argument("--trn_tbl_type", type=str, default="order", help="types of variables fixed by given timetable")
    parser.add_argument("--sol_limit", type=int, default=1, help="limit of solution numbers")
    parser.add_argument("--time_limit", type=int, default=3600, help="limit of time(second)")
    parser.add_argument("--direction", type=str, default="down", help="direction of trains")
    arguments = parser.parse_args()
    return arguments


# 站名 up=1
station_name_list = ['北京南', '廊坊', '京津线路所', '津沪线路所', '天津南',
                     '沧州西', '德州东', '济南西', '崔马庄线路所', '泰安',
                     '曲阜东', '滕州东', '枣庄', '徐州东', '宿州东',
                     '蚌埠南', '定远', '滁州', '扬州线路所', '南京南',
                     '秦淮河线路所', '镇江南', '丹阳北', '常州北', '无锡东',
                     '苏州北', '昆山南', '黄渡线路所', '上海虹桥']


def main_jsp(params_sys):
    # 读取数据
    miles, station_list = read_station('jsp/data/1-station.xlsx')
    g, h = read_station_extra('jsp/data/2-station-extra.xlsx')
    sec_times = read_section('jsp/data/3-section-time.xlsx')
    wait_time_lb, wait_time_ub = read_station_stops('jsp/data/4-station-stops.xlsx')
    aa_speed, dd_speed, pp_speed, ap_speed, pa_speed, dp_speed, pd_speed = read_safe_interval(
        'jsp/data/5-safe-intervals.xlsx')
    if params_sys.up:
        train_list = read_train('jsp/data/7-lineplan-up.xlsx', station_list, g, h, miles)
    else:
        train_list = read_train('jsp/data/6-lineplan-down.xlsx', station_list, g, h, miles)

    # 设置运行参数
    args = get_args()
    args.trn_tbl_type = ''  # 'min_run_time'
    args.obj_type = 'max_trn_num'
    args.TimeSpan = params_sys.time_span

    selected_train_list = train_list[:params_sys.train_size]  # 排全部的车

    dep_trains, arr_trains, pass_trains = get_dep_arr_pass_trains(station_list, selected_train_list)

    # 建立模型与设置参数
    model = Model('TTP_JSP')
    model.setParam('OutputFlag', 1)
    # 增加各占出发时间变量、到达时间变量及其约束
    model, A_var, D_var, W_var, T_var = add_time_var_and_cons(
        {}, args.trn_tbl_type, model, selected_train_list,
        sec_times, wait_time_lb, wait_time_ub,
        obj_type=args.obj_type,
        TimeSpan=args.TimeSpan,
        TimeStart=360,
        delta=args.TimeSpan
    )
    # 定义目标函数
    model, x_var = set_obj_func(model, selected_train_list, W_var, T_var, A_var, args.obj_type, args.obj_num,
                                solutionLimit=1e3, time_limit=600)
    # 增加同一站台任意一对车辆的安全间隔约束，以及防止站间越行的约束
    model, theta = add_safe_cons({}, args.trn_tbl_type, model, x_var, station_list, selected_train_list, dep_trains,
                                 arr_trains,
                                 pass_trains, A_var, D_var, aa_speed, dd_speed, pp_speed, ap_speed, pa_speed, dp_speed,
                                 pd_speed, sec_times, args.M)
    theta_aa, theta_ap, theta_pa, theta_pp, theta_dd, theta_dp, theta_pd = \
        theta['aa'], theta['ap'], theta['pa'], theta['pp'], theta['dd'], theta['dp'], theta['pd']
    model.setParam('MIPFocus', 1)  # 专注于可行解

    return model, theta_aa, theta_ap, theta_pa, theta_pp, theta_dd, theta_dp, theta_pd, x_var, D_var, A_var
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
