from gurobipy import *

from model import set_obj_func, add_time_var_and_cons, add_safe_cons
from util import get_dep_arr_pass_trains


def get_schedule(pre_train_table, trn_tbl_type, selected_train_list, station_list, args, sec_times, wait_time_lb,
                 wait_time_ub, aa_speed, dd_speed, pp_speed, ap_speed, pa_speed, dp_speed, pd_speed):
    # 数据预处理：得到所有车站的停站车辆和通过车辆
    dep_trains, arr_trains, pass_trains = get_dep_arr_pass_trains(station_list, selected_train_list)

    # 建立模型与设置参数
    model = Model('TTP_JSP')
    model.setParam('OutputFlag', 1)
    model.setParam('MIPFocus', 1)  # 专注于可行解
    model.setParam('Presolve', 2)  # 提高presolve的力度
    # 增加各占出发时间变量、到达时间变量及其约束
    model, A_var, D_var, W_var, T_var = add_time_var_and_cons(pre_train_table, trn_tbl_type, model, selected_train_list,
                                                              sec_times, wait_time_lb, wait_time_ub, args.obj_type,
                                                              args.TimeSpan)
    # 定义目标函数
    model, x_var = set_obj_func(model, selected_train_list, W_var, T_var, A_var, args.obj_type, args.obj_num,
                                args.sol_limit, args.time_limit)
    # 增加同一站台任意一对车辆的安全间隔约束，以及防止站间越行的约束
    model = add_safe_cons(pre_train_table, trn_tbl_type, model, x_var, station_list, selected_train_list, dep_trains,
                          arr_trains, pass_trains, A_var, D_var, aa_speed, dd_speed, pp_speed, ap_speed, pa_speed,
                          dp_speed, pd_speed, sec_times, args.M)
    # 模型求解
    model.optimize()

    if model.status != GRB.INFEASIBLE:
        print('\n Total running time: {:.4f}'.format(model.runtime))

        if model.status != GRB.INTERRUPTED:
            model.write("output/ttp_jsp_{0}_{1}.sol".format(len(selected_train_list), int(model.runtime)))
    else:
        model.computeIIS()
        model.write("output/ttp_jsp_{}.ilp".format(len(selected_train_list)))

    train_table = {}
    for train in selected_train_list:
        train_id = train.traNo
        train_table[train_id] = {}
        for station in train.staList:
            train_table[train_id][station] = {}
            train_table[train_id][station]['dep'] = round(D_var[train][station].x, 2)
            train_table[train_id][station]['arr'] = round(A_var[train][station].x, 2)

    return train_table
