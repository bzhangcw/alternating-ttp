from gurobipy import *


def set_obj_func(model, train_list, W_var, T_var, force_feasibility=0, obj_num=-1, solutionLimit=1):
    """
    :param:
    force_feasibility: 0 必排标杆车，最大化排的车数
                       1 必排所有车，可行性问题
                       2 必排所有车，最小化总运行时间，找到第一个可行解就停止求解并返回
    """
    x_var = {}
    obj = LinExpr()
    for train in train_list:
        x_var[train.traNo] = model.addVar(vtype=GRB.BINARY, name='x_' + str(int(train.traNo)))
    sense = GRB.MAXIMIZE
    obj.add(quicksum(x_var.values()))
    model.setParam(GRB.Param.TimeLimit, 100)  # 找到可行解就停止求解并返回
    model.setObjective(obj, sense)

    return model, x_var


def add_time_var_and_cons(model, train_list, sec_times, wait_time_lb, wait_time_ub, TimeSpan=1080, TimeStart=360, delta=2):
    D_var = {}
    A_var = {}
    W_var = {}
    T_var = {}

    for train in train_list:
        D_var[train.traNo] = {}
        A_var[train.traNo] = {}
        W_var[train.traNo] = {}
        for station in train.staList:
            # 对经过站与停站增加时间变量与约束
            # 变量：该列车在该站台出发时间
            D_var[train.traNo][station] = model.addVar(vtype=GRB.CONTINUOUS,
                                                 ub=TimeStart + TimeSpan,
                                                 lb=TimeStart,
                                                 name="dep_t_" + str(int(train.traNo)) + "_" + station)
            # 变量：该列车在该站台的到达时间
            A_var[train.traNo][station] = model.addVar(vtype=GRB.CONTINUOUS,
                                                 ub=TimeStart + TimeSpan,
                                                 lb=TimeStart,
                                                 name="arr_t_" + str(int(train.traNo)) + "_" + station)
            # 变量：该列车在该站台的等待时间
            W_var[train.traNo][station] = model.addVar(vtype=GRB.CONTINUOUS,
                                                 ub=TimeSpan,
                                                 lb=0,
                                                 name="wait_t_" + str(int(train.traNo)) + "_" + station)
            # 约束： 该列车在该站台的等待时间 = 该列车在该站台出发时间 - 该列车在该站台的到达时间
            model.addConstr(W_var[train.traNo][station] == D_var[train.traNo][station] - A_var[train.traNo][station],
                            name="cons_" + str(train.traNo) + station)
            if train.depSta == station or train.arrSta == station:
                # 约束： 列车在出发站或到达站的等待时间为0
                model.addConstr(W_var[train.traNo][station] == 0,
                                name="cons_wait_time_equation_" + str(train.traNo) + station)
            elif train.linePlan[station] in {0, -1}:
                # 约束： 直接通过的列车等待时间为0
                model.addConstr(W_var[train.traNo][station] == 0,
                                name="cons_wait_time_equation_" + str(train.traNo) + station)
            elif train.linePlan[station] == 1:
                if train.standard == 0:  # 非标杆车
                    # 约束： 在非出发站或非到达站停站的列车等待时间不低于下界
                    model.addConstr(W_var[train.traNo][station] >= wait_time_lb[station],
                                    name="cons_wait_time_lb_" + str(train.traNo) + station)
                    # 约束： 停站的列车等待时间不高于上界
                    model.addConstr(W_var[train.traNo][station] <= wait_time_ub[station],
                                    name="cons_wait_time_ub_" + str(train.traNo) + station)
                elif train.standard == 1:  # 标杆车
                    # 约束： 在非出发站或非到达站停站的列车等待时间不低于下界
                    model.addConstr(W_var[train.traNo][station] >= wait_time_lb[station],
                                    name="cons_wait_time_high_lb_" + str(train.traNo) + station)
                    # 约束： 停站的列车等待时间不高于上界
                    model.addConstr(W_var[train.traNo][station] <= wait_time_ub[station],
                                    name="cons_wait_time_high_ub_" + str(train.traNo) + station)
            # # 标杆车出发时间偏好
            # if train.standard == 1:
            #     model.addConstr(D_var[train.traNo][train.depSta] >= train.preferred_time - delta)
            #     model.addConstr(D_var[train.traNo][train.depSta] <= train.preferred_time + delta)

    for train in train_list:
        T_var[train.traNo] = {}
        for station in train.staList[:-1]:
            # 变量：该列车在该站与下一站之间的运行时间
            T_var[train.traNo][station] = model.addVar(vtype=GRB.CONTINUOUS,
                                                 ub=TimeSpan,
                                                 lb=0,
                                                 name="run_t_" + str(int(train.traNo)) + "_" + station)
            # 约束：该列车在该站与下一站之间的运行时间 = 该列车在下一站的到达时间 - 该列车在该站出发时间
            model.addConstr(
                A_var[train.traNo][str(int(station) + int(2 * train.up - 1))] - D_var[train.traNo][station] == train.delta[
                    (station, str(int(station) + int(2 * train.up - 1)))] + T_var[train.traNo][station],
                name="cons_run_time_equation_" + str(train.traNo) + station)
            # # 约束：该列车在该站与下一站之间的运行时间不能小于最短运行时间
            # model.addConstr(T_var[train.traNo][station] >= sec_times[300][(station, str(int(station) + 1))],
            #                 name="cons_run_time_lb_" + str(train.traNo) + station)
            # # 约束：该列车在该站与下一站之间的运行时间不能大于人工设定的最长运行时间
            # model.addConstr(T_var[train.traNo][station] <= amplification_factor_of_run_time * sec_times[300][
            #     (station, str(int(station) + 1))], name="cons_run_time_lb_" + str(train.traNo) + station)

            # 约束：该列车在该站与下一站之间的通通运行时间等于给定的运行时间
            if train.up == 1:
                model.addConstr(T_var[train.traNo][station] == sec_times[train.speed][(station, str(int(station) + 1))],
                                name="cons_run_time_" + str(train.traNo) + station)
            elif train.up == 0:
                model.addConstr(T_var[train.traNo][station] == sec_times[train.speed][(str(int(station) - 1), station)],
                                name="cons_run_time_" + str(train.traNo) + station)

    return model, A_var, D_var, W_var, T_var


def add_safe_cons(model, x_var, station_list, stop_trains, pass_trains, A_var, D_var, aa_speed, dd_speed, pp_speed,
                  ap_speed, pa_speed, dp_speed, pd_speed, M, return_theta=True):
    theta_aa = {}
    theta_dd = {}
    theta_ap = {}
    theta_pa = {}
    theta_dp = {}
    theta_pd = {}
    theta_pp = {}

    # 安全间隔约束
    for sttn in station_list:
        theta_aa[sttn] = {}
        theta_dd[sttn] = {}
        theta_ap[sttn] = {}
        theta_dp[sttn] = {}
        theta_pa[sttn] = {}
        theta_pd[sttn] = {}
        theta_pp[sttn] = {}
        for trn in stop_trains[sttn]:
            theta_aa[sttn][trn.traNo] = {}
            theta_dd[sttn][trn.traNo] = {}
            theta_ap[sttn][trn.traNo] = {}
            theta_dp[sttn][trn.traNo] = {}
        for trn in pass_trains[sttn]:
            theta_pa[sttn][trn.traNo] = {}
            theta_pd[sttn][trn.traNo] = {}
            theta_pp[sttn][trn.traNo] = {}

        for stop_trn_1 in stop_trains[sttn]:
            for stop_trn_2 in stop_trains[sttn]:
                sec_name = sttn + "_" + str(stop_trn_1.traNo) + "_" + str(stop_trn_2.traNo)
                if stop_trn_1 != stop_trn_2:
                    theta_aa[sttn][stop_trn_1.traNo][stop_trn_2.traNo] = model.addVar(vtype=GRB.BINARY,
                                                                          name="theta_aa_" + sec_name)
                    theta_dd[sttn][stop_trn_1.traNo][stop_trn_2.traNo] = model.addVar(vtype=GRB.BINARY,
                                                                          name="theta_dd_" + sec_name)

        for i in range(len(stop_trains[sttn])):
            stop_trn_1 = stop_trains[sttn][i]
            for j in range(i + 1, len(stop_trains[sttn])):
                stop_trn_2 = stop_trains[sttn][j]

                # 约束：列车次序
                # 到到 aa
                model.addConstr(theta_aa[sttn][stop_trn_1.traNo][stop_trn_2.traNo] + theta_aa[sttn][stop_trn_2.traNo][stop_trn_1.traNo] == 1)
                # 发发 dd
                model.addConstr(theta_dd[sttn][stop_trn_1.traNo][stop_trn_2.traNo] + theta_dd[sttn][stop_trn_2.traNo][stop_trn_1.traNo] == 1)

                # 约束：列车安全间隔
                # 到到 aa
                model.addConstr(A_var[stop_trn_2.traNo][sttn] + M * (
                        3 - x_var[stop_trn_1.traNo] - x_var[stop_trn_2.traNo] - theta_aa[sttn][stop_trn_1.traNo][stop_trn_2.traNo]) >=
                                A_var[stop_trn_1.traNo][sttn] + aa_speed[stop_trn_2.speed][sttn])
                model.addConstr(A_var[stop_trn_1.traNo][sttn] + M * (
                        3 - x_var[stop_trn_1.traNo] - x_var[stop_trn_2.traNo] - theta_aa[sttn][stop_trn_2.traNo][stop_trn_1.traNo]) >=
                                A_var[stop_trn_2.traNo][sttn] + aa_speed[stop_trn_1.speed][sttn])
                # 发发 dd
                model.addConstr(D_var[stop_trn_2.traNo][sttn] + M * (
                        3 - x_var[stop_trn_1.traNo] - x_var[stop_trn_2.traNo] - theta_dd[sttn][stop_trn_1.traNo][stop_trn_2.traNo]) >=
                                D_var[stop_trn_1.traNo][sttn] + dd_speed[stop_trn_2.speed][sttn])
                model.addConstr(D_var[stop_trn_1.traNo][sttn] + M * (
                        3 - x_var[stop_trn_1.traNo] - x_var[stop_trn_2.traNo] - theta_dd[sttn][stop_trn_2.traNo][stop_trn_1.traNo]) >=
                                D_var[stop_trn_2.traNo][sttn] + dd_speed[stop_trn_1.speed][sttn])

        for pass_trn_1 in pass_trains[sttn]:
            for pass_trn_2 in pass_trains[sttn]:
                sec_name = sttn + "_" + str(pass_trn_1.traNo) + "_" + str(pass_trn_2.traNo)
                if pass_trn_1.traNo != pass_trn_2.traNo:
                    theta_pp[sttn][pass_trn_1.traNo][pass_trn_2.traNo] = model.addVar(vtype=GRB.BINARY,
                                                                          name="theta_pp_" + sec_name)

        for i in range(len(pass_trains[sttn])):
            pass_trn_1 = pass_trains[sttn][i]
            for j in range(i + 1, len(pass_trains[sttn])):
                pass_trn_2 = pass_trains[sttn][j]
                # 约束：列车次序 通通 pp
                model.addConstr(theta_pp[sttn][pass_trn_1.traNo][pass_trn_2.traNo] + theta_pp[sttn][pass_trn_2.traNo][pass_trn_1.traNo] == 1)

                # 约束：列车安全间隔 通通 pp
                model.addConstr(A_var[pass_trn_2.traNo][sttn] + M * (
                        3 - x_var[pass_trn_1.traNo] - x_var[pass_trn_2.traNo] - theta_pp[sttn][pass_trn_1.traNo][pass_trn_2.traNo]) >=
                                A_var[pass_trn_1.traNo][sttn] + pp_speed[pass_trn_2.speed][sttn])
                model.addConstr(A_var[pass_trn_1.traNo][sttn] + M * (
                        3 - x_var[pass_trn_1.traNo] - x_var[pass_trn_2.traNo] - theta_pp[sttn][pass_trn_2.traNo][pass_trn_1.traNo]) >=
                                A_var[pass_trn_2.traNo][sttn] + pp_speed[pass_trn_1.speed][sttn])

        for stop_trn in stop_trains[sttn]:
            for pass_train in pass_trains[sttn]:
                sec_name = sttn + "_" + str(stop_trn.traNo) + "_" + str(pass_train.traNo)
                theta_ap[sttn][stop_trn.traNo][pass_train.traNo] = model.addVar(vtype=GRB.BINARY,
                                                                    name="theta_ap_" + sec_name)
                theta_pa[sttn][pass_train.traNo][stop_trn.traNo] = model.addVar(vtype=GRB.BINARY,
                                                                    name="theta_pa_" + sec_name)
                theta_dp[sttn][stop_trn.traNo][pass_train.traNo] = model.addVar(vtype=GRB.BINARY,
                                                                    name="theta_dp_" + sec_name)
                theta_pd[sttn][pass_train.traNo][stop_trn.traNo] = model.addVar(vtype=GRB.BINARY,
                                                                    name="theta_pd_" + sec_name)

        for stop_trn in stop_trains[sttn]:
            for pass_train in pass_trains[sttn]:
                # 约束：列车次序 到通与通到 ap pa
                model.addConstr(theta_ap[sttn][stop_trn.traNo][pass_train.traNo] + theta_pa[sttn][pass_train.traNo][stop_trn.traNo] == 1)
                # 约束：列车次序 发通与通发 dp pd
                model.addConstr(theta_dp[sttn][stop_trn.traNo][pass_train.traNo] + theta_pd[sttn][pass_train.traNo][stop_trn.traNo] == 1)

                # 约束：列车安全间隔
                # 到通与通到 ap pa
                model.addConstr(A_var[pass_train.traNo][sttn] + M * (
                        3 - x_var[stop_trn.traNo] - x_var[pass_train.traNo] - theta_ap[sttn][stop_trn.traNo][pass_train.traNo]) >=
                                A_var[stop_trn.traNo][sttn] + ap_speed[pass_train.speed][sttn])
                model.addConstr(A_var[stop_trn.traNo][sttn] + M * (
                        3 - x_var[stop_trn.traNo] - x_var[pass_train.traNo] - theta_pa[sttn][pass_train.traNo][stop_trn.traNo]) >=
                                A_var[pass_train.traNo][sttn] + pa_speed[stop_trn.speed][sttn])
                # 发通与通发 dp pd
                model.addConstr(D_var[pass_train.traNo][sttn] + M * (
                        3 - x_var[stop_trn.traNo] - x_var[pass_train.traNo] - theta_dp[sttn][stop_trn.traNo][pass_train.traNo]) >=
                                D_var[stop_trn.traNo][sttn] + dp_speed[pass_train.speed][sttn])
                model.addConstr(D_var[stop_trn.traNo][sttn] + M * (
                        3 - x_var[stop_trn.traNo] - x_var[pass_train.traNo] - theta_pd[sttn][pass_train.traNo][stop_trn.traNo]) >=
                                D_var[pass_train.traNo][sttn] + pd_speed[stop_trn.speed][sttn])

    # 防止站间越行约束
    for sttn in station_list:
        for trn1 in stop_trains[sttn]:
            for trn2 in stop_trains[sttn]:
                if trn1.traNo != trn2.traNo:
                    nextStation1 = str(int(sttn) + int(2 * trn1.up - 1))  # trn1的下一站
                    nextStation2 = str(int(sttn) + int(2 * trn2.up - 1))  # trn2的下一站
                    if nextStation1 in trn1.staList and nextStation2 in trn2.staList and nextStation1 == nextStation2:
                        # trn1: da, trn2: da
                        if trn1.linePlan[nextStation1] == 1 and trn2.linePlan[nextStation2] == 1:
                            model.addConstr(theta_dd[sttn][trn1.traNo][trn2.traNo] == theta_aa[nextStation1][trn1.traNo][trn2.traNo])
                        # trn1: da, trn2: dp
                        elif trn1.linePlan[nextStation1] == 1 and trn2.linePlan[nextStation2] in (0, -1):
                            model.addConstr(theta_dd[sttn][trn1.traNo][trn2.traNo] == theta_ap[nextStation1][trn1.traNo][trn2.traNo])
                        # trn1: dp, trn2: da
                        elif trn1.linePlan[nextStation1] in (0, -1) and trn2.linePlan[nextStation2] == 1:
                            model.addConstr(theta_dd[sttn][trn1.traNo][trn2.traNo] == theta_pa[nextStation1][trn1.traNo][trn2.traNo])
                        # trn1: dp, trn2: dp
                        elif trn1.linePlan[nextStation1] in (0, -1) and trn2.linePlan[nextStation2] in (0, -1):
                            model.addConstr(theta_dd[sttn][trn1.traNo][trn2.traNo] == theta_pp[nextStation1][trn1.traNo][trn2.traNo])

            for trn2 in pass_trains[sttn]:
                if trn1.traNo != trn2.traNo:
                    nextStation1 = str(int(sttn) + int(2 * trn1.up - 1))  # trn1的下一站
                    nextStation2 = str(int(sttn) + int(2 * trn2.up - 1))  # trn2的下一站
                    if nextStation1 in trn1.staList and nextStation2 in trn2.staList and nextStation1 == nextStation2:
                        # trn1: da, trn2: pa
                        if trn1.linePlan[nextStation1] == 1 and trn2.linePlan[nextStation2] == 1:
                            model.addConstr(theta_dp[sttn][trn1.traNo][trn2.traNo] == theta_aa[nextStation1][trn1.traNo][trn2.traNo])
                        # trn1: da, trn2: pp
                        elif trn1.linePlan[nextStation1] == 1 and trn2.linePlan[nextStation2] in (0, -1):
                            model.addConstr(theta_dp[sttn][trn1.traNo][trn2.traNo] == theta_ap[nextStation1][trn1.traNo][trn2.traNo])
                        # trn1: dp, trn2: pa
                        elif trn1.linePlan[nextStation1] in (0, -1) and trn2.linePlan[nextStation2] == 1:
                            model.addConstr(theta_dp[sttn][trn1.traNo][trn2.traNo] == theta_pa[nextStation1][trn1.traNo][trn2.traNo])
                        # trn1: dp, trn2: pp
                        elif trn1.linePlan[nextStation1] in (0, -1) and trn2.linePlan[nextStation2] in (0, -1):
                            model.addConstr(theta_dp[sttn][trn1.traNo][trn2.traNo] == theta_pp[nextStation1][trn1.traNo][trn2.traNo])

        for trn1 in pass_trains[sttn]:
            for trn2 in stop_trains[sttn]:
                if trn1.traNo != trn2.traNo:
                    nextStation1 = str(int(sttn) + int(2 * trn1.up - 1))  # trn1的下一站
                    nextStation2 = str(int(sttn) + int(2 * trn2.up - 1))  # trn2的下一站
                    if nextStation1 in trn1.staList and nextStation2 in trn2.staList and nextStation1 == nextStation2:
                        # trn1: pa, trn2: da
                        if trn1.linePlan[nextStation1] == 1 and trn2.linePlan[nextStation2] == 1:
                            model.addConstr(theta_pd[sttn][trn1.traNo][trn2.traNo] == theta_aa[nextStation1][trn1.traNo][trn2.traNo])
                        # trn1: pa, trn2: dp
                        elif trn1.linePlan[nextStation1] == 1 and trn2.linePlan[nextStation2] in (0, -1):
                            model.addConstr(theta_pd[sttn][trn1.traNo][trn2.traNo] == theta_ap[nextStation1][trn1.traNo][trn2.traNo])
                        # trn1: pp, trn2: da
                        elif trn1.linePlan[nextStation1] in (0, -1) and trn2.linePlan[nextStation2] == 1:
                            model.addConstr(theta_pd[sttn][trn1.traNo][trn2.traNo] == theta_pa[nextStation1][trn1.traNo][trn2.traNo])
                        # trn1: pp, trn2: dp
                        elif trn1.linePlan[nextStation1] in (0, -1) and trn2.linePlan[nextStation2] in (0, -1):
                            model.addConstr(theta_pd[sttn][trn1.traNo][trn2.traNo] == theta_pp[nextStation1][trn1.traNo][trn2.traNo])

            for trn2 in pass_trains[sttn]:
                if trn1.traNo != trn2.traNo:
                    nextStation1 = str(int(sttn) + int(2 * trn1.up - 1))  # trn1的下一站
                    nextStation2 = str(int(sttn) + int(2 * trn2.up - 1))  # trn2的下一站
                    if nextStation1 in trn1.staList and nextStation2 in trn2.staList and nextStation1 == nextStation2:
                        # trn1: pa, trn2: pa
                        if trn1.linePlan[nextStation1] == 1 and trn2.linePlan[nextStation2] == 1:
                            model.addConstr(theta_pp[sttn][trn1.traNo][trn2.traNo] == theta_aa[nextStation1][trn1.traNo][trn2.traNo])
                        # trn1: pa, trn2: pp
                        elif trn1.linePlan[nextStation1] == 1 and trn2.linePlan[nextStation2] in (0, -1):
                            model.addConstr(theta_pp[sttn][trn1.traNo][trn2.traNo] == theta_ap[nextStation1][trn1.traNo][trn2.traNo])
                        # trn1: pp, trn2: pa
                        elif trn1.linePlan[nextStation1] in (0, -1) and trn2.linePlan[nextStation2] == 1:
                            model.addConstr(theta_pp[sttn][trn1.traNo][trn2.traNo] == theta_pa[nextStation1][trn1.traNo][trn2.traNo])
                        # trn1: pp, trn2: pp
                        elif trn1.linePlan[nextStation1] in (0, -1) and trn2.linePlan[nextStation2] in (0, -1):
                            model.addConstr(theta_pp[sttn][trn1.traNo][trn2.traNo] == theta_pp[nextStation1][trn1.traNo][trn2.traNo])
    if return_theta:
        return model, theta_aa, theta_ap, theta_pa, theta_pp, theta_dd, theta_dp, theta_pd
    else:
        return model
