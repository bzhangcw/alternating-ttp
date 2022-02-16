from gurobipy import *


def set_obj_func(model, train_list, W_var, T_var, A_var, obj_type='max_trn_num', obj_num=-1, solutionLimit=1,
                 time_limit=3600):
    """
    :param:
    obj_type:   max_trn_num   必排标杆车，最大化排的车数
                max_miles     必排标杆车，最大化里程数
                get_fsb_sol   必排所有车，可行性问题
                min_run_time  必排所有车，最小化总运行时间，找到第一个可行解就停止求解并返回
                min_time_span 必排所有车，最小化到达时间
    """
    x_var = {}
    x_var_list = []
    obj = LinExpr()
    sense = None

    if obj_type == "max_miles":
        miles_list = []
        for train in train_list:
            if train.standard == 1:
                x_var[train] = 1
            else:
                x_var[train] = model.addVar(vtype=GRB.BINARY, name='x_' + str(int(train.traNo)))
                x_var_list.append(x_var[train])
                miles_list.append(train.miles)
        obj.addTerms(miles_list, x_var_list)
        sense = GRB.MAXIMIZE
    elif obj_type in ("max_trn_num", "get_fsb_sol"):
        for train in train_list:
            # 标杆车优先排
            if train.standard == 1 or obj_type == "get_fsb_sol":
                x_var[train] = 1
            elif train.standard == 0 and obj_type == "max_trn_num":
                x_var[train] = model.addVar(vtype=GRB.BINARY, name='x_' + str(int(train.traNo)))
            x_var_list.append(x_var[train])
        # 最大化 排入的列车数
        obj.add(quicksum(x_var_list))
        sense = GRB.MAXIMIZE
        if obj_type == "max_trn_num" and obj_num > 0:
            model.params.BestObjStop = min(obj_num, len(train_list))  # 不一定求到最优
    elif obj_type == "min_run_time":
        for train in train_list:
            x_var[train] = 1
        # 最小化 总列车运行时间
        for train in train_list:
            obj.add(quicksum(W_var[train].values()))
            # obj.add(quicksum(T_var[train].values()))
        sense = GRB.MINIMIZE
        # model.setParam('SolutionLimit', solutionLimit)  # 限制可行解的数量
        model.setParam('TimeLimit', time_limit)
    elif obj_type == "min_time_span":
        for train in train_list:
            x_var[train] = 1
            if train.standard != 1:
                obj.add(A_var[train][train.arrSta])
        sense = GRB.MINIMIZE
        model.setParam('SolutionLimit', solutionLimit)  # 限制可行解的数量
        model.setParam('TimeLimit', time_limit)
    model.setObjective(obj, sense)
    return model, x_var


def add_time_var_and_cons(pre_train_table, trn_tbl_type, model, train_list, sec_times, wait_time_lb, wait_time_ub,
                          obj_type, TimeSpan=1080, TimeStart=360, delta=2):
    if obj_type == "min_time_span":
        # TimeSpan = GRB.INFINITY
        TimeSpan = 1080 * 2

    D_var = {}
    A_var = {}
    W_var = {}
    T_var = {}

    for train in train_list:
        D_var[train] = {}
        A_var[train] = {}
        W_var[train] = {}
        for station in train.staList:
            trn_sttn_name = str(int(train.traNo)) + "_" + station
            # 对经过站与停站增加时间变量与约束
            # 变量：该列车在该站台出发时间
            D_var[train][station] = model.addVar(vtype=GRB.CONTINUOUS,
                                                 ub=TimeStart + TimeSpan,
                                                 lb=TimeStart,
                                                 name="dep_t_" + trn_sttn_name)
            # 变量：该列车在该站台的到达时间
            A_var[train][station] = model.addVar(vtype=GRB.CONTINUOUS,
                                                 ub=TimeStart + TimeSpan,
                                                 lb=TimeStart,
                                                 name="arr_t_" + trn_sttn_name)
            # 变量：该列车在该站台的等待时间
            W_var[train][station] = model.addVar(vtype=GRB.CONTINUOUS,
                                                 ub=TimeSpan,
                                                 lb=0,
                                                 name="wait_t_" + trn_sttn_name)
            # 约束： 该列车在该站台的等待时间 = 该列车在该站台出发时间 - 该列车在该站台的到达时间
            model.addConstr(W_var[train][station] == D_var[train][station] - A_var[train][station],
                            name="cons_" + trn_sttn_name)
            if train.depSta == station or train.arrSta == station:
                # 约束： 列车在出发站或到达站的等待时间为0
                model.addConstr(W_var[train][station] == 0,
                                name="cons_wait_time_equation_" + trn_sttn_name)
            elif train.linePlan[station] in {0, -1}:
                # 约束： 直接通过的列车等待时间为0
                model.addConstr(W_var[train][station] == 0,
                                name="cons_wait_time_equation_" + trn_sttn_name)
            elif train.linePlan[station] == 1:
                if train.standard == 0:  # 非标杆车
                    # 约束： 在非出发站或非到达站停站的列车等待时间不低于下界
                    model.addConstr(W_var[train][station] >= wait_time_lb[station],
                                    name="cons_wait_time_lb_" + trn_sttn_name)
                    # 约束： 停站的列车等待时间不高于上界
                    model.addConstr(W_var[train][station] <= wait_time_ub[station],
                                    name="cons_wait_time_ub_" + trn_sttn_name)
                elif train.standard == 1:  # 标杆车
                    # 约束： 在非出发站或非到达站停站的列车等待时间不低于下界
                    model.addConstr(W_var[train][station] >= wait_time_lb[station],
                                    name="cons_wait_time_high_lb_" + trn_sttn_name)
                    # 约束： 停站的列车等待时间不高于上界
                    model.addConstr(W_var[train][station] <= wait_time_ub[station],
                                    name="cons_wait_time_high_ub_" + trn_sttn_name)
            # 标杆车出发时间偏好
            if train.standard == 1:
                model.addConstr(D_var[train][train.depSta] >= train.preferred_time - delta)
                model.addConstr(D_var[train][train.depSta] <= train.preferred_time + delta)

    for train in train_list:
        T_var[train] = {}
        for station in train.staList[:-1]:
            trn_sttn_name = str(int(train.traNo)) + "_" + station
            # 变量：该列车在该站与下一站之间的运行时间
            T_var[train][station] = model.addVar(vtype=GRB.CONTINUOUS,
                                                 ub=TimeSpan,
                                                 lb=0,
                                                 name="run_t_" + trn_sttn_name)
            # 约束：该列车在该站与下一站之间的运行时间 = 该列车在下一站的到达时间 - 该列车在该站出发时间
            model.addConstr(
                A_var[train][str(int(station) + int(2 * train.up - 1))] - D_var[train][station] == train.delta[
                    (station, str(int(station) + int(2 * train.up - 1)))] + T_var[train][station],
                name="cons_run_time_equation_" + trn_sttn_name)
            # # 约束：该列车在该站与下一站之间的运行时间不能小于最短运行时间
            # model.addConstr(T_var[train][station] >= sec_times[300][(station, str(int(station) + 1))],
            #                 name="cons_run_time_lb_" + str(train.traNo) + station)
            # # 约束：该列车在该站与下一站之间的运行时间不能大于人工设定的最长运行时间
            # model.addConstr(T_var[train][station] <= amplification_factor_of_run_time * sec_times[300][
            #     (station, str(int(station) + 1))], name="cons_run_time_lb_" + str(train.traNo) + station)

            # 约束：该列车在该站与下一站之间的通通运行时间等于给定的运行时间
            if train.up == 1:
                model.addConstr(T_var[train][station] == sec_times[train.speed][(station, str(int(station) + 1))],
                                name="cons_run_time_" + trn_sttn_name)
            elif train.up == 0:
                model.addConstr(T_var[train][station] == sec_times[train.speed][(str(int(station) - 1), station)],
                                name="cons_run_time_" + trn_sttn_name)

    # 根据给定的列车运行表固定列车的出发和到达时间
    if trn_tbl_type == "time":
        for train in train_list:
            if train.traNo in pre_train_table.keys():
                for station in pre_train_table[train.traNo].keys():
                    model.addConstr(A_var[train][station] == pre_train_table[train.traNo][station]['arr'])
                    model.addConstr(D_var[train][station] == pre_train_table[train.traNo][station]['dep'])

    return model, A_var, D_var, W_var, T_var


def add_safe_cons(pre_train_table, trn_tbl_type, model, x_var, station_list, train_list, dep_trains, arr_trains, pass_trains,
                  A_var, D_var, aa_speed, dd_speed, pp_speed, ap_speed, pa_speed, dp_speed, pd_speed, sec_times, M):
    pair_states = ['aa', 'dd', 'ap', 'pa', 'dp', 'pd', 'pp']

    theta = {}
    safe_constrs = {}
    for pair_state in pair_states:
        theta[pair_state] = {}
        safe_constrs[pair_state] = {}
        for sttn in station_list:
            theta[pair_state][sttn] = {}
            safe_constrs[pair_state][sttn] = {}

    # 安全间隔约束
    for sttn in station_list:
        for trn in dep_trains[sttn]:
            theta['dd'][sttn][trn] = {}
            theta['dp'][sttn][trn] = {}
            safe_constrs['dd'][sttn][trn] = {}
            safe_constrs['dp'][sttn][trn] = {}
        
        for trn in arr_trains[sttn]:
            theta['aa'][sttn][trn] = {}
            theta['ap'][sttn][trn] = {}
            safe_constrs['aa'][sttn][trn] = {}
            safe_constrs['ap'][sttn][trn] = {}

        for trn in pass_trains[sttn]:
            theta['pa'][sttn][trn] = {}
            theta['pd'][sttn][trn] = {}
            theta['pp'][sttn][trn] = {}
            safe_constrs['pa'][sttn][trn] = {}
            safe_constrs['pd'][sttn][trn] = {}
            safe_constrs['pp'][sttn][trn] = {}

        for dep_trn_1 in dep_trains[sttn]:
            for dep_trn_2 in dep_trains[sttn]:
                sec_name = sttn + "_" + str(dep_trn_1.traNo) + "_" + str(dep_trn_2.traNo)
                if dep_trn_1 != dep_trn_2:
                    theta['dd'][sttn][dep_trn_1][dep_trn_2] = model.addVar(vtype=GRB.BINARY,
                                                                             name="theta_dd_" + sec_name)

        for arr_trn_1 in arr_trains[sttn]:
            for arr_trn_2 in arr_trains[sttn]:
                sec_name = sttn + "_" + str(arr_trn_1.traNo) + "_" + str(arr_trn_2.traNo)
                if arr_trn_1 != arr_trn_2:
                    theta['aa'][sttn][arr_trn_1][arr_trn_2] = model.addVar(vtype=GRB.BINARY,
                                                                             name="theta_aa_" + sec_name)


        for i in range(len(dep_trains[sttn])):
            dep_trn_1 = dep_trains[sttn][i]
            for j in range(i + 1, len(dep_trains[sttn])):
                dep_trn_2 = dep_trains[sttn][j]

                # 约束：列车次序
                # 发发 dd
                model.addConstr(
                    theta['dd'][sttn][dep_trn_1][dep_trn_2] + theta['dd'][sttn][dep_trn_2][dep_trn_1] == 1)

                # 约束：列车安全间隔
                # 发发 dd
                safe_constrs['dd'][sttn][dep_trn_1][dep_trn_2] = model.addConstr(D_var[dep_trn_2][sttn] + M * (
                        3 - x_var[dep_trn_1] - x_var[dep_trn_2] - theta['dd'][sttn][dep_trn_1][dep_trn_2]) >=
                                                                                   D_var[dep_trn_1][sttn] +
                                                                                   dd_speed[dep_trn_2.speed][sttn])
                safe_constrs['dd'][sttn][dep_trn_2][dep_trn_1] = model.addConstr(D_var[dep_trn_1][sttn] + M * (
                        3 - x_var[dep_trn_1] - x_var[dep_trn_2] - theta['dd'][sttn][dep_trn_2][dep_trn_1]) >=
                                                                                   D_var[dep_trn_2][sttn] +
                                                                                   dd_speed[dep_trn_1.speed][sttn])

        for i in range(len(arr_trains[sttn])):
            arr_trn_1 = arr_trains[sttn][i]
            for j in range(i + 1, len(arr_trains[sttn])):
                arr_trn_2 = arr_trains[sttn][j]

                # 约束：列车次序
                # 到到 aa
                model.addConstr(
                    theta['aa'][sttn][arr_trn_1][arr_trn_2] + theta['aa'][sttn][arr_trn_2][arr_trn_1] == 1)
                
                # 约束：列车安全间隔
                # 到到 aa
                safe_constrs['aa'][sttn][arr_trn_1][arr_trn_2] = model.addConstr(A_var[arr_trn_2][sttn] + M * (
                        3 - x_var[arr_trn_1] - x_var[arr_trn_2] - theta['aa'][sttn][arr_trn_1][arr_trn_2]) >=
                                                                                   A_var[arr_trn_1][sttn] +
                                                                                   aa_speed[arr_trn_2.speed][sttn])
                safe_constrs['aa'][sttn][arr_trn_2][arr_trn_1] = model.addConstr(A_var[arr_trn_1][sttn] + M * (
                        3 - x_var[arr_trn_1] - x_var[arr_trn_2] - theta['aa'][sttn][arr_trn_2][arr_trn_1]) >=
                                                                                   A_var[arr_trn_2][sttn] +
                                                                                   aa_speed[arr_trn_1.speed][sttn])

        for pass_trn_1 in pass_trains[sttn]:
            for pass_trn_2 in pass_trains[sttn]:
                sec_name = sttn + "_" + str(pass_trn_1.traNo) + "_" + str(pass_trn_2.traNo)
                if pass_trn_1 != pass_trn_2:
                    theta['pp'][sttn][pass_trn_1][pass_trn_2] = model.addVar(vtype=GRB.BINARY,
                                                                             name="theta_pp_" + sec_name)

        for i in range(len(pass_trains[sttn])):
            pass_trn_1 = pass_trains[sttn][i]
            for j in range(i + 1, len(pass_trains[sttn])):
                pass_trn_2 = pass_trains[sttn][j]
                # 约束：列车次序 通通 pp
                model.addConstr(
                    theta['pp'][sttn][pass_trn_1][pass_trn_2] + theta['pp'][sttn][pass_trn_2][pass_trn_1] == 1)

                # 约束：列车安全间隔 通通 pp
                safe_constrs['pp'][sttn][pass_trn_1][pass_trn_2] = model.addConstr(A_var[pass_trn_2][sttn] + M * (
                        3 - x_var[pass_trn_1] - x_var[pass_trn_2] - theta['pp'][sttn][pass_trn_1][pass_trn_2]) >=
                                                                                   A_var[pass_trn_1][sttn] +
                                                                                   pp_speed[pass_trn_2.speed][sttn])
                safe_constrs['pp'][sttn][pass_trn_2][pass_trn_1] = model.addConstr(A_var[pass_trn_1][sttn] + M * (
                        3 - x_var[pass_trn_1] - x_var[pass_trn_2] - theta['pp'][sttn][pass_trn_2][pass_trn_1]) >=
                                                                                   A_var[pass_trn_2][sttn] +
                                                                                   pp_speed[pass_trn_1.speed][sttn])

        for dep_trn in dep_trains[sttn]:
            for pass_trn in pass_trains[sttn]:
                sec_name = sttn + "_" + str(dep_trn.traNo) + "_" + str(pass_trn.traNo)
                theta['dp'][sttn][dep_trn][pass_trn] = model.addVar(vtype=GRB.BINARY,
                                                                     name="theta_dp_" + sec_name)
                theta['pd'][sttn][pass_trn][dep_trn] = model.addVar(vtype=GRB.BINARY,
                                                                     name="theta_pd_" + sec_name)

        for arr_trn in arr_trains[sttn]:
            for pass_trn in pass_trains[sttn]:
                sec_name = sttn + "_" + str(arr_trn.traNo) + "_" + str(pass_trn.traNo)
                theta['ap'][sttn][arr_trn][pass_trn] = model.addVar(vtype=GRB.BINARY,
                                                                     name="theta_ap_" + sec_name)
                theta['pa'][sttn][pass_trn][arr_trn] = model.addVar(vtype=GRB.BINARY,
                                                                     name="theta_pa_" + sec_name)


        for dep_trn in dep_trains[sttn]:
            for pass_trn in pass_trains[sttn]:
                # 约束：列车次序 发通与通发 dp pd
                model.addConstr(theta['dp'][sttn][dep_trn][pass_trn] + theta['pd'][sttn][pass_trn][dep_trn] == 1)

                # 约束：列车安全间隔
                # 发通与通发 dp pd
                safe_constrs['dp'][sttn][dep_trn][pass_trn] = model.addConstr(D_var[pass_trn][sttn] + M * (
                        3 - x_var[dep_trn] - x_var[pass_trn] - theta['dp'][sttn][dep_trn][pass_trn]) >=
                                                                               D_var[dep_trn][sttn] +
                                                                               dp_speed[pass_trn.speed][sttn])
                safe_constrs['pd'][sttn][pass_trn][dep_trn] = model.addConstr(D_var[dep_trn][sttn] + M * (
                        3 - x_var[dep_trn] - x_var[pass_trn] - theta['pd'][sttn][pass_trn][dep_trn]) >=
                                                                               D_var[pass_trn][sttn] +
                                                                               pd_speed[dep_trn.speed][sttn])

        for arr_trn in arr_trains[sttn]:
            for pass_trn in pass_trains[sttn]:
                # 约束：列车次序 到通与通到 ap pa
                model.addConstr(theta['ap'][sttn][arr_trn][pass_trn] + theta['pa'][sttn][pass_trn][arr_trn] == 1)
                
                # 约束：列车安全间隔
                # 到通与通到 ap pa
                safe_constrs['ap'][sttn][arr_trn][pass_trn] = model.addConstr(A_var[pass_trn][sttn] + M * (
                        3 - x_var[arr_trn] - x_var[pass_trn] - theta['ap'][sttn][arr_trn][pass_trn]) >=
                                                                               A_var[arr_trn][sttn] +
                                                                               ap_speed[pass_trn.speed][sttn])
                safe_constrs['pa'][sttn][pass_trn][arr_trn] = model.addConstr(A_var[arr_trn][sttn] + M * (
                        3 - x_var[arr_trn] - x_var[pass_trn] - theta['pa'][sttn][pass_trn][arr_trn]) >=
                                                                               A_var[pass_trn][sttn] +
                                                                               pa_speed[arr_trn.speed][sttn])

    # 根据给定的列车运行表固定列车的相对顺序
    if trn_tbl_type == "order":
        for train_1 in train_list:
            for train_2 in train_list:
                if train_1.traNo in pre_train_table.keys() and train_2.traNo in pre_train_table.keys():
                    station_intersection = set(pre_train_table[train_1.traNo].keys()) & set(
                        pre_train_table[train_2.traNo].keys())

                    for station in station_intersection:
                        if train_1 in pass_trains[station]:
                            if train_2 in pass_trains[station] and train_1 != train_2:
                                # 通通 pp
                                time_diff = pre_train_table[train_1.traNo][station]['arr'] - \
                                            pre_train_table[train_2.traNo][station]['arr']
                                fixed_theta = 1 if time_diff < 0 else 0
                                model.addConstr(theta['pp'][station][train_1][train_2] == fixed_theta)
                            elif train_2 in dep_trains[station]:
                                # 通发 pd
                                time_diff = pre_train_table[train_1.traNo][station]['arr'] - \
                                            pre_train_table[train_2.traNo][station]['dep']
                                fixed_theta = 1 if time_diff < 0 else 0
                                model.addConstr(theta['pd'][station][train_1][train_2] == fixed_theta)
                            elif train_2 in arr_trains[station]:
                                # 通到 pa
                                time_diff = pre_train_table[train_1.traNo][station]['arr'] - \
                                            pre_train_table[train_2.traNo][station]['arr']
                                fixed_theta = 1 if time_diff < 0 else 0
                                model.addConstr(theta['pa'][station][train_1][train_2] == fixed_theta)

                        elif train_1 in dep_trains[station]:
                            if train_2 in pass_trains[station]:
                                # 发通 dp
                                time_diff = pre_train_table[train_1.traNo][station]['dep'] - \
                                            pre_train_table[train_2.traNo][station]['arr']
                                fixed_theta = 1 if time_diff < 0 else 0
                                model.addConstr(theta['dp'][station][train_1][train_2] == fixed_theta)
                            elif train_2 in dep_trains[station] and train_1 != train_2:
                                # 发发 dd
                                time_diff = pre_train_table[train_1.traNo][station]['dep'] - \
                                            pre_train_table[train_2.traNo][station]['dep']
                                fixed_theta = 1 if time_diff < 0 else 0
                                model.addConstr(theta['dd'][station][train_1][train_2] == fixed_theta)
                        
                        elif train_1 in arr_trains[station]:
                            if train_2 in pass_trains[station]:
                                # 到通 ap
                                time_diff = pre_train_table[train_1.traNo][station]['arr'] - \
                                            pre_train_table[train_2.traNo][station]['arr']
                                fixed_theta = 1 if time_diff < 0 else 0
                                model.addConstr(theta['ap'][station][train_1][train_2] == fixed_theta)
                            elif train_2 in arr_trains[station] and train_1 != train_2:
                                # 到到 aa
                                time_diff = pre_train_table[train_1.traNo][station]['arr'] - \
                                            pre_train_table[train_2.traNo][station]['arr']
                                fixed_theta = 1 if time_diff < 0 else 0
                                model.addConstr(theta['aa'][station][train_1][train_2] == fixed_theta)

    # 防止站间越行约束
    for sttn in station_list:
        for trn1 in dep_trains[sttn]:
            for trn2 in dep_trains[sttn]:
                if trn1 != trn2:
                    nextSttn1 = str(int(sttn) + int(2 * trn1.up - 1))  # trn1的下一站
                    nextSttn2 = str(int(sttn) + int(2 * trn2.up - 1))  # trn2的下一站

                    if nextSttn1 in trn1.staList and nextSttn2 in trn2.staList and nextSttn1 == nextSttn2:
                        if trn1.up == 1:
                            sec_time_1 = sec_times[trn1.speed][(sttn, nextSttn1)]
                            sec_time_2 = sec_times[trn2.speed][(sttn, nextSttn2)]
                        elif trn1.up == 0:
                            sec_time_1 = sec_times[trn1.speed][(nextSttn1, sttn)]
                            sec_time_2 = sec_times[trn2.speed][(nextSttn2, sttn)]
                        # duration time of train 1 and train 2
                        drtn_time_1 = sec_time_1 + trn1.delta[(sttn, nextSttn1)]
                        drtn_time_2 = sec_time_2 + trn2.delta[(sttn, nextSttn2)]

                        # trn1: da, trn2: da
                        if trn1.linePlan[nextSttn1] == 1 and trn2.linePlan[nextSttn2] == 1:
                            model.addConstr(theta['dd'][sttn][trn1][trn2] == theta['aa'][nextSttn1][trn1][trn2])
                            if drtn_time_1 + aa_speed[trn2.speed][nextSttn1] > drtn_time_2 + dd_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['dd'][sttn][trn1][trn2])
                            elif drtn_time_1 + aa_speed[trn2.speed][nextSttn1] <= drtn_time_2 + \
                                    dd_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['aa'][nextSttn1][trn1][trn2])
                        # trn1: da, trn2: dp
                        elif trn1.linePlan[nextSttn1] == 1 and trn2.linePlan[nextSttn2] in (0, -1):
                            model.addConstr(theta['dd'][sttn][trn1][trn2] == theta['ap'][nextSttn1][trn1][trn2])
                            if drtn_time_1 + ap_speed[trn2.speed][nextSttn1] > drtn_time_2 + dd_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['dd'][sttn][trn1][trn2])
                            elif drtn_time_1 + ap_speed[trn2.speed][nextSttn1] <= drtn_time_2 + \
                                    dd_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['ap'][nextSttn1][trn1][trn2])
                        # trn1: dp, trn2: da
                        elif trn1.linePlan[nextSttn1] in (0, -1) and trn2.linePlan[nextSttn2] == 1:
                            model.addConstr(theta['dd'][sttn][trn1][trn2] == theta['pa'][nextSttn1][trn1][trn2])
                            if drtn_time_1 + pa_speed[trn2.speed][nextSttn1] > drtn_time_2 + dd_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['dd'][sttn][trn1][trn2])
                            elif drtn_time_1 + pa_speed[trn2.speed][nextSttn1] <= drtn_time_2 + \
                                    dd_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['pa'][nextSttn1][trn1][trn2])
                        # trn1: dp, trn2: dp
                        elif trn1.linePlan[nextSttn1] in (0, -1) and trn2.linePlan[nextSttn2] in (0, -1):
                            model.addConstr(theta['dd'][sttn][trn1][trn2] == theta['pp'][nextSttn1][trn1][trn2])
                            if drtn_time_1 + pp_speed[trn2.speed][nextSttn1] > drtn_time_2 + dd_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['dd'][sttn][trn1][trn2])
                            elif drtn_time_1 + pp_speed[trn2.speed][nextSttn1] <= drtn_time_2 + \
                                    dd_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['pp'][nextSttn1][trn1][trn2])

            for trn2 in pass_trains[sttn]:
                if trn1 != trn2:
                    nextSttn1 = str(int(sttn) + int(2 * trn1.up - 1))  # trn1的下一站
                    nextSttn2 = str(int(sttn) + int(2 * trn2.up - 1))  # trn2的下一站

                    if nextSttn1 in trn1.staList and nextSttn2 in trn2.staList and nextSttn1 == nextSttn2:
                        if trn1.up == 1:
                            sec_time_1 = sec_times[trn1.speed][(sttn, nextSttn1)]
                            sec_time_2 = sec_times[trn2.speed][(sttn, nextSttn2)]
                        elif trn1.up == 0:
                            sec_time_1 = sec_times[trn1.speed][(nextSttn1, sttn)]
                            sec_time_2 = sec_times[trn2.speed][(nextSttn2, sttn)]
                        # duration time of train 1 and train 2
                        drtn_time_1 = sec_time_1 + trn1.delta[(sttn, nextSttn1)]
                        drtn_time_2 = sec_time_2 + trn2.delta[(sttn, nextSttn2)]

                        # trn1: da, trn2: pa
                        if trn1.linePlan[nextSttn1] == 1 and trn2.linePlan[nextSttn2] == 1:
                            model.addConstr(theta['dp'][sttn][trn1][trn2] == theta['aa'][nextSttn1][trn1][trn2])
                            if drtn_time_1 + aa_speed[trn2.speed][nextSttn1] > drtn_time_2 + dp_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['dp'][sttn][trn1][trn2])
                            elif drtn_time_1 + aa_speed[trn2.speed][nextSttn1] <= drtn_time_2 + \
                                    dp_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['aa'][nextSttn1][trn1][trn2])
                        # trn1: da, trn2: pp
                        elif trn1.linePlan[nextSttn1] == 1 and trn2.linePlan[nextSttn2] in (0, -1):
                            model.addConstr(theta['dp'][sttn][trn1][trn2] == theta['ap'][nextSttn1][trn1][trn2])
                            if drtn_time_1 + ap_speed[trn2.speed][nextSttn1] > drtn_time_2 + dp_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['dp'][sttn][trn1][trn2])
                            elif drtn_time_1 + ap_speed[trn2.speed][nextSttn1] <= drtn_time_2 + \
                                    dp_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['ap'][nextSttn1][trn1][trn2])
                        # trn1: dp, trn2: pa
                        elif trn1.linePlan[nextSttn1] in (0, -1) and trn2.linePlan[nextSttn2] == 1:
                            model.addConstr(theta['dp'][sttn][trn1][trn2] == theta['pa'][nextSttn1][trn1][trn2])
                            if drtn_time_1 + pa_speed[trn2.speed][nextSttn1] > drtn_time_2 + dp_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['dp'][sttn][trn1][trn2])
                            elif drtn_time_1 + pa_speed[trn2.speed][nextSttn1] <= drtn_time_2 + \
                                    dp_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['pa'][nextSttn1][trn1][trn2])
                        # trn1: dp, trn2: pp
                        elif trn1.linePlan[nextSttn1] in (0, -1) and trn2.linePlan[nextSttn2] in (0, -1):
                            model.addConstr(theta['dp'][sttn][trn1][trn2] == theta['pp'][nextSttn1][trn1][trn2])
                            if drtn_time_1 + pp_speed[trn2.speed][nextSttn1] > drtn_time_2 + dp_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['dp'][sttn][trn1][trn2])
                            elif drtn_time_1 + pp_speed[trn2.speed][nextSttn1] <= drtn_time_2 + \
                                    dp_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['pp'][nextSttn1][trn1][trn2])

        for trn1 in pass_trains[sttn]:
            for trn2 in dep_trains[sttn]:
                if trn1 != trn2:
                    nextSttn1 = str(int(sttn) + int(2 * trn1.up - 1))  # trn1的下一站
                    nextSttn2 = str(int(sttn) + int(2 * trn2.up - 1))  # trn2的下一站
                    if nextSttn1 in trn1.staList and nextSttn2 in trn2.staList and nextSttn1 == nextSttn2:
                        if trn1.up == 1:
                            sec_time_1 = sec_times[trn1.speed][(sttn, nextSttn1)]
                            sec_time_2 = sec_times[trn2.speed][(sttn, nextSttn2)]
                        elif trn1.up == 0:
                            sec_time_1 = sec_times[trn1.speed][(nextSttn1, sttn)]
                            sec_time_2 = sec_times[trn2.speed][(nextSttn2, sttn)]
                        # duration time of train 1 and train 2
                        drtn_time_1 = sec_time_1 + trn1.delta[(sttn, nextSttn1)]
                        drtn_time_2 = sec_time_2 + trn2.delta[(sttn, nextSttn2)]

                        # trn1: pa, trn2: da
                        if trn1.linePlan[nextSttn1] == 1 and trn2.linePlan[nextSttn2] == 1:
                            model.addConstr(theta['pd'][sttn][trn1][trn2] == theta['aa'][nextSttn1][trn1][trn2])
                            if drtn_time_1 + aa_speed[trn2.speed][nextSttn1] > drtn_time_2 + pd_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['pd'][sttn][trn1][trn2])
                            elif drtn_time_1 + aa_speed[trn2.speed][nextSttn1] <= drtn_time_2 + \
                                    pd_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['aa'][nextSttn1][trn1][trn2])
                        # trn1: pa, trn2: dp
                        elif trn1.linePlan[nextSttn1] == 1 and trn2.linePlan[nextSttn2] in (0, -1):
                            model.addConstr(theta['pd'][sttn][trn1][trn2] == theta['ap'][nextSttn1][trn1][trn2])
                            if drtn_time_1 + ap_speed[trn2.speed][nextSttn1] > drtn_time_2 + pd_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['pd'][sttn][trn1][trn2])
                            elif drtn_time_1 + ap_speed[trn2.speed][nextSttn1] <= drtn_time_2 + \
                                    pd_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['ap'][nextSttn1][trn1][trn2])
                        # trn1: pp, trn2: da
                        elif trn1.linePlan[nextSttn1] in (0, -1) and trn2.linePlan[nextSttn2] == 1:
                            model.addConstr(theta['pd'][sttn][trn1][trn2] == theta['pa'][nextSttn1][trn1][trn2])
                            if drtn_time_1 + pa_speed[trn2.speed][nextSttn1] > drtn_time_2 + pd_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['pd'][sttn][trn1][trn2])
                            elif drtn_time_1 + pa_speed[trn2.speed][nextSttn1] <= drtn_time_2 + \
                                    pd_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['pa'][nextSttn1][trn1][trn2])
                        # trn1: pp, trn2: dp
                        elif trn1.linePlan[nextSttn1] in (0, -1) and trn2.linePlan[nextSttn2] in (0, -1):
                            model.addConstr(theta['pd'][sttn][trn1][trn2] == theta['pp'][nextSttn1][trn1][trn2])
                            if drtn_time_1 + pp_speed[trn2.speed][nextSttn1] > drtn_time_2 + pd_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['pd'][sttn][trn1][trn2])
                            elif drtn_time_1 + pp_speed[trn2.speed][nextSttn1] <= drtn_time_2 + \
                                    pd_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['pp'][nextSttn1][trn1][trn2])

            for trn2 in pass_trains[sttn]:
                if trn1 != trn2:
                    nextSttn1 = str(int(sttn) + int(2 * trn1.up - 1))  # trn1的下一站
                    nextSttn2 = str(int(sttn) + int(2 * trn2.up - 1))  # trn2的下一站
                    if nextSttn1 in trn1.staList and nextSttn2 in trn2.staList and nextSttn1 == nextSttn2:
                        if trn1.up == 1:
                            sec_time_1 = sec_times[trn1.speed][(sttn, nextSttn1)]
                            sec_time_2 = sec_times[trn2.speed][(sttn, nextSttn2)]
                        elif trn1.up == 0:
                            sec_time_1 = sec_times[trn1.speed][(nextSttn1, sttn)]
                            sec_time_2 = sec_times[trn2.speed][(nextSttn2, sttn)]
                        # duration time of train 1 and train 2
                        drtn_time_1 = sec_time_1 + trn1.delta[(sttn, nextSttn1)]
                        drtn_time_2 = sec_time_2 + trn2.delta[(sttn, nextSttn2)]

                        # trn1: pa, trn2: pa
                        if trn1.linePlan[nextSttn1] == 1 and trn2.linePlan[nextSttn2] == 1:
                            model.addConstr(theta['pp'][sttn][trn1][trn2] == theta['aa'][nextSttn1][trn1][trn2])
                            if drtn_time_1 + aa_speed[trn2.speed][nextSttn1] > drtn_time_2 + pp_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['pp'][sttn][trn1][trn2])
                            elif drtn_time_1 + aa_speed[trn2.speed][nextSttn1] <= drtn_time_2 + \
                                    pp_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['aa'][nextSttn1][trn1][trn2])
                        # trn1: pa, trn2: pp
                        elif trn1.linePlan[nextSttn1] == 1 and trn2.linePlan[nextSttn2] in (0, -1):
                            model.addConstr(theta['pp'][sttn][trn1][trn2] == theta['ap'][nextSttn1][trn1][trn2])
                            if drtn_time_1 + ap_speed[trn2.speed][nextSttn1] > drtn_time_2 + pp_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['pp'][sttn][trn1][trn2])
                            elif drtn_time_1 + ap_speed[trn2.speed][nextSttn1] <= drtn_time_2 + \
                                    pp_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['ap'][nextSttn1][trn1][trn2])
                        # trn1: pp, trn2: pa
                        elif trn1.linePlan[nextSttn1] in (0, -1) and trn2.linePlan[nextSttn2] == 1:
                            model.addConstr(theta['pp'][sttn][trn1][trn2] == theta['pa'][nextSttn1][trn1][trn2])
                            if drtn_time_1 + pa_speed[trn2.speed][nextSttn1] > drtn_time_2 + pp_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['pp'][sttn][trn1][trn2])
                            elif drtn_time_1 + pa_speed[trn2.speed][nextSttn1] <= drtn_time_2 + \
                                    pp_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['pa'][nextSttn1][trn1][trn2])
                        # trn1: pp, trn2: pp
                        elif trn1.linePlan[nextSttn1] in (0, -1) and trn2.linePlan[nextSttn2] in (0, -1):
                            model.addConstr(theta['pp'][sttn][trn1][trn2] == theta['pp'][nextSttn1][trn1][trn2])
                            if drtn_time_1 + pp_speed[trn2.speed][nextSttn1] > drtn_time_2 + pp_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['pp'][sttn][trn1][trn2])
                            elif drtn_time_1 + pp_speed[trn2.speed][nextSttn1] <= drtn_time_2 + \
                                    pp_speed[trn2.speed][sttn]:
                                model.remove(safe_constrs['pp'][nextSttn1][trn1][trn2])

    return model, theta
