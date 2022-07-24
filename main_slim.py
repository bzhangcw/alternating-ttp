import datetime
import logging
import os
import time
from collections import defaultdict
from typing import *

import jsp.model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import util_output as uo
from Train import *
from jsp.main import main_jsp, station_name_list
from jsp.util import get_train_table
from jsp.dataLoader import write_train_table
from gurobipy import GRB, Var
from PathPoolMananger import PathPoolManager
from joblib import Parallel, delayed, parallel_backend

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)
logger = logging.getLogger("railway")
logger.setLevel(logging.INFO)

'''
initialize stations, sections, trains and train arcs
'''
station_list = []  # 实际车站列表
station_name_map = {}  # 实际车站对应名
v_station_list = []  # 时空网车站列表，车站一分为二 # 源节点s, 终结点t
miles = []
miles_up = []
train_list: List[Train] = []
start_time = time.time()
sec_times_all = {}
pass_station = {}
min_dwell_time = {}
max_dwell_time = {}
stop_addTime = {}
start_addTime = {}


def read_intervals(path):
    global safe_int, safe_int_df
    df = pd.read_excel(
        path, engine='openpyxl'
    ).rename(columns={
        "车站": 'station',
        "到到安全间隔": 'aa',
        "到通安全间隔": 'ap',
        "发发安全间隔": 'ss',
        "发通安全间隔": 'sp',
        "通到安全间隔": 'pa',
        "通发安全间隔": 'ps',
        "通通安全间隔": 'pp',
    }).astype({'station': str}).query(
        "station in @station_list"
    ).set_index(['station', 'speed'])

    safe_int = df.to_dict()
    safe_int_df = df


def read_station(path, size):
    global miles, v_station_list, station_list, station_name_map
    df = pd.read_excel(
        path,
        engine='openpyxl',
        dtype={"里程": np.float64,
               "站名": str,
               "站全名": str
               }
    ).sort_values('里程')
    df = df.iloc[:size, :]
    miles = df['里程'].values
    station_list = df['站名'].to_list()
    v_station_list.append('s_')
    for sta in station_list:
        if station_list.index(sta) != 0:  # 不为首站，有到达
            v_station_list.append('_' + sta)
        if station_list.index(sta) != len(station_list) - 1:
            v_station_list.append(sta + '_')  # 不为尾站，又出发
    v_station_list.append('_t')
    station_name_map = {**df.set_index('站名')['站全名'].to_dict(), **df.set_index('站全名')['站名'].to_dict()}


def read_section(path):
    df = pd.read_excel(path, engine='openpyxl').assign(
        interval=lambda dfs: dfs['区间名'].apply(lambda x: tuple(x.split("-")))
    ).set_index("interval")
    global sec_times_all
    sec_300 = df[300].to_dict()
    sec_350 = df[350].to_dict()
    sec_times_all = {
        300: {**sec_300, **{(k[-1], k[0]): v for k, v in sec_300.items()}},
        350: {**sec_350, **{(k[-1], k[0]): v for k, v in sec_350.items()}},
    }
    return 1


def parse_row_to_train(row, time_span=1080):
    tr = Train(row['车次ID'], 0, time_span, backend=1)
    tr.preferred_time = row['偏好始发时间'] - 360  # minus 6h, translate to zero time
    tr.up = row['上下行']
    tr.standard = row['标杆车']
    tr.speed = row['速度']
    tr.linePlan = {k: row[k] for k in station_list}
    if all(value == 0 for value in tr.linePlan.values()):
        return None
    tr.init_traStaList(station_list)
    tr.stop_addTime = stop_addTime[tr.speed]
    tr.start_addTime = start_addTime[tr.speed]
    tr.min_dwellTime = min_dwell_time
    tr.max_dwellTime = max_dwell_time
    tr.pass_station = pass_station

    return tr


def read_train(path, size=10, time_span=1080):
    df = pd.read_excel(path, dtype={"车次ID": int, "车次": str}, engine='openpyxl')
    df = df.rename(columns={k: str(k) for k in df.columns})
    df = df.iloc[:size, :]
    train_series = df.apply(lambda row: parse_row_to_train(row, time_span), axis=1).dropna()
    global train_list
    train_list = train_series.to_list()


def read_dwell_time(path):
    global pass_station, min_dwell_time, max_dwell_time
    df = pd.read_excel(path, dtype={"最小停站时间": int, "最大停站时间": int, "station": str}, engine='openpyxl')
    pass_station = df[(df["最小停站时间"] == 0) & (df["最大停站时间"] == 0)]["station"].to_dict()
    min_dwell_time = df.set_index("station")["最小停站时间"].to_dict()
    max_dwell_time = df.set_index("station")["最大停站时间"].to_dict()


def read_station_stop_start_addtime(path):
    global stop_addTime, start_addTime
    df = pd.read_excel(path, dtype={"上行停车附加时分": int, "上行起车附加时分": int, "车站名称": str}, engine='openpyxl')
    df_350 = df[df["列车速度"] == 350].set_index("车站名称")
    df_300 = df[df["列车速度"] == 300].set_index("车站名称")
    stop_addTime[350] = df_350["上行停车附加时分"].to_dict()
    stop_addTime[300] = df_300["上行停车附加时分"].to_dict()
    start_addTime[350] = df_350["上行起车附加时分"].to_dict()
    start_addTime[300] = df_300["上行起车附加时分"].to_dict()


def get_train_timetable_from_result():
    for train in train_list:
        if not train.is_best_feasible:
            continue
        for node in train.best_path:
            train.timetable[node[0]] = node[1]


def update_lagrangian_multipliers(alpha, subgradient_dict):
    for node, multi in multiplier.items():
        for cc in multi.keys():
            multiplier[node][cc] += alpha * subgradient_dict[node][cc]
            multiplier[node][cc] = max(multiplier[node][cc], 0)


def update_yvc_multiplier(multiplier):
    for v_station, t in multiplier.keys():
        if v_station in ["s_", "_t"]:
            continue
        station = v_station.replace("_", "")
        if v_station.startswith("_"):
            for c in ["a", "p"]:
                interval = max(safe_int[c + c][station, 350], safe_int[c + c][station, 300])
                if c == "a":
                    interval2 = max(safe_int["pa"][station, 350], safe_int["pa"][station, 300])
                    yvc_multiplier[v_station, t][c] = sum(multiplier[v_station, t - dt]["aa"] for dt in range(min(interval, t + 1))) \
                                                      + multiplier[v_station, t]["ap"] \
                                                      + sum(multiplier[v_station, t - dt]["pa"] for dt in range(1, min(interval2, t + 1)))
                elif c == "p":
                    interval2 = max(safe_int["ap"][station, 350], safe_int["ap"][station, 300])
                    yvc_multiplier[v_station, t][c] = sum(multiplier[v_station, t - dt]["pp"] for dt in range(min(interval, t + 1))) \
                                                      + multiplier[v_station, t]["pa"] \
                                                      + sum(multiplier[v_station, t - dt]["ap"] for dt in range(1, min(interval2, t + 1)))
                else:
                    yvc_multiplier[v_station, t][c] = 0
        elif v_station.endswith("_"):
            for c in ["s", "p"]:
                interval = max(safe_int[c + c][station, 350], safe_int[c + c][station, 300])
                if c == "s":
                    interval2 = max(safe_int["ps"][station, 350], safe_int["ps"][station, 300])
                    yvc_multiplier[v_station, t][c] = sum(multiplier[v_station, t - dt]["ss"] for dt in range(min(interval, t + 1))) \
                                                      + multiplier[v_station, t]["sp"] \
                                                      + sum(multiplier[v_station, t - dt]["ps"] for dt in range(1, min(interval2, t + 1)))
                elif c == "p":
                    interval2 = max(safe_int["sp"][station, 350], safe_int["sp"][station, 300])
                    yvc_multiplier[v_station, t][c] = multiplier[v_station, t]["ps"] \
                                                      + sum(multiplier[v_station, t - dt]["sp"] for dt in range(1, min(interval2, t + 1)))


def update_subgradient_dict(node_occupy_dict):
    subgradient_dict = {}
    for node, multi in multiplier.items():
        subgradient_dict[node] = {}
        for cc, mu in multi.items():
            station = node[0].replace("_", "")
            interval = max(safe_int[cc][station, 350], safe_int[cc][station, 300])
            subgradient_dict[node][cc] = node_occupy_dict[node][cc[0]] \
                                         + sum(node_occupy_dict[node[0], node[1] + t][cc[1]] for t in range(1, min(interval, time_span - node[1]))) \
                                         - 1
    return subgradient_dict


def update_node_occupy_dict(node_occupy_dict, train, option="lagrange", alpha=1):
    if option == "lagrange":
        for node in train.opt_path_LR[1:-1]:  # z_{j v}
            node_occupy_dict[node][train.v_sta_type[node[0]]] += 1
    elif option == "pdhg":
        for node in train.opt_path_LR[1:-1]:
            node_occupy_dict[node][train.v_sta_type[node[0]]] += 1 + alpha * (1 - (node in train.opt_path_LR_prev_dict))
    else:
        raise ValueError(f"option {option} is not supported")


def update_step_size(params_subgrad, method="polyak"):
    global subgradient_dict, UB, LB
    if method == "simple":
        if params_subgrad.iter < 20:
            params_subgrad.alpha = 0.5 / (params_subgrad.iter + 1)
        else:
            logger.info("switch to constant step size")
            params_subgrad.alpha = 0.5 / 20
    elif method == "polyak":
        subg_norm = np.linalg.norm([v for d in subgradient_dict.values() for v in d.values()]) ** 2
        params_subgrad.alpha = params_subgrad.kappa * (params_subgrad.ub_arr[-1] - params_subgrad.lb_arr[-1]) / subg_norm
    else:
        raise ValueError(f"undefined method {method}")


def get_occupied_nodes(train):
    _centers = train.feasible_path[1:-1]
    _nodes = {}
    for vs, t in _centers:
        _nodes[vs, t] = train.v_sta_type[vs]

    return _nodes


def get_occupied_arcs_and_clique(feasible_path):
    _this_occupied_arcs = {
        (i, j): (t1, t2) for (i, t1), (j, t2) in
        zip(feasible_path[1:-2], feasible_path[2:-1])
        if i[0] != j[1]  # if not same station
    }
    # edges starting later yet arriving earlier
    _this_new_incompatible_arcs = {
        ((i, st), (j, et))
        for (i, j), (t1, t2) in _this_occupied_arcs.items()
        for st in range(t1 + 1, t2) for et in range(st, t2)
    }
    # edges starting earlier yet arriving later
    _this_new_incompatible_arcs.update(
        ((i, st), (j, et))
        for (i, j), (t1, t2) in _this_occupied_arcs.items()
        for st in range(t1 - 10, t1)
        for et in range(t2, t2 + 10)
    )

    return _this_occupied_arcs, _this_new_incompatible_arcs


def check_dual_feasibility(subgradient_dict, multiplier, train_list, LB):
    lambda_mul_subg = 0
    for node in multiplier.keys():
        lambda_mul_subg += multiplier[node] * subgradient_dict[node]

    dual_cost = LB[-1]
    dual_cost_2 = 0
    for train in train_list:
        dual_cost_2 += nx.path_weight(train.subgraph, train.opt_path_LR, weight="weight")

    assert dual_cost == dual_cost_2 + lambda_mul_subg


def primal_heuristic(train_list, safe_int, jsp_init, buffer, method="jsp", params_sys=None, *args, **kwargs):
    """
    produce path: iterable of (station, time)

    Args:
        method: the name of primal feasible algorithm.
    """

    __unused = args, kwargs
    path_cost_feasible = 0
    occupied_nodes = {}
    occupied_arcs = defaultdict(lambda: set())
    incompatible_arcs = set()
    count = 0
    not_feasible_trains = []
    feasible_provider = 'seq'
    for idx, train in enumerate(sorted(train_list, key=lambda tr: (tr.standard, tr.opt_cost_LR), reverse=True)):
        train.update_primal_graph(occupied_nodes, occupied_arcs, incompatible_arcs, safe_int)

        train.feasible_path, train.feasible_cost = train.shortest_path_primal()
        if not train.is_feasible:
            path_cost_feasible += train.max_edge_weight * (
                    len(train.staList) - 1)
            not_feasible_trains.append(train.traNo)
            continue
        else:
            count += 1
        _this_occupied_nodes = get_occupied_nodes(train)
        occupied_nodes.update(_this_occupied_nodes)
        _this_occupied_arcs, _this_new_incompatible_arcs = get_occupied_arcs_and_clique(train.feasible_path)
        for k, v in _this_occupied_arcs.items():
            occupied_arcs[k].add(v)
        incompatible_arcs.update(_this_new_incompatible_arcs)
        path_cost_feasible += train.feasible_cost
    logger.info(f"seq maximum cardinality: {count}")
    if method == "jsp":
        # path_method = "dual"  # use primal path
        path_method = "primal"
        max_iter = 30
        if not jsp_init:
            # @update, add d_var, a_var
            model, theta, x_var, d_var, a_var, *_ = main_jsp(params_sys)
            buffer.extend([model, theta, x_var, d_var, a_var])
            buffer = tuple(buffer)
        else:
            model, theta, x_var, d_var, a_var, *_ = buffer
        train_order, overtaking_dict = from_train_path_to_train_order(train_list, method=path_method)
        if path_method == "primal":
            fix_x_constrs = fix_train_at_station(model, x_var, [trn for trn in train_list if trn.is_feasible])
        fix_train_order_at_station(model, train_order, safe_int, overtaking_dict, theta)
        model.setParam(GRB.Param.TimeLimit, 600)  # 找到可行解就停止求解并返回
        model.optimize()

        if model.status == GRB.INFEASIBLE:
            # todo, fix this
            if params_sys.DBG:
                for idx, train in enumerate(train_list):
                    train.best_path = train.feasible_path
                    train.is_best_feasible = train.is_feasible
                    if not train.is_best_feasible:
                        continue
                    for node in train.best_path:
                        train.timetable[node[0]] = node[1]
                uo.plot_timetables_h5(train_list, miles, station_list, param_sys=params_sys,
                                      param_subgrad=params_subgrad)
                # IIS_resolve(model, max_iter)
                model.computeIIS()
                model.write("ttp.ilp")
            pass

        model.remove(getConstrByPrefix(model, "headway_fix"))  # remove added headway fix constraints
        if path_method == "primal":
            model.remove(fix_x_constrs)
        if model.status not in [GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED] and model.SolCount > 0:
            jsp_count = 0
            x_sol = {k: v.x if isinstance(v, Var) else v for k, v in x_var.items()}
            jsp_not_feasible_trains = [k for k, v in x_sol.items() if v == 0]
            d_sol = {tr: {s: v.x for s, v in vl.items()} for tr, vl in d_var.items() if x_sol[tr] == 1}
            a_sol = {tr: {s: v.x for s, v in vl.items()} for tr, vl in a_var.items() if x_sol[tr] == 1}

            for tr in train_list:
                if not tr in d_sol:
                    continue
                _arr = a_sol[tr]
                _dep = d_sol[tr]

                # tr.create_feasible_by_time(a_sol[tridx], d_sol[tridx])
                def _gen_path(_arr, _dep):
                    yield 's_', -1
                    for s, t in _arr.items():
                        yield f"_{s}", t - 360
                        yield f"{s}_", _dep[s] - 360
                    yield '_t', -1

                _path = list(_gen_path(_arr, _dep))
                tr.feasible_path_jsp = _path
                if (len(_path) > 2) and _path[-2][-1] <= params_sys.time_span:
                    # double check
                    jsp_count += 1

            # if jsp is better.
            if jsp_count > count:
                count = jsp_count
                not_feasible_trains = jsp_not_feasible_trains
                feasible_provider = 'jsp'
                for tr in train_list:
                    if x_sol[tr] == 1:
                        tr.feasible_path = tr.feasible_path_jsp
                        tr.is_feasible = True
                    else:
                        tr.feasible_path = []
                        tr.is_feasible = False
            logger.info(f"assert info: {len(jsp_not_feasible_trains) + jsp_count}")
            logger.info(f"jsp maximum cardinality: {jsp_count}")
            train_table = get_train_table(train_list, d_var, a_var)
            direction = "up" if params_sys.up == 1 else "down"
            write_train_table(params_sys.fdir_result + f"/lr_{direction}.csv", train_table, station_name_list, direction)
    elif method == "seq":
        pass
    elif method == "path":
        path_pool_manager: PathPoolManager = kwargs.get("path_pool_manager", None)
        assert path_pool_manager is not None
        for train in train_list:
            if train.is_feasible:
                path_pool_manager.add_path(train.traNo, train.feasible_path)
            path_pool_manager.add_path(train.traNo, train.opt_path_LR)
        try:
            logger.info(f"path pooling graph size: |V|: {path_pool_manager.graph.vcount()}, |E|: {path_pool_manager.graph.ecount()}")
        except AttributeError:
            logger.info(f"path pooling graph size: |V|: {path_pool_manager.graph.number_of_nodes()}, |E|: {path_pool_manager.graph.number_of_edges()}")
        new_solution = path_pool_manager.largest_independent_vertex_sets()
        path_count = len(new_solution)
        logger.info("path pooling maximal cardinality: " + str(path_count))
        if path_count >= count:
            feasible_provider = 'path'
            feasible_train_paths = {}
            for path_id in new_solution:
                train_id, path = path_pool_manager.inverse_path_ids[path_id]
                feasible_train_paths[train_id] = path

            feasible_trains = set(feasible_train_paths.keys())
            for train in train_list:
                if train.traNo in feasible_trains:
                    train.feasible_path = feasible_train_paths[train.traNo]
                    train.is_feasible = True
                else:
                    train.is_feasible = False
            count = path_count
    else:
        raise TypeError(f"method has no wrong type: {method}")

    return feasible_provider, path_cost_feasible, count, not_feasible_trains, buffer


def init_multipliers(multiplier, v_station_list):
    for sta in v_station_list:  # 循环车站
        for t in range(0, time_span):  # 循环时刻t
            if sta not in ["s_", "_t"]:
                if sta.startswith("_"):  # arrival
                    multiplier[sta, t] = {"aa": 0, "ap": 0, "pa": 0, "pp": 0}
                elif sta.endswith("_"):  # departure
                    multiplier[sta, t] = {"ss": 0, "sp": 0, "ps": 0}
                else:
                    raise TypeError(f"virtual station has the wrong name: {sta}")
    return dict(multiplier)


def process_updata():
    global miles_up
    miles_up = np.array(list(reversed(miles)))
    miles_up = - miles_up + miles_up.max()


def create_tr_subgraph(tr):
    tr.create_subgraph(sec_times_all[tr.speed], time_span, params_sys.fix_preferred_time)
    return tr


if __name__ == '__main__':

    params_sys = SysParams()
    params_subgrad = SubgradParam()

    params_sys.parse_environ()
    params_subgrad.parse_environ()

    station_size = params_sys.station_size
    train_size = params_sys.train_size
    time_span = params_sys.time_span
    iter_max = params_sys.iter_max

    dual = params_subgrad.dual_method
    primal = params_subgrad.primal_heuristic_method

    # create result-folder.
    subdir_result = params_sys.subdir_result = datetime.datetime.now().strftime('%y%m%d-%H%M')
    fdir_result = params_sys.fdir_result = f"result/{subdir_result}"
    os.makedirs(fdir_result, exist_ok=True)
    logger.info(f"size: {params_sys.__dict__}")
    read_station('raw_data/1-station.xlsx', station_size)
    read_station_stop_start_addtime('raw_data/2-station-extra.xlsx')
    read_section('raw_data/3-section-time.xlsx')
    read_dwell_time('raw_data/4-dwell-time.xlsx')
    if params_sys.up:
        read_train('raw_data/7-lineplan-up.xlsx', train_size, time_span)
    else:
        read_train('raw_data/6-lineplan-down.xlsx', train_size, time_span)
    read_intervals('raw_data/5-safe-intervals.xlsx')

    '''
    initialization
    '''
    logger.info("step 1: reading finish")
    multiplier = init_multipliers(multiplier, v_station_list)
    logger.info(f"maximum estimate of active nodes {gc.vc}")

    with parallel_backend("loky", n_jobs=os.cpu_count()):
        start_time = time.time()
        train_list = Parallel()(delayed(create_tr_subgraph)(tr) for tr in train_list)
    logger.info(f"parallel graph creation time: {time.time() - start_time:.2f}s")

    logger.info(f"step 2: graph building finish")
    logger.info(f"actual train size {len(train_list)}")

    interval = 1
    interval_primal = 1

    ######################
    # best primals
    ######################
    max_number = 0
    minGap = 0.1
    time_start = time.time()
    jsp_init = False
    path_cost_feasible = 1e6
    buffer = []
    path_pool_manager = PathPoolManager(train_list, safe_int, params_sys.up)
    while params_subgrad.gap > minGap and params_subgrad.iter < iter_max:
        time_start_iter = time.time()
        # compile adjusted multiplier for each node
        #   from the original Lagrangian
        logger.info("dual subproblems begins")
        update_yvc_multiplier(multiplier)
        # LR: train sub-problems solving
        path_cost_LR = 0
        subgradient_dict = {}
        node_occupy_dict = defaultdict(lambda: {"a": 0, "s": 0, "p": 0})
        for train in train_list:
            train.update_arc_multiplier(option=params_subgrad.dual_method, gamma=params_subgrad.gamma)
            train.save_prev_lr_path()
            train.opt_path_LR, train.opt_cost_LR, train.opt_cost_multiplier = train.shortest_path()
            path_cost_LR += train.opt_cost_LR
            update_node_occupy_dict(node_occupy_dict, train)
        subgradient_dict = update_subgradient_dict(node_occupy_dict)
        logger.info(f"dual subproblems finished")

        if params_subgrad.iter % interval_primal != 0:  # or iter == iter_max - 1:
            logger.info(f"no primal stage this iter: {params_subgrad.iter}")
        else:
            logger.info("primal stage begins")
            # feasible solutions
            feasible_provider, path_cost_feasible, count, not_feasible_trains, buffer = primal_heuristic(
                train_list,
                safe_int,
                jsp_init,
                buffer,
                method=params_subgrad.primal_heuristic_method,
                params_sys=params_sys,
                path_pool_manager=path_pool_manager
            )
            params_subgrad.feasible_provider = feasible_provider
            jsp_init = True
            logger.info(f"maximum cardinality of feasible paths: {count}")
            logger.info(f"infeasible trains: {not_feasible_trains}")
            logger.info("primal stage finished")
            direction = "up" if params_sys.up == 1 else "down"
            train_table = get_train_table_from_feas_path(train_list)
            write_train_table_feas_path(params_sys.fdir_result + f"/lr_{'up' if params_sys.up else 'down'}.csv", train_table, station_name_list, direction)

            # update best primal solution
            if count > max_number:
                logger.info("best primal solution updated")
                params_subgrad.max_number = max_number = count
                for idx, train in enumerate(train_list):
                    train.timetable = {}
                    train.best_path = train.feasible_path
                    train.is_best_feasible = train.is_feasible
                    if not train.is_best_feasible:
                        continue
                    for node in train.best_path:
                        train.timetable[node[0]] = node[1]
                uo.plot_timetables_h5(train_list, miles, station_list, param_sys=params_sys,
                                      param_subgrad=params_subgrad)

        # check feasibility
        # check_dual_feasibility(subgradient_dict, multiplier, train_list, LB)
        lb = path_cost_LR - sum(v for d in multiplier.values() for v in d.values())
        params_subgrad.update_bound(lb)
        params_subgrad.update_incumbent(path_cost_feasible)
        params_subgrad.update_gap()

        # update lagrangian multipliers
        update_step_size(params_subgrad, method='polyak')
        update_lagrangian_multipliers(params_subgrad.alpha, subgradient_dict)

        params_subgrad.iter += 1

        if params_subgrad.iter % interval == 0:
            logger.info(f"subgrad params: {params_subgrad.__dict__}")
            time_end_iter = time.time()
            logger.info(
                f"time:{time_end_iter - time_start:.3e}/{time_end_iter - time_start_iter:.2f}, iter#: {params_subgrad.iter:.2e}, step:{params_subgrad.alpha:.4f}, gap: {params_subgrad.gap:.2%} @[{params_subgrad.lb_arr[-1]:.3e} - {params_subgrad.ub_arr[-1]:.3e}]"
            )

    end_time = time.time()
    time_elapsed = end_time - start_time
    logger.info(f"# of iterations {params_subgrad.iter} in {time_elapsed: .3e} seconds")
    uo.plot_convergence(param_sys=params_sys, param_subgrad=params_subgrad)
