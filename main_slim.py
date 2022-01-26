import datetime
import logging
import os
import time
from collections import defaultdict
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Train import *
from jsp.main import main_jsp
from gurobipy import GRB

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)
logger = logging.getLogger("railway")
logger.setLevel(logging.INFO)

'''
initialize stations, sections, trains and train arcs
'''
station_list = []  # 实际车站列表
v_station_list = []  # 时空网车站列表，车站一分为二 # 源节点s, 终结点t
sec_times = {}  # total miles for stations
miles = []
train_list: List[Train] = []
start_time = time.time()
sec_times_all = {}
pass_station = {}
min_dwell_time = {}
max_dwell_time = {}
stop_addTime = {}
start_addTime = {}


def read_intervals(path):
    global safe_int
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


def read_station(path, size):
    global miles, v_station_list, station_list
    df = pd.read_excel(path, engine='openpyxl').sort_values('站名')
    df = df.iloc[:size, :]
    miles = df['里程'].values
    station_list = df['站名'].astype(str).to_list()
    v_station_list.append('_s')
    for sta in station_list:
        if station_list.index(sta) != 0:  # 不为首站，有到达
            v_station_list.append('_' + sta)
        if station_list.index(sta) != len(station_list) - 1:
            v_station_list.append(sta + '_')  # 不为尾站，又出发
    v_station_list.append('_t')


def read_section(path):
    df = pd.read_excel(path, engine='openpyxl').assign(
        interval=lambda dfs: dfs['区间名'].apply(lambda x: tuple(x.split("-")))
    ).set_index("interval")
    global sec_times, sec_times_all
    sec_times = df[350].to_dict()
    sec_times_all = {300: df[300].to_dict(), 350: df[350].to_dict()}


def parse_row_to_train(row):
    tr = Train(row['车次ID'], 0, time_span, backend=1)
    tr.preferred_time = row['偏好始发时间']
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


def read_train(path, size=10):
    df = pd.read_excel(path, dtype={"车次ID": str}, engine='openpyxl')
    df = df.rename(columns={k: str(k) for k in df.columns})
    df = df.iloc[:size, :]
    train_series = df.apply(lambda row: parse_row_to_train(row), axis=1).dropna()
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


def initialize_node_precedence():
    for station in v_station_list:
        for t in range(0, time_span):
            node_prec_map[station, t] = [
                (station, t - tau) for tau in
                range(min(t, eps))
            ]


def get_train_timetable_from_result():
    for train in train_list:
        if not train.is_best_feasible:
            continue
        for node in train.best_path:
            train.timetable[node[0]] = node[1]


def update_lagrangian_multipliers(alpha, subgradient_dict):
    for node in multiplier.keys():
        multiplier[node] += alpha * subgradient_dict[node]
        multiplier[node] = max(multiplier[node], 0)


def update_yv_multiplier():
    """
    multiplier for each node (v)
    """
    for (station, t), predecessors in node_prec_map.items():
        yv_multiplier[station, t] = sum(
            multiplier[p] for p in predecessors
        )
    logger.info("multiplier for 'v (nodes)' updated")


def update_subgradient_dict(subgradient_dict, node_occupy_dict):
    for node in multiplier.keys():
        subgradient_dict[node] = sum(
            node_occupy_dict[node[0], node[1] + t] for t in range(min(eps, time_span - node[1]))) - 1


def update_node_occupy_dict(node_occupy_dict, opt_path_LR):
    for node in opt_path_LR[1:-1]:  # z_{j v}
        node_occupy_dict[node] += 1


def update_step_size(params_subgrad, method="polyak"):
    global subgradient_dict, UB, LB
    if method == "simple":
        if params_subgrad.iter < 20:
            params_subgrad.alpha = 0.5 / (params_subgrad.iter + 1)
        else:
            logger.info("switch to constant step size")
            params_subgrad.alpha = 0.5 / 20
    elif method == "polyak":
        params_subgrad.alpha = params_subgrad.kappa * (UB[-1] - LB[-1]) / np.linalg.norm(
            list(subgradient_dict.values())) ** 2
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


def primal_heuristic(train_list, safe_int, jsp_init, buffer, method="seq"):
    # if method == "seq":
    # feasible solutions
    path_cost_feasible = 0
    occupied_nodes = {}
    occupied_arcs = defaultdict(lambda: set())
    incompatible_arcs = set()
    count = 0
    not_feasible_trains = []
    for idx, train in enumerate(sorted(train_list, key=lambda tr: - tr.opt_cost_LR)):
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

    if method == "jsp":
        max_iter = 30
        if not jsp_init:
            model, theta_aa, theta_ap, theta_pa, theta_pp, theta_dd, theta_dp, theta_pd = main_jsp()
            buffer.extend([model, theta_aa, theta_ap, theta_pa, theta_pp, theta_dd, theta_dp, theta_pd])
            buffer = tuple(buffer)
        else:
            model, theta_aa, theta_ap, theta_pa, theta_pp, theta_dd, theta_dp, theta_pd = buffer
        train_order, overtaking_dict = from_train_path_to_train_order(train_list)
        fix_train_order_at_station(model, train_order, safe_int, overtaking_dict, theta_aa, theta_ap, theta_pa, theta_pp, theta_dd, theta_dp, theta_pd)
        model.optimize()
        if model.status == GRB.INFEASIBLE:
            IIS_resolve(model, max_iter)
            model.write("ttp.ilp")
        model.remove(getConstrByPrefix(model, "headway_fix"))  # remove added headway fix constraints
        return path_cost_feasible, count, not_feasible_trains, buffer
    else:
        raise TypeError(f"method has no wrong type: {method}")


if __name__ == '__main__':

    station_size = int(os.environ.get('station_size', 29))
    train_size = int(os.environ.get('train_size', 80))
    time_span = int(os.environ.get('time_span', 500))
    iter_max = int(os.environ.get('iter_max', 100))
    primal_heuristic_method = "jsp"
    # primal_heuristic_method = "seq"
    logger.info(f"size: #train,#station,#timespan: {train_size, station_size, time_span}")
    read_station('raw_data/1-station.xlsx', station_size)
    read_station_stop_start_addtime('raw_data/2-station-extra.xlsx')
    read_section('raw_data/3-section-time.xlsx')
    read_dwell_time('raw_data/4-dwell-time.xlsx')
    read_train('raw_data/6-lineplan-down.xlsx', train_size)
    read_intervals('raw_data/5-safe-intervals.xlsx')

    '''
    initialization
    '''
    # create result-founder.
    subdir_result = datetime.datetime.now().strftime('%y%m%d-%H%M')
    fdir_result = f"result/{subdir_result}"
    os.makedirs(fdir_result, exist_ok=True)
    logger.info("reading finish")
    logger.info("step 1")
    initialize_node_precedence()
    logger.info(f"maximum estimate of active nodes {gc.vc}")

    for tr in train_list:
        tr.create_subgraph(sec_times_all[tr.speed], time_span)

    logger.info("step 2")
    logger.info("step 3")
    logger.info(f"actual train size {len(train_list)}")

    '''
    Lagrangian relaxation approach
    '''
    LB = []
    UB = []

    params_subgrad = SubgradParam()

    interval = 1
    interval_primal = 1

    ######################
    # best primals
    ######################
    max_number = 0
    minGap = 0.1
    time_start = time.time()
    jsp_init = False
    buffer = []
    while params_subgrad.gap > minGap and params_subgrad.iter < iter_max:
        time_start_iter = time.time()
        # compile adjusted multiplier for each node
        #   from the original Lagrangian
        logger.info("dual subproblems begins")
        update_yv_multiplier()

        # LR: train sub-problems solving
        path_cost_LR = 0
        subgradient_dict = {}
        node_occupy_dict = defaultdict(int)

        for train in train_list:
            train.update_arc_multiplier()
            train.opt_path_LR, train.opt_cost_LR = train.shortest_path()

            path_cost_LR += train.opt_cost_LR
            update_node_occupy_dict(node_occupy_dict, train.opt_path_LR)

        update_subgradient_dict(subgradient_dict, node_occupy_dict)

        lb = path_cost_LR - sum(multiplier.values())
        params_subgrad.update_bound(lb)
        LB.append(lb)
        logger.info(f"dual subproblems finished")

        if params_subgrad.iter % interval_primal != 0:  # or iter == iter_max - 1:
            logger.info(f"no primal stage this iter: {params_subgrad.iter}")
        else:
            logger.info("primal stage begins")
            # feasible solutions
            path_cost_feasible, count, not_feasible_trains, buffer = primal_heuristic(train_list, safe_int, jsp_init, buffer, method=primal_heuristic_method)
            jsp_init = True
            UB.append(path_cost_feasible)
            logger.info(f"maximum cardinality of feasible paths: {count}")
            logger.info(f"infeasible trains: {not_feasible_trains}")
            logger.info("primal stage finished")

            # update best primal solution
            if count > max_number:
                logger.info("best primal solution updated")
                max_number = count
                for idx, train in enumerate(train_list):
                    train.best_path = train.feasible_path
                    train.is_best_feasible = train.is_feasible

        # check feasibility
        # check_dual_feasibility(subgradient_dict, multiplier, train_list, LB)

        # update lagrangian multipliers
        update_step_size(params_subgrad, method='polyak')
        update_lagrangian_multipliers(params_subgrad.alpha, subgradient_dict)

        params_subgrad.iter += 1
        params_subgrad.gap = (UB[-1] - LB[-1]) / (abs(UB[-1]) + 1e-3)

        if params_subgrad.iter % interval == 0:
            logger.info(f"subgrad params: {params_subgrad.__dict__}")
            time_end_iter = time.time()
            logger.info(
                f"time:{time_end_iter - time_start:.3e}/{time_end_iter - time_start_iter:.2f}, iter#: {params_subgrad.iter:.2e}, step:{params_subgrad.alpha:.4f}, gap: {params_subgrad.gap:.2%} @[{lb:.3e} - {UB[-1]:.3e}]")

    get_train_timetable_from_result()

    '''
    draw timetable
    '''
    plt.rcParams['figure.figsize'] = (18.0, 9.0)
    # plt.rcParams["font.family"] = 'Times'
    plt.rcParams["font.size"] = 9
    fig = plt.figure(dpi=200)
    color_value = {
        '0': 'midnightblue',
        '1': 'mediumblue',
        '2': 'c',
        '3': 'orangered',
        '4': 'm',
        '5': 'fuchsia',
        '6': 'olive'
    }

    xlist = []
    ylist = []
    for i in range(len(train_list)):
        train = train_list[i]
        xlist = []
        ylist = []
        if not train.is_best_feasible:
            continue
        for sta_id in range(len(train.staList)):
            sta = train.staList[sta_id]
            if sta_id != 0:  # 不为首站, 有到达
                if "_" + sta in train.v_staList:
                    xlist.append(train.timetable["_" + sta])
                    ylist.append(miles[station_list.index(sta)])
            if sta_id != len(train.staList) - 1:  # 不为末站，有出发
                if sta + "_" in train.v_staList:
                    xlist.append(train.timetable[sta + "_"])
                    ylist.append(miles[station_list.index(sta)])
        plt.plot(xlist, ylist, color=color_value[str(i % 7)], linewidth=1.5)
        plt.text(xlist[0] + 0.8, ylist[0] + 4, train.traNo, ha='center', va='bottom',
                 color=color_value[str(i % 7)], weight='bold', family='Times', fontsize=9)

    plt.grid(True)  # show the grid
    plt.ylim(0, miles[-1])  # y range

    plt.xlim(0, time_span)  # x range
    sticks = 20
    plt.xticks(np.linspace(0, time_span, sticks))

    plt.yticks(miles, station_list, family='Times')
    plt.xlabel('Time (min)', family='Times new roman')
    plt.ylabel('Space (km)', family='Times new roman')
    plt.title(f"Best primal solution of # trains, station, periods: ({len(train_list)}, {station_size}, {time_span})\n"
              f"Number of trains {max_number}", fontdict={"weight": 500, "size": 20})

    plt.savefig(f"{fdir_result}/lagrangian-{train_size}.{station_size}.{time_span}.png", dpi=500)

    end_time = time.time()
    time_elapsed = end_time - start_time
    print(time_elapsed)

    plt.clf()
    ## plot the bound updates
    font_dic = {
        "family": "Times",
        "style": "oblique",
        "weight": "normal",
        "color": "green",
        "size": 20
    }
    logger.info(f"# of iterations {len(LB)}")
    x_cor = range(1, len(LB) + 1)
    plt.plot(x_cor, LB, label='LB')
    plt.plot(x_cor, UB, label='UB')
    plt.legend()
    plt.xlabel('Iteration', fontdict=font_dic)
    plt.ylabel('Bounds update', fontdict=font_dic)
    plt.title('LR: Bounds updates \n', fontsize=23)
    plt.savefig(f"{fdir_result}/lagrangian-{train_size}.{station_size}.{time_span}.convergence.png", dpi=500)
    # plt.show()
