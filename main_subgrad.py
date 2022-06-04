import datetime
import os
import sys
import time
from typing import *

import pandas as pd
from gurobipy import GRB

import util_output as uo
from train import *
from jsp.main import main_jsp

# logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)
logFormatter = logging.Formatter("%(asctime)s: %(message)s")
logger = logging.getLogger("railway")
logger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

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


def read_train(path, size=10, time_span=1080):
    df = pd.read_excel(path, dtype={"车次ID": int, "车次": str}, engine='openpyxl')
    df = df.rename(columns={k: str(k) for k in df.columns})
    df = df.iloc[:size, :]
    train_series = df.apply(lambda row: parse_row_to_train(row, time_span), axis=1).dropna()
    global train_list
    train_list = [tr for tr in train_series.to_list() if tr.v_staList.__len__() > 2]  # sanity check


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
                    yvc_multiplier[v_station, t][c] = sum(
                        multiplier[v_station, t - dt]["aa"] for dt in range(min(interval, t + 1))) \
                                                      + multiplier[v_station, t]["ap"] \
                                                      + sum(
                        multiplier[v_station, t - dt]["pa"] for dt in range(1, min(interval2, t + 1)))
                elif c == "p":
                    interval2 = max(safe_int["ap"][station, 350], safe_int["ap"][station, 300])
                    yvc_multiplier[v_station, t][c] = sum(
                        multiplier[v_station, t - dt]["pp"] for dt in range(min(interval, t + 1))) \
                                                      + multiplier[v_station, t]["pa"] \
                                                      + sum(
                        multiplier[v_station, t - dt]["ap"] for dt in range(1, min(interval2, t + 1)))
                else:
                    yvc_multiplier[v_station, t][c] = 0
        elif v_station.endswith("_"):
            for c in ["s", "p"]:
                interval = max(safe_int[c + c][station, 350], safe_int[c + c][station, 300])
                if c == "s":
                    interval2 = max(safe_int["ps"][station, 350], safe_int["ps"][station, 300])
                    yvc_multiplier[v_station, t][c] = sum(
                        multiplier[v_station, t - dt]["ss"] for dt in range(min(interval, t + 1))) \
                                                      + multiplier[v_station, t]["sp"] \
                                                      + sum(
                        multiplier[v_station, t - dt]["ps"] for dt in range(1, min(interval2, t + 1)))
                elif c == "p":
                    interval2 = max(safe_int["sp"][station, 350], safe_int["sp"][station, 300])
                    yvc_multiplier[v_station, t][c] = multiplier[v_station, t]["ps"] \
                                                      + sum(
                        multiplier[v_station, t - dt]["sp"] for dt in range(1, min(interval2, t + 1)))


def update_subgradient_dict(params_sys, node_occupy_dict, option="lagrange", z_vars=None):
    subgradient_dict = {}
    primal_feas = {}
    for node, multi in multiplier.items():
        subgradient_dict[node] = {}
        primal_feas[node] = {}
        for cc, mu in multi.items():
            station = node[0].replace("_", "")
            interval = max(safe_int[cc][station, 350], safe_int[cc][station, 300])
            if option != "pdhg_alm":
                raise ValueError("only support pdhg_alm")
            _Ax = node_occupy_dict[node][cc[0]] \
                  + sum(
                node_occupy_dict[node[0], node[1] + t][cc[1]]
                for t in range(1, min(interval, params_sys.time_span - node[1]))
            )
            _z = z_vars[node][cc] = min(1, multiplier[node][cc] / params_subgrad.alpha + _Ax)

            subgradient_dict[node][cc] = _Ax - _z
            primal_feas[node][cc] = max(_Ax - 1, 0)

    return subgradient_dict, primal_feas


def update_primal_infeasibility(primal_feas):
    subgs = [d for subg_dict in primal_feas.values() for d in subg_dict.values()]
    return np.linalg.norm(subgs, 1), np.linalg.norm(subgs, 2), np.max(subgs)


def update_dual_excess(multiplier):
    multipliers = [mu for multi in multiplier.values() for mu in multi.values()]
    return np.linalg.norm(multipliers, 1), np.linalg.norm(multipliers, 2), np.max(multipliers)


def update_node_occupy_dict(node_occupy_dict, train, option="lagrange", beta=1):
    if option in ("lagrange", "pdhg_alm"):
        for node in train.opt_path_LR[1:-1]:  # z_{j v}
            node_occupy_dict[node][train.v_sta_type[node[0]]] += 1
    elif option == "pdhg":
        for node in train.opt_path_LR[1:-1]:
            node_occupy_dict[node][train.v_sta_type[node[0]]] += 1 + beta * (1 - (node in train.opt_path_LR_prev_dict))
    elif option == "dual_prox":
        for (i, j), v in train.opt_path_LR.items():
            if j[0] == '_t':
                continue
            node_occupy_dict[j][train.v_sta_type[j[0]]] += v
    else:
        raise ValueError(f"option {option} is not supported")


def update_step_size(params_subgrad, method="polyak", alpha=None):
    global subgradient_dict, UB, LB
    if method == "simple":
        if params_subgrad.iter < 20:
            params_subgrad.alpha = 0.5 / (params_subgrad.iter + 1)
        else:
            logger.info("switch to constant step size")
            params_subgrad.alpha = 0.5 / 20
    elif method == "polyak":
        subg_norm = np.linalg.norm([v for d in subgradient_dict.values() for v in d.values()]) ** 2
        params_subgrad.alpha = params_subgrad.kappa * (
                params_subgrad.ub_arr[-1] - params_subgrad.lb_arr[-1]) / subg_norm
    elif method == "constant":
        params_subgrad.alpha = alpha
        params_subgrad.gamma = 1.005
    # increase
    elif method == "power":
        # params_subgrad.alpha *= 0.998
        params_subgrad.gamma *= 1.005
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
    for idx, train in enumerate(sorted(train_list, key=lambda tr: tr.opt_cost_LR, reverse=True)):
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
            x_sol = {k: 1 if np.isreal(v) else v.x for k, v in x_var.items()}
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
                if _path[-2][-1] <= params_sys.time_span:
                    jsp_count += 1
            # if jsp is better.
            if jsp_count > count:
                count = jsp_count
                for tr in train_list:
                    tr.feasible_path = tr.feasible_path_jsp
                feasible_provider = 'jsp'

    elif method == "seq":
        pass
    else:
        raise TypeError(f"method has no wrong type: {method}")

    return feasible_provider, path_cost_feasible, count, not_feasible_trains, buffer


def init_multipliers(params_sys, multiplier, v_station_list, option="lagrange", z_vars=None):
    for sta in v_station_list:  # 循环车站
        for t in range(0, params_sys.time_span):  # 循环时刻t
            if sta not in ["s_", "_t"]:
                if sta.startswith("_"):  # arrival
                    multiplier[sta, t] = {"aa": 0, "ap": 0, "pa": 0, "pp": 0}
                elif sta.endswith("_"):  # departure
                    multiplier[sta, t] = {"ss": 0, "sp": 0, "ps": 0}
                else:
                    raise TypeError(f"virtual station has the wrong name: {sta}")
    if option == "pdhg_alm":
        assert z_vars is not None
        for sta in v_station_list:  # 循环车站
            for t in range(0, params_sys.time_span):  # 循环时刻t
                if sta not in ["s_", "_t"]:
                    if sta.startswith("_"):  # arrival
                        z_vars[sta, t] = {"aa": 1, "ap": 1, "pa": 1, "pp": 1}
                    elif sta.endswith("_"):  # departure
                        z_vars[sta, t] = {"ss": 1, "sp": 1, "ps": 1}
                    else:
                        raise TypeError(f"virtual station has the wrong name: {sta}")
    return dict(multiplier)


def setup(params_sys, params_subgrad):
    # create result-folder.
    subdir_result = params_sys.subdir_result = datetime.datetime.now().strftime('%y%m%d-%H%M')
    fdir_result = params_sys.fdir_result = f"result/{subdir_result}"
    os.makedirs(fdir_result, exist_ok=True)

    fileHandler = logging.FileHandler("{0}/{1}.log".format(fdir_result, "out"))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    params_sys.log_problem_size(logger)

    read_station('raw_data/1-station.xlsx', params_sys.station_size)
    read_station_stop_start_addtime('raw_data/2-station-extra.xlsx')
    read_section('raw_data/3-section-time.xlsx')
    read_dwell_time('raw_data/4-dwell-time.xlsx')
    if params_sys.up:
        read_train('raw_data/7-lineplan-up.xlsx', params_sys.train_size, params_sys.time_span)
    else:
        read_train('raw_data/6-lineplan-down.xlsx', params_sys.train_size, params_sys.time_span)
    read_intervals('raw_data/5-safe-intervals.xlsx')
    logger.info("reading finish")

    for tr in train_list:
        tr.create_subgraph(sec_times_all[tr.speed], params_sys.time_span)
    logger.info("graph initialization finish")

    return params_sys, params_subgrad


if __name__ == '__main__':
    params_sys = SysParams()
    params_subgrad = SubgradParam()

    params_sys.parse_environ()
    params_subgrad.parse_environ()

    setup(params_sys)

    for tr in train_list:
        tr.create_subgraph(sec_times_all[tr.speed], params_sys.time_span)

    alpha_list = [0.05, 0.2, 0.5, 1]
    print(alpha_list)

    # an underestimate
    hat_norm = 20

    # gamma_list = np.logspace(1, 4, num=5)
    # for alpha, gamma in product(alpha_list, gamma_list):
    for alpha in alpha_list:
        logger.info("Start")
        alpha = alpha  # np.sqrt(tr.subgraph.ecount())
        gamma = 2  # # np.sqrt(tr.subgraph.ecount())
        for tr in train_list:
            tr.reset()
        params_subgrad.reset()
        params_subgrad.alpha = alpha
        params_subgrad.gamma = gamma
        logger.info(f"alpha={alpha}, gamma={gamma}")
        print(params_subgrad.__dict__)
        '''
        initialization
        '''
        multiplier = init_multipliers(params_sys, multiplier, v_station_list, params_subgrad.dual_method, z_vars)

        '''
        Lagrangian relaxation approach
        '''

        interval = 1

        ######################
        # best primals
        ######################
        max_number = 0
        minGap = 0.1
        time_start = time.time()
        jsp_init = False
        path_cost_feasible = 1e6
        buffer = []
        while params_subgrad.iter < params_sys.iter_max:
            time_start_iter = time.time()
            # compile adjusted multiplier for each node
            #   from the original Lagrangian
            update_yvc_multiplier(multiplier)
            multiplier_norm_1, multiplier_norm_2, multiplier_norm_inf = update_dual_excess(multiplier)
            params_subgrad.multipliers[0].append(multiplier_norm_1)
            params_subgrad.multipliers[1].append(multiplier_norm_2)
            params_subgrad.multipliers[2].append(multiplier_norm_inf)
            # LR: train sub-problems solving
            path_cost_LR = 0
            subgradient_dict = {}
            node_occupy_dict = defaultdict(lambda: {"a": 0, "s": 0, "p": 0})

            for train in train_list:
                train.update_arc_multiplier(option=params_subgrad.dual_method, gamma=params_subgrad.gamma)
                train.save_prev_lr_path()
                train.opt_path_LR, train.opt_cost_LR, train.opt_cost_multiplier = train.shortest_path(
                    option='dual_prox')
                path_cost_LR += train.opt_cost_LR
                update_node_occupy_dict(node_occupy_dict, train, "dual_prox", params_subgrad.beta)
            subgradient_dict, primal_feas = update_subgradient_dict(
                params_sys, node_occupy_dict, params_subgrad.dual_method,
                z_vars
            )
            norm_1, norm_2, norm_inf = update_primal_infeasibility(primal_feas)
            params_subgrad.norms[0].append(norm_1)
            params_subgrad.norms[1].append(norm_2)
            params_subgrad.norms[2].append(norm_inf)

            # check feasibility
            # check_dual_feasibility(subgradient_dict, multiplier, train_list, LB)
            lb = path_cost_LR - sum(v for d in multiplier.values() for v in d.values())
            params_subgrad.update_bound(lb)
            params_subgrad.update_incumbent(path_cost_feasible)
            params_subgrad.update_gap()

            # update lagrangian multipliers
            update_step_size(params_subgrad, method='power', alpha=params_subgrad.alpha)
            update_lagrangian_multipliers(params_subgrad.alpha, subgradient_dict)

            params_subgrad.iter += 1

            if params_subgrad.iter % interval == 0:
                time_end_iter = time.time()
                logger.info(
                    f"iter#: {params_subgrad.iter}, step:{params_subgrad.alpha:.3e}, {params_subgrad.gamma: .3e}, lb: [{params_subgrad.lb_arr[-1]:.3e}], \n"
                    f"||(Ax-b)+||_1: {norm_1:.2f}, ||(Ax-b)+||_2: {norm_2:.2f}, ||(Ax-b)+||_inf: {norm_inf}, \n"
                    f"||lamdba||_1: {multiplier_norm_1:.2f}, ||lamdba||_2: {multiplier_norm_2:.2f}, ||lamdba||_inf: {multiplier_norm_inf:.2f}, \n"
                )

        end_time = time.time()
        time_elapsed = end_time - start_time
        logger.info(f"# of iterations {params_subgrad.iter} in {time_elapsed: .3e} seconds")
        uo.plot_convergence(param_sys=params_sys, param_subgrad=params_subgrad)
