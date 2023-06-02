import datetime
import os
import time
from typing import *

from joblib import Parallel, delayed, parallel_backend
import logging
import pandas as pd
import util_output as uo
from PathPoolMananger import PathPoolManager
from Train import *

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)
logger = logging.getLogger("railway")
logger.setLevel(logging.INFO)

station_name_list = [
    "北京南",
    "廊坊",
    "京津线路所",
    "津沪线路所",
    "天津南",
    "沧州西",
    "德州东",
    "济南西",
    "崔马庄线路所",
    "泰安",
    "曲阜东",
    "滕州东",
    "枣庄",
    "徐州东",
    "宿州东",
    "蚌埠南",
    "定远",
    "滁州",
    "扬州线路所",
    "南京南",
    "秦淮河线路所",
    "镇江南",
    "丹阳北",
    "常州北",
    "无锡东",
    "苏州北",
    "昆山南",
    "黄渡线路所",
    "上海虹桥",
]
"""
initialize stations, sections, trains and train arcs
"""
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
    df = (
        pd.read_excel(path, engine="openpyxl")
        .rename(
            columns={
                "车站": "station",
                "到到安全间隔": "aa",
                "到通安全间隔": "ap",
                "发发安全间隔": "ss",
                "发通安全间隔": "sp",
                "通到安全间隔": "pa",
                "通发安全间隔": "ps",
                "通通安全间隔": "pp",
            }
        )
        .astype({"station": str})
        .query("station in @station_list")
        .set_index(["station", "speed"])
    )

    safe_int = df.to_dict()
    safe_int_df = df


def read_station(path, size):
    global miles, v_station_list, station_list, station_name_map
    df = pd.read_excel(
        path, engine="openpyxl", dtype={"里程": np.float64, "站名": str, "站全名": str}
    ).sort_values("里程")
    df = df.iloc[:size, :]
    miles = df["里程"].values
    station_list = df["站名"].to_list()
    v_station_list.append("s_")
    for sta in station_list:
        if station_list.index(sta) != 0:  # 不为首站，有到达
            v_station_list.append("_" + sta)
        if station_list.index(sta) != len(station_list) - 1:
            v_station_list.append(sta + "_")  # 不为尾站，又出发
    v_station_list.append("_t")
    station_name_map = {
        **df.set_index("站名")["站全名"].to_dict(),
        **df.set_index("站全名")["站名"].to_dict(),
    }


def read_section(path):
    df = (
        pd.read_excel(path, engine="openpyxl")
        .assign(interval=lambda dfs: dfs["区间名"].apply(lambda x: tuple(x.split("-"))))
        .set_index("interval")
    )
    global sec_times_all
    sec_300 = df[300].to_dict()
    sec_350 = df[350].to_dict()
    sec_times_all = {
        300: {**sec_300, **{(k[-1], k[0]): v for k, v in sec_300.items()}},
        350: {**sec_350, **{(k[-1], k[0]): v for k, v in sec_350.items()}},
    }
    return 1


def parse_row_to_train(row, time_span=1080):
    tr = Train(row["车次ID"], 0, time_span, backend=1)
    tr.preferred_time = row["偏好始发时间"] - 360  # minus 6h, translate to zero time
    tr.up = row["上下行"]
    tr.standard = row["标杆车"]
    tr.speed = row["速度"]
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
    df = pd.read_excel(path, dtype={"车次ID": int, "车次": str}, engine="openpyxl")
    df = df.rename(columns={k: str(k) for k in df.columns})
    df = df.iloc[:size, :]
    train_series = df.apply(
        lambda row: parse_row_to_train(row, time_span), axis=1
    ).dropna()
    global train_list
    train_list = train_series.to_list()


def read_dwell_time(path):
    global pass_station, min_dwell_time, max_dwell_time
    df = pd.read_excel(
        path, dtype={"最小停站时间": int, "最大停站时间": int, "station": str}, engine="openpyxl"
    )
    pass_station = df[(df["最小停站时间"] == 0) & (df["最大停站时间"] == 0)]["station"].to_dict()
    min_dwell_time = df.set_index("station")["最小停站时间"].to_dict()
    max_dwell_time = df.set_index("station")["最大停站时间"].to_dict()


def read_station_stop_start_addtime(path):
    global stop_addTime, start_addTime
    df = pd.read_excel(
        path, dtype={"上行停车附加时分": int, "上行起车附加时分": int, "车站名称": str}, engine="openpyxl"
    )
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
                interval = max(
                    safe_int[c + c][station, 350], safe_int[c + c][station, 300]
                )
                if c == "a":
                    interval2 = max(
                        safe_int["pa"][station, 350], safe_int["pa"][station, 300]
                    )
                    yvc_multiplier[v_station, t][c] = (
                            sum(
                                multiplier[v_station, t - dt]["aa"]
                                for dt in range(min(interval, t + 1))
                            )
                            + multiplier[v_station, t]["ap"]
                            + sum(
                        multiplier[v_station, t - dt]["pa"]
                        for dt in range(1, min(interval2, t + 1))
                    )
                    )
                elif c == "p":
                    interval2 = max(
                        safe_int["ap"][station, 350], safe_int["ap"][station, 300]
                    )
                    yvc_multiplier[v_station, t][c] = (
                            sum(
                                multiplier[v_station, t - dt]["pp"]
                                for dt in range(min(interval, t + 1))
                            )
                            + multiplier[v_station, t]["pa"]
                            + sum(
                        multiplier[v_station, t - dt]["ap"]
                        for dt in range(1, min(interval2, t + 1))
                    )
                    )
                else:
                    yvc_multiplier[v_station, t][c] = 0
        elif v_station.endswith("_"):
            for c in ["s", "p"]:
                interval = max(
                    safe_int[c + c][station, 350], safe_int[c + c][station, 300]
                )
                if c == "s":
                    interval2 = max(
                        safe_int["ps"][station, 350], safe_int["ps"][station, 300]
                    )
                    yvc_multiplier[v_station, t][c] = (
                            sum(
                                multiplier[v_station, t - dt]["ss"]
                                for dt in range(min(interval, t + 1))
                            )
                            + multiplier[v_station, t]["sp"]
                            + sum(
                        multiplier[v_station, t - dt]["ps"]
                        for dt in range(1, min(interval2, t + 1))
                    )
                    )
                elif c == "p":
                    interval2 = max(
                        safe_int["sp"][station, 350], safe_int["sp"][station, 300]
                    )
                    yvc_multiplier[v_station, t][c] = multiplier[v_station, t][
                                                          "ps"
                                                      ] + sum(
                        multiplier[v_station, t - dt]["sp"]
                        for dt in range(1, min(interval2, t + 1))
                    )


def update_subgradient_dict(node_occupy_dict):
    subgradient_dict = {}
    for node, multi in multiplier.items():
        subgradient_dict[node] = {}
        for cc, mu in multi.items():
            station = node[0].replace("_", "")
            interval = max(safe_int[cc][station, 350], safe_int[cc][station, 300])
            subgradient_dict[node][cc] = (
                    node_occupy_dict[node][cc[0]]
                    + sum(
                node_occupy_dict[node[0], node[1] + t][cc[1]]
                for t in range(1, min(interval, time_span - node[1]))
            )
                    - 1
            )
    return subgradient_dict


def update_node_occupy_dict(node_occupy_dict, train, option="lagrange"):
    """
    Args:
        node_occupy_dict:
        train:
        option:

    Returns:

    """
    if option == "lagrange":
        for node in train.opt_path_LR[1:-1]:  # z_{j v}
            node_occupy_dict[node][train.v_sta_type[node[0]]] += 1
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
        subg_norm = (
                np.linalg.norm([v for d in subgradient_dict.values() for v in d.values()])
                ** 2
        )
        params_subgrad.alpha = (
                params_subgrad.kappa
                * (params_subgrad.ub_arr[-1] - params_subgrad.lb_arr[-1])
                / subg_norm
        )
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
        (i, j): (t1, t2)
        for (i, t1), (j, t2) in zip(feasible_path[1:-2], feasible_path[2:-1])
        if i[0] != j[1]  # if not same station
    }
    # edges starting later yet arriving earlier
    _this_new_incompatible_arcs = {
        ((i, st), (j, et))
        for (i, j), (t1, t2) in _this_occupied_arcs.items()
        for st in range(t1 + 1, t2)
        for et in range(st, t2)
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
        dual_cost_2 += nx.path_weight(
            train.subgraph, train.opt_path_LR, weight="weight"
        )

    assert dual_cost == dual_cost_2 + lambda_mul_subg


def primal_heuristic(
        train_list,
        safe_int,
        jsp_init,
        buffer,
        method="jsp",
        params_sys=None,
        target='count',
        *args,
        **kwargs,
):
    """
    produce path: iterable of (station, time)

    Args:
        method: the name of primal feasible algorithm.
    """

    __unused = args, kwargs
    if method == "seq":
        feasible_provider = "seq"
        iter_max = 1
    elif method == "seq_iter":
        feasible_provider = "seq_iter"
        iter_max = 10
    elif method == "jsp":
        raise ValueError("primal method JSP is deprecated")
    else:  # path still needs seq or seq_iter
        feasible_provider = "seq_iter"
        iter_max = 10

    best_count = count = -1
    best_obj = -1e6
    new_train_order = sorted(
        train_list, key=lambda tr: (tr.standard, tr.opt_cost_LR), reverse=True
    )

    for j in range(iter_max):
        path_cost_feasible = 0
        occupied_nodes = {}
        occupied_arcs = defaultdict(lambda: set())
        incompatible_arcs = set()
        count = 0
        not_feasible_trains = []
        train_order = new_train_order

        for idx, train in enumerate(train_order):
            train.update_primal_graph(
                occupied_nodes, occupied_arcs, incompatible_arcs, safe_int
            )
            # the heuristic only works for 1, but we may summarize using 0
            train.feasible_path, train.feasible_cost = train.shortest_path_primal(
                objtype=1
            )
            if not train.is_feasible:
                # path_cost_feasible += train.max_edge_weight * (len(train.staList) - 1)
                not_feasible_trains.append((idx, train.traNo))
                continue
            else:
                count += 1
            _this_occupied_nodes = get_occupied_nodes(train)
            occupied_nodes.update(_this_occupied_nodes)
            (
                _this_occupied_arcs,
                _this_new_incompatible_arcs,
            ) = get_occupied_arcs_and_clique(train.feasible_path)
            for k, v in _this_occupied_arcs.items():
                occupied_arcs[k].add(v)
            incompatible_arcs.update(_this_new_incompatible_arcs)
            path_cost_feasible += train.feasible_cost

        if (count > best_count) and (target == 'count'):
            logger.debug("better feasible number: {}".format(count))
            best_count = count
            best_obj = path_cost_feasible
            if best_count == len(train_list):
                logger.info("all trains in the graph!")
                break
        elif (path_cost_feasible > best_obj) and (target == 'obj'):
            best_obj = path_cost_feasible
            best_count = count
        else:
            first_train_idx = not_feasible_trains[0][0]
            train_order.insert(0, train_order.pop(first_train_idx))
            logger.debug(
                "only feasible number: {}, incumbent: {}".format(count, best_count)
            )

    logger.debug(f"seq maximum cardinality: {count}")

    if method == "path":
        path_pool_manager: PathPoolManager = kwargs.get("path_pool_manager", None)
        assert path_pool_manager is not None
        for train in train_list:
            path_list = gen_more_path(train)
            for path in path_list:
                path_pool_manager.add_path(train.traNo, path)
            # logger.info(f"path pooling graph size: |V|: {path_pool_manager.graph.vcount()}, |E|: {path_pool_manager.graph.ecount()}")
        try:
            logger.debug(
                f"path pooling graph size: |V|: {path_pool_manager.graph.vcount()}, |E|: {path_pool_manager.graph.ecount()}"
            )
        except AttributeError:
            logger.debug(
                f"path pooling graph size: |V|: {path_pool_manager.graph.number_of_nodes()}, |E|: {path_pool_manager.graph.number_of_edges()}"
            )
        new_solution = path_pool_manager.largest_independent_vertex_sets()
        path_pool_manager.update_usage(new_solution)
        path_count = len(new_solution)
        logger.debug("path pooling maximal cardinality: " + str(path_count))
        feasible_train_paths = {}
        if path_count >= count:
            feasible_provider = "path"
            for path_id in new_solution:
                train_id, path = path_pool_manager.inverse_path_ids[path_id]
                feasible_train_paths[train_id] = tuple(path)

            feasible_trains = set(feasible_train_paths.keys())
            for train in train_list:
                if train.traNo in feasible_trains:
                    train.feasible_path = feasible_train_paths[train.traNo]
                    train.is_feasible = True
                else:
                    train.is_feasible = False
            count = path_count

    #################################################################################
    # SUMMARIZATION
    #################################################################################
    best_result = best_obj if params_sys.obj == 1 else -best_count
    return feasible_provider, best_result, best_count, not_feasible_trains, buffer


def gen_more_path(train):
    T_feas = 0
    path_list = []

    if train.is_feasible:
        for t in range(-T_feas, T_feas + 1):
            new_path = train.feasible_path.copy()
            valid = True
            if t != 0:
                for i in range(len(new_path) - 2):
                    v_sta, time = new_path[i + 1]
                    new_time = time + t
                    if new_time < 0 or new_time > train.right_time_bound[v_sta]:
                        valid = False
                        break
                    new_path[i + 1] = (v_sta, new_time)
            if valid:
                path_list.append(new_path)

        T_dual = T_feas
    else:
        T_dual = 2

    for t in range(-T_dual, T_dual + 1):
        new_path = train.opt_path_LR.copy()
        valid = True
        if t != 0:
            for i in range(len(new_path) - 2):
                v_sta, time = new_path[i + 1]
                new_time = time + t
                if new_time < 0 or new_time > train.right_time_bound[v_sta]:
                    valid = False
                    break
                new_path[i + 1] = (v_sta, new_time)
        if valid:
            path_list.append(new_path)

    return path_list


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
    miles_up = -miles_up + miles_up.max()


def create_tr_subgraph(tr):
    tr.create_subgraph(sec_times_all[tr.speed], time_span)
    return tr


if __name__ == "__main__":

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
    subdir_result = params_sys.subdir_result = datetime.datetime.now().strftime(
        "%y%m%d-%H%M"
    )
    fdir_result = params_sys.fdir_result = f"result/{subdir_result}"
    os.makedirs(fdir_result, exist_ok=True)
    logger.info(f"size: {params_sys.__dict__}")
    read_station("raw_data/1-station.xlsx", station_size)
    read_station_stop_start_addtime("raw_data/2-station-extra.xlsx")
    read_section("raw_data/3-section-time.xlsx")
    read_dwell_time("raw_data/4-dwell-time.xlsx")
    if params_sys.up:
        read_train("raw_data/7-lineplan-up.xlsx", train_size, time_span)
    else:
        read_train("raw_data/6-lineplan-down.xlsx", train_size, time_span)
    read_intervals("raw_data/5-safe-intervals.xlsx")

    """
    initialization
    """
    logger.info("step 1: reading finish")
    multiplier = init_multipliers(multiplier, v_station_list)
    logger.info(f"maximum estimate of active nodes {gc.vc}")

    # with parallel_backend("loky", n_jobs=1):
    #     start_time = time.time()
    #     train_list = Parallel()(delayed(create_tr_subgraph)(tr) for tr in train_list)
    for tr in train_list:
        try:
            tr.create_subgraph(sec_times_all[tr.speed], time_span)
        except:
            tr.is_valid = False
    train_list = [tr for tr in train_list if tr.is_valid]
    logger.info(f"parallel graph creation time: {time.time() - start_time:.2f}s")

    logger.info(f"step 2: graph building finish")
    logger.info(f"actual train size {len(train_list)}")

    interval = 1
    interval_primal = 1

    ######################
    # best primals
    ######################
    max_number = 0
    mip_gap = 0.1
    time_start = time.time()
    jsp_init = False
    path_cost_feasible = 1e6
    buffer = []
    path_pool_manager = PathPoolManager(train_list, safe_int, params_sys.up)
    params_subgrad.show()
    while params_subgrad.gap > mip_gap and params_subgrad.iter < iter_max:
        time_start_iter = time.time()
        # compile adjusted multiplier for each node
        #   from the original Lagrangian
        # logger.info("dual subproblems begins")
        update_yvc_multiplier(multiplier)
        # LR: train sub-problems solving
        path_cost_LR = 0
        subgradient_dict = {}
        node_occupy_dict = defaultdict(lambda: {"a": 0, "s": 0, "p": 0})
        for train in train_list:
            train.update_arc_multiplier(
                option=params_subgrad.dual_method,
                gamma=params_subgrad.gamma,
                param_sys=params_sys,
            )
            train.save_prev_lr_path()
            (
                train.opt_path_LR,
                train.opt_cost_LR,
                train.opt_cost_multiplier,
            ) = train.shortest_path(objtype=params_sys.obj)
            path_cost_LR += train.opt_cost_LR
            update_node_occupy_dict(node_occupy_dict, train)
        subgradient_dict = update_subgradient_dict(node_occupy_dict)
        # logger.info(f"dual subproblems finished")

        if params_subgrad.iter % interval_primal != 0:  # or iter == iter_max - 1:
            logger.info(f"no primal stage this iter: {params_subgrad.iter}")
        else:
            # logger.info("primal stage begins")
            # feasible solutions
            (
                feasible_provider,
                path_cost_feasible,
                count,
                not_feasible_trains,
                buffer,
            ) = primal_heuristic(
                train_list,
                safe_int,
                jsp_init,
                buffer,
                method=params_subgrad.primal_heuristic_method,
                target="obj",
                params_sys=params_sys,
                path_pool_manager=path_pool_manager,
            )
            params_subgrad.feasible_provider = feasible_provider
            jsp_init = True
            logger.info(f"maximum cardinality of feasible paths: {count}")
            # logger.info(f"infeasible trains: {not_feasible_trains}")
            # logger.info("primal stage finished")
            direction = "up" if params_sys.up == 1 else "down"

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
                uo.plot_timetables_h5(
                    train_list,
                    miles,
                    station_list,
                    param_sys=params_sys,
                    param_subgrad=params_subgrad,
                )

        # do evaluation
        # cost = train.evaluate(train.feasible_path)
        # check feasibility
        lb = path_cost_LR - sum(v for d in multiplier.values() for v in d.values())

        params_subgrad.update_bound(lb)
        params_subgrad.update_incumbent(-path_cost_feasible)
        params_subgrad.update_gap()

        # update lagrangian multipliers
        update_step_size(params_subgrad, method="polyak")
        update_lagrangian_multipliers(params_subgrad.alpha, subgradient_dict)

        params_subgrad.iter += 1

        if params_subgrad.iter % interval == 0:
            # logger.info(f"subgrad params: {params_subgrad.__dict__}")
            time_end_iter = time.time()
            logger.info(
                f"time:{time_end_iter - time_start:.3e}/{time_end_iter - time_start_iter:.2f}, iter#: {params_subgrad.iter:.2e}, step:{params_subgrad.alpha:.2e}, gap: {params_subgrad.gap:.2%} @[{max(params_subgrad.lb_arr):.3e} - {min(params_subgrad.ub_arr):.3e}]"
            )

    end_time = time.time()
    time_elapsed = end_time - start_time
    logger.info(f"# of iterations {params_subgrad.iter} in {time_elapsed: .3e} seconds")
    uo.plot_convergence(param_sys=params_sys, param_subgrad=params_subgrad)
