"""
module for input functions.
"""

import datetime
import os
import sys
import time
from typing import *

import pandas as pd

from Train import *

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


def setup(params_sys: SysParams):
    """
    - create result-folder.
    - read raw data files.

    Args:
        params_sys: SysParams
        data kept in global variables of this module
    Returns:

    """

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

    return params_sys
