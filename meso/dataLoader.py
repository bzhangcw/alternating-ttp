import re
import pandas as pd
import numpy as np
import time
from Train import Train
from typing import *
from util import *

######################################################
# macro info
######################################################
# output format
COLUMNS_CHS_ENG = {
    '车次': 'id',
    '站序': 'station#',
    '站名': 'station_name',
    '到点': 'str_time_arr',
    '发点': 'str_time_dep'
}

station_name_list = [
    '北京南', '廊坊', '京津线路所', '津沪线路所', '天津南',
    '沧州西', '德州东', '济南西', '崔马庄线路所', '泰安',
    '曲阜东', '滕州东', '枣庄', '徐州东', '宿州东',
    '蚌埠南', '定远', '滁州', '扬州线路所', '南京南',
    '秦淮河线路所', '镇江南', '丹阳北', '常州北', '无锡东',
    '苏州北', '昆山南', '黄渡线路所', '上海虹桥'
]
# initialize stations, sections, trains and train arcs
station_list = []
station_name_map = {}
v_station_list = []
miles = []
miles_up = []
train_list: List[Train] = []
start_time = time()
sec_times_all = {}
pass_station = {}
min_dwell_time = {}
max_dwell_time = {}
stop_addTime = {}
start_addTime = {}
# some conventions
START_PERIOD = 360
######################################################
# meso
######################################################
route_index = route_conflicts = routes = \
    reachable_boundary_track_route_pairs = \
    station_platform_dict = station_boundaries = tracks = None


def parse_row_to_train(row, time_span=1080):
    ### FIXME
    import numpy as np
    down = row['上下行']
    macro_time = np.random.randint(0, time_span)
    ### FIXME
    tr = Train(row['车次'], down, macro_time)
    tr.preferred_time = row['偏好始发时间'] - START_PERIOD  # minus 6h, translate to zero time
    tr.down = row['上下行']
    tr.standard = row['标杆车']
    tr.speed = row['速度']

    return tr


def read_train(path, size=10, time_span=1080, down=-1):
    df = pd.read_excel(path, dtype={"车次ID": int, "车次": str}, engine='openpyxl').reset_index()
    df = df.rename(columns={k: str(k) for k in df.columns})
    # df = df.iloc[:size, :]
    df = df.sample(n=min(size, df.shape[0]), random_state=1)
    if down != -1:
        df = df[df['上下行'] == down].reset_index()
    train_series = df.apply(lambda row: parse_row_to_train(row, time_span), axis=1).dropna()
    train_list = train_series.to_list()
    return train_list


def read_route_conflict_info(path):
    routes = []  # 路径
    route_conflicts = {}
    reachable_boundary_track_route_pairs = {}
    end_points = []
    route_index = {}
    with open(path, 'r', encoding='gbk') as f:
        for line in f.readlines():
            line_list = re.split("\t|\n|,| |-->|<|>", line)
            line_list = list(filter(lambda val: val != '', line_list))
            l = re.split(':', line_list[2])
            line_list[2] = l[0] + ':' + l[2]
            routes.append([line_list[0], line_list[1], line_list[2], l[0] + ':' + l[1] + ':' + l[2]])
            route_conflicts[line_list[0]] = line_list[3:]
            route_index[(line_list[1], line_list[2])] = line_list[0]
            end_points.extend([line_list[1], line_list[2]])
    end_points = list(set(end_points))
    station_boundaries = list(filter(lambda val: val[:2] == "站界", end_points))  # 站界
    tracks = list(filter(lambda val: val[:2] == "站线", end_points))  # 此处的“站线”即指“股道”，而不是进出站路径

    for boundary in station_boundaries:
        reachable_boundary_track_route_pairs[boundary] = {}
        for track in tracks:
            reachable_boundary_track_route_pairs[boundary][track] = [route[0] for route in routes if
                                                                     (route[1] == boundary and route[2] == track)]

    return route_index, route_conflicts, station_boundaries, tracks, routes, reachable_boundary_track_route_pairs


def filter_meso_info(route_index, route_conflicts, routes, reachable_boundary_track_route_pairs, station_boundaries,
                     tracks):
    route_index = {k: v for k, v in route_index.items() if k[0] in station_boundaries and k[1] in tracks}
    routes = [route for route in routes if route[1] in station_boundaries and route[2] in tracks]
    all_routes = set(route_index.values())
    route_conflicts = {k: set(v) & all_routes for k, v in route_conflicts.items() if k in all_routes}
    reachable_boundary_track_route_pairs = {k: v for k, v in reachable_boundary_track_route_pairs.items() if
                                            'B8' in k or 'B9' in k}
    return route_index, route_conflicts, routes, reachable_boundary_track_route_pairs


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


def setup(params_sys: SysParams):
    """
    - create result-folder.
    - read raw data files.

    Args:
        params_sys: SysParams
        data kept in global variables of this module
    Returns:

    """
    global train_list, route_index, route_conflicts, routes, reachable_boundary_track_route_pairs, \
        station_platform_dict, station_boundaries, tracks

    # macro info
    train_list = read_train(
        'raw_data/7-lineplan-all.xlsx', params_sys.train_size, params_sys.time_span,
        down=params_sys.down
    )
    read_station('raw_data/1-station.xlsx', params_sys.station_size)
    read_station_stop_start_addtime('raw_data/2-station-extra.xlsx')
    read_section('raw_data/3-section-time.xlsx')
    read_dwell_time('raw_data/4-dwell-time.xlsx')

    # meso info
    route_index, route_conflicts, _, _, routes, reachable_boundary_track_route_pairs = \
        read_route_conflict_info('raw_data/RouteConflictInfo.txt')
    station_boundaries = {'站界:B8', '站界:B9'}
    tracks = {'站线:8', '站线:9', '站线:10', '站线:11', '站线:XII', '站线:XIII', '站线:XIV', '站线:XV', '站线:16',
              '站线:17', '站线:18', '站线:19'}
    station_platform_dict = {'1': tracks}
    route_index, \
    route_conflicts, \
    routes, \
    reachable_boundary_track_route_pairs = filter_meso_info(route_index,
                                                            route_conflicts,
                                                            routes,
                                                            reachable_boundary_track_route_pairs,
                                                            station_boundaries,
                                                            tracks)


def read_timetable_csv(fpath, st=None, station_name_map=None):
    """

    read standard timetable csv

    Args:
        fpath: file path
        st:    start-time of the timetable.
            if not defined, infer from the data.

    Returns:
    """
    df = pd.read_csv(fpath).rename(
        columns=COLUMNS_CHS_ENG
    ).assign(
        station_id=lambda df: df['station_name'].apply(station_name_map.get),
        station_i=lambda df: df['station_id'].apply(lambda x: f"_{x}"),
        station_o=lambda df: df['station_id'].apply(lambda x: f"{x}_"),
        time_arr=lambda df: pd.to_datetime(df['str_time_arr']).apply(lambda x: x.hour * 60 + x.minute - st),
        time_dep=lambda df: pd.to_datetime(df['str_time_dep']).apply(lambda x: x.hour * 60 + x.minute - st)
    )
    train_paths = df.groupby('id').apply(
        lambda grp: sorted(
            list(zip(grp['station_i'], grp['time_arr'])) + list(zip(grp['station_o'], grp['time_dep'])),
            key=lambda x: x[-1]
        )
    )
    return df, train_paths


def read_local_macro_solution(fp, station_name='北京南'):
    df, train_paths = read_timetable_csv(fp, st=START_PERIOD, station_name_map=station_name_map)
    df = df.astype({"id": str})
    dfs = df.query(
        f"station_name == '{station_name}'"
    ).set_index("id")

    df_last = df.groupby("id")['station#'].max()

    for tr in train_list:
        tr_st_seq = dfs["station#"].get(tr.traNo, None)
        if tr_st_seq is None:
            tr.macro_time = None
            tr.bool_in_station = None
            continue
        if tr_st_seq == 0:
            tr.bool_in_station = 0
        elif tr_st_seq == df_last[tr.traNo]:
            tr.bool_in_station = 1
        else:
            tr.bool_in_station = -1
        try:
            tr.macro_time = dfs['time_dep'][tr.traNo]
        except:
            tr.macro_time = None
    return dfs
