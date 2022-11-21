import re
import pandas as pd
from Train import Train


def parse_row_to_train(row, time_span=1080):
    ### FIXME
    import numpy as np
    in_sta = row['上下行']
    macro_time = np.random.randint(0, time_span)
    ### FIXME
    tr = Train(row['车次'], in_sta, macro_time)
    tr.preferred_time = row['偏好始发时间'] - 360  # minus 6h, translate to zero time
    tr.down = row['上下行']
    tr.standard = row['标杆车']
    tr.speed = row['速度']

    return tr


def read_train(path, size=10, time_span=1080, up=-1):
    df = pd.read_excel(path, dtype={"车次ID": int, "车次": str}, engine='openpyxl').reset_index()
    df = df.rename(columns={k: str(k) for k in df.columns})
    # df = df.iloc[:size, :]
    df = df.sample(n=min(size, df.shape[0]), random_state=1)
    if up != -1:
        df = df[df['上下行'] == 1 - up].reset_index()
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


def filter_meso_info(route_index, route_conflicts, routes, reachable_boundary_track_route_pairs, station_boundaries, tracks):
    route_index = {k: v for k, v in route_index.items() if k[0] in station_boundaries and k[1] in tracks}
    routes = [route for route in routes if route[1] in station_boundaries and route[2] in tracks]
    all_routes = set(route_index.values())
    route_conflicts = {k: set(v) & all_routes for k, v in route_conflicts.items() if k in all_routes}
    reachable_boundary_track_route_pairs = {k: v for k, v in reachable_boundary_track_route_pairs.items() if 'B8' in k or 'B9' in k}
    return route_index, route_conflicts, routes, reachable_boundary_track_route_pairs
