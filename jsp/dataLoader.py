import pandas as pd

from jsp.Train import Train
from jsp.util import time2num, num2time


def read_station(path):
    """
    return miles, station_list

    miles: list of station miles
    station_list: list of stations in string format
    """
    df = pd.read_excel(path).sort_values('站名')
    miles = df['里程'].values
    station_list = df['站名'].astype(str).to_list()
    return miles, station_list


def read_section(path):
    """
    return sec_times, sec_times_all

    sec_times: dict of 350 section times, key is (station, station+1)
    sec_times_all: dict of all section times, both 300 and 350
    """
    df = pd.read_excel(path).assign(
        interval=lambda dfs: dfs['区间名'].apply(lambda x: tuple(x.split("-")))
    ).set_index("interval")

    sec_times = {}
    for speed in [300, 350]:
        sec_times[speed] = df[speed].to_dict()

    return sec_times


def parse_row_to_train(row, station_list, g, h, miles):
    tr = Train(int(row['车次ID']))
    tr.preferred_time = row['偏好始发时间']
    tr.up = row['上下行']
    tr.standard = row['标杆车']
    tr.speed = row['速度']
    if tr.up == 1:  # from 1 to 29
        tr.linePlan = {k: row[k] for k in station_list}
    elif tr.up == 0:  # from 29 to 1
        tr.linePlan = {k: row[k] for k in station_list[::-1]}
    tr.decode_line_plan(g, h, miles)
    return tr


def read_train(path, station_list, g, h, miles):
    """
    return train_list

    train_list: list of Train
    """
    df = pd.read_excel(path)
    df = df.rename(columns={k: str(k) for k in df.columns})
    train_series = df.apply(lambda row: parse_row_to_train(
        row, station_list, g, h, miles), axis=1)
    train_list = train_series.to_list()
    return train_list


def read_station_stops(path):
    """ 获取站内停车时长上下界 """
    df = pd.read_excel(path)
    df['station'] = df['station'].astype(str)
    df = df.set_index('station')
    wait_time_lb = dict(df['最小停站时间'])
    wait_time_ub = dict(df['最大停站时间'])
    return wait_time_lb, wait_time_ub


def read_station_extra(path):
    """
    return g, h

    g: dict[speed][station] 上行停车附加时分
    h: dict[speed][station] 上行起车附加时分
    """
    df = pd.read_excel(path)
    df['车站名称'] = df['车站名称'].astype(str)

    g = {}
    h = {}
    for speed in [300, 350]:
        df_speed = df[df['列车速度'] == speed].set_index('车站名称')
        g[speed] = df_speed['上行停车附加时分'].to_dict()
        h[speed] = df_speed['上行起车附加时分'].to_dict()

    return g, h


def read_safe_interval(path):
    """
    return dict of safe_interval
    key in ['aa', 'dd', 'pp', 'ap', 'pa', 'dp', 'pd']

    for example: safe_interval['aa'][speed][station] 到到安全间隔到达时
    """
    df = pd.read_excel(path)
    df['车站'] = df['车站'].astype(str)

    pairs = ['aa', 'dd', 'pp', 'ap', 'pa', 'dp', 'pd']
    safe_interval = {}
    for p in pairs:
        safe_interval[p] = {}

    for speed in [300, 350]:
        df_speed = df[df['speed'] == speed].set_index('车站')
        safe_interval['aa'][speed] = df_speed['到到安全间隔'].to_dict()
        safe_interval['dd'][speed] = df_speed['发发安全间隔'].to_dict()
        safe_interval['pp'][speed] = df_speed['通通安全间隔'].to_dict()
        safe_interval['ap'][speed] = df_speed['到通安全间隔'].to_dict()
        safe_interval['pa'][speed] = df_speed['通到安全间隔'].to_dict()
        safe_interval['dp'][speed] = df_speed['发通安全间隔'].to_dict()
        safe_interval['pd'][speed] = df_speed['通发安全间隔'].to_dict()

    aa_speed, dd_speed, pp_speed, ap_speed, pa_speed, dp_speed, pd_speed = safe_interval['aa'], safe_interval['dd'], safe_interval['pp'], \
                                                                           safe_interval['ap'], safe_interval['pa'], safe_interval['dp'], \
                                                                           safe_interval['pd']
    return aa_speed, dd_speed, pp_speed, ap_speed, pa_speed, dp_speed, pd_speed


def read_train_table(path, station_name_list, up=0, encoding='gbk'):
    df = pd.read_csv(path, encoding=encoding)
    df.loc[:, '车站编号'] = df['站名'].apply(
        lambda name: str(station_name_list.index(name) + 1))
    grouped = df.groupby('车次')

    train_table = {}
    for trn, group in grouped:
        if up == 0:
            train_id = trn / 2
        elif up == 1:
            train_id = (trn + 1) / 2
        else:
            raise ValueError("the argument up should be 0 or 1.")
        train_table[train_id] = {}

        k = 0  # 是否是起始站
        for row in group.index:
            station = group.loc[row, '车站编号']
            train_table[train_id][station] = {}
            if k == 0:  # 始发站
                train_table[train_id][station]['dep'] = round(
                    time2num(group.loc[row, '发点']), 2)
                train_table[train_id][station]['arr'] = round(
                    time2num(group.loc[row, '发点']), 2)
            elif k == len(group) - 1:  # 终点站
                train_table[train_id][station]['dep'] = round(
                    time2num(group.loc[row, '到点']), 2)
                train_table[train_id][station]['arr'] = round(
                    time2num(group.loc[row, '到点']), 2)
            else:
                train_table[train_id][station]['dep'] = round(
                    time2num(group.loc[row, '发点']), 2)
                train_table[train_id][station]['arr'] = round(
                    time2num(group.loc[row, '到点']), 2)
            k = k + 1
    return train_table


def write_train_table(path, train_table, station_name_list, direction='up', sort=True, encoding='utf-8'):
    """
    列: 车次ID, 车次, 站序, 站名, 到点, 发点
    """

    df = pd.DataFrame(columns=['车次ID', '车次', '站序', '站名', '到点', '发点'])

    train_list = list(train_table.keys())
    if sort:
        train_list.sort(key=lambda tr: int(tr), reverse=False)

    for train_id in train_list:
        if direction == 'up':
            trn = 2 * train_id
        elif direction == 'down':
            trn = 2 * train_id - 1
        else:
            raise ValueError("The direction of train should be up or down.")
        k = 0
        train_route = train_table[train_id]
        for station, times in train_route.items():
            row = {'车次ID': train_id,
                   '车次': trn,
                   '站序': k,
                   '站名': station_name_list[int(station) - 1],
                   '到点': num2time(times['arr']),
                   '发点': num2time(times['dep'])}
            df = df.append(row, ignore_index=True)
            k = k + 1

    df.to_csv(path, encoding=encoding, index=False)
    return
