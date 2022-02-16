import pandas as pd

from jsp.Train import Train
from jsp.util import time2num


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
    train_series = df.apply(lambda row: parse_row_to_train(row, station_list, g, h, miles), axis=1)
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
    return aa_speed, dd_speed, pp_speed, ap_speed, pa_speed, dp_speed, pd_speed

    aa_speed[speed][station] 安全间隔到达时
    """
    df = pd.read_excel(path)
    df['车站'] = df['车站'].astype(str)

    aa_speed = {}
    dd_speed = {}
    pp_speed = {}
    ap_speed = {}
    pa_speed = {}
    dp_speed = {}
    pd_speed = {}

    for speed in [300, 350]:
        df_speed = df[df['speed'] == speed].set_index('车站')
        aa_speed[speed] = df_speed['到到安全间隔'].to_dict()
        dd_speed[speed] = df_speed['发发安全间隔'].to_dict()
        pp_speed[speed] = df_speed['通通安全间隔'].to_dict()
        ap_speed[speed] = df_speed['到通安全间隔'].to_dict()
        pa_speed[speed] = df_speed['通到安全间隔'].to_dict()
        dp_speed[speed] = df_speed['发通安全间隔'].to_dict()
        pd_speed[speed] = df_speed['通发安全间隔'].to_dict()

    return aa_speed, dd_speed, pp_speed, ap_speed, pa_speed, dp_speed, pd_speed


def read_train_table(path, up=0, encoding='gbk'):
    # 站名 up=1
    station_name_list = ['北京南', '廊坊', '京津线路所', '津沪线路所', '天津南', '沧州西', '德州东', '济南西', '崔马庄线路所', '泰安',
                         '曲阜东', '滕州东', '枣庄', '徐州东', '宿州东', '蚌埠南', '定远', '滁州', '扬州线路所', '南京南',
                         '秦淮河线路所', '镇江南', '丹阳北', '常州北', '无锡东', '苏州北', '昆山南', '黄渡线路所', '上海虹桥']

    df = pd.read_csv(path, encoding=encoding)
    df.loc[:, '车站编号'] = df['站名'].apply(lambda name: str(station_name_list.index(name) + 1))
    grouped = df.groupby('车次')

    train_table = {}
    for trn, group in grouped:
        if up == 0:
            train_id = trn / 2
        elif up == 1:
            train_id = (trn + 1) / 2
        train_table[train_id] = {}

        k = 0  # 是否是起始站
        for row in group.index:
            station = group.loc[row, '车站编号']
            train_table[train_id][station] = {}
            if k == 0:  # 始发站
                train_table[train_id][station]['dep'] = round(time2num(group.loc[row, '发点']), 2)
                train_table[train_id][station]['arr'] = round(time2num(group.loc[row, '发点']), 2)
            elif k == len(group) - 1:  # 终点站
                train_table[train_id][station]['dep'] = round(time2num(group.loc[row, '到点']), 2)
                train_table[train_id][station]['arr'] = round(time2num(group.loc[row, '到点']), 2)
            else:
                train_table[train_id][station]['dep'] = round(time2num(group.loc[row, '发点']), 2)
                train_table[train_id][station]['arr'] = round(time2num(group.loc[row, '到点']), 2)
            k = k + 1
    return train_table
