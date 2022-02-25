import math


def get_dep_arr_pass_trains(station_list, train_list):
    dep_trains = {}
    arr_trains = {}
    pass_trains = {}
    for station in station_list:
        dep_trains[station] = []
        arr_trains[station] = []
        pass_trains[station] = []
        for train in train_list:
            if station in train.passSta:
                pass_trains[station].append(train)
            elif station in train.stopSta:
                if station != train.arrSta:
                    dep_trains[station].append(train)
                if station != train.depSta:
                    arr_trains[station].append(train)
    return dep_trains, arr_trains, pass_trains


def gen_sub_train_list(train_list, interval_list):
    """
    :param:
    
    interval_list should be a list of tuples, where the i-th tuple is (begin_{i}, end_{i})
    """
    select_train_list = []
    for interval in interval_list:
        select_train_list = select_train_list + train_list[interval[0]: interval[1]]
    return select_train_list


def remove_pre_trains(pre_train_table, train_list, interval_list, infeas_trains):
    for interval in interval_list:
        for traNo in set([train.traNo for train in train_list[interval[0]: interval[1]]] + infeas_trains):
            pre_train_table.pop(traNo)


def time2num(t):
    """
    input:  t is time string
            t's format is 'hour:min:sec'

    output: num is a floating point number in minutes
    """
    h, m, s = t.split(':')
    num = 60 * float(h) + float(m) + float(s) / 60
    return num


def num2time(num):
    """
    input:  num is a floating point number in minutes

    output: t is time string
            t's format is 'hour:min:sec'
    """
    num_dec, num_int = math.modf(num)
    s = round(num_dec * 60) # second
    if s >= 60:
        num_int = num_int + 1
        s = 0
    num_int = int(num_int)
    h = num_int // 60 # hour
    m = math.floor(num_int - h * 60) # min
    t = "{0:d}:{1:02d}:{2:02d}".format(h,m,s) # time string
    return t


def get_train_table(train_list, d_var, a_var):
    train_table = {}
    for train in train_list:
        train_id = train.traNo
        train_table[train_id] = {}
        for station in train.staList:
            train_table[train_id][station] = {}
            train_table[train_id][station]['dep'] = round(d_var[train][station].x, 2)
            train_table[train_id][station]['arr'] = round(a_var[train][station].x, 2)
    return train_table
