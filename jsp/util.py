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


def time2num(t):
    """
    input:  t is time
            t's format is 'hour:min:sec'

    output: num is a floating point number in minutes
    """
    h, m, s = t.split(':')
    num = 60 * float(h) + float(m) + float(s) / 60
    return num
