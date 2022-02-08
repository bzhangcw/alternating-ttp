# def get_stop_pass_trains(station_list, train_list):
#     stop_trains = {}
#     pass_trains = {}
#
#     for station in station_list:
#         stop_trains[station] = []
#         pass_trains[station] = []
#     for train in train_list:
#         for station in station_list[::int(2 * train.up - 1)]:
#             if train.linePlan[station] in {-1, 1}:
#                 enter_station = station
#                 break
#         for station in station_list[::int(1 - 2 * train.up)]:
#             if train.linePlan[station] in {-1, 1}:
#                 leave_station = station
#                 break
#         # print("Train: ", train.traNo)
#         # print("enter_station:", enter_station)
#         # print("leave_station:", leave_station)
#
#         in_rail_flag = 0
#         for station in station_list[::int(2 * train.up - 1)]:
#             if station == enter_station:
#                 in_rail_flag = 1
#             if station == leave_station:
#                 in_rail_flag = 0
#
#             if train.linePlan[station] == 1:
#                 stop_trains[station].append(train)
#                 # print("stop at station ", station)
#             if in_rail_flag == 1 and train.linePlan[station] in {0, -1}:
#                 pass_trains[station].append(train)
#                 # print("pass through station ", station)
#     return stop_trains, pass_trains

def get_stop_pass_trains(station_list, train_list):
    stop_trains = {}
    pass_trains = {}
    for station in station_list:
        stop_trains[station] = []
        pass_trains[station] = []
        for train in train_list:
            if station in train.passSta:
                pass_trains[station].append(train)
            elif station in train.stopSta:
                stop_trains[station].append(train)
    return stop_trains, pass_trains


def assign_high_level_train(train_list, sec_times, wait_time_lb, wait_time_ub, TimeSpan=1080, TimeStart=360):

    D_var = {}
    A_var = {}

    high_train_list = [trn for trn in train_list if trn.standard == 1]

    for i in range(len(high_train_list)):
        train = high_train_list[i]

        D_var[train] = {}
        A_var[train] = {}

        for station in train.staList[:-1]:
            if station == train.depSta:
                A_var[train][station] = train.preferred_time
                D_var[train][station] = train.preferred_time
            elif train.linePlan[station] in {0, -1}:
                D_var[train][station] = A_var[train][station]
            elif train.linePlan[station] == 1:
                D_var[train][station] = A_var[train][station] + wait_time_lb[station]

            nextStation = str(int(station) + int(2 * train.up - 1))
            if train.up == 1:
                A_var[train][nextStation] = D_var[train][station] + sec_times[train.speed][
                    (station, nextStation)] + train.delta[(station, nextStation)]
            elif train.up == 0:
                A_var[train][nextStation] = D_var[train][station] + sec_times[train.speed][
                    (nextStation, station)] + train.delta[(station, nextStation)]

        D_var[train][train.arrSta] = A_var[train][train.arrSta]

    assigned_train_list = []
    for train in high_train_list:
        if train.standard == 1:
            if A_var[train][train.staList[-1]] > TimeSpan + TimeStart or D_var[train][train.staList[0]] < TimeStart:
                print('Train {} cannot be assigned!'.format(train.traNo))
                del D_var[train]
                del A_var[train]
            else:
                assigned_train_list.append(train)

    return assigned_train_list, D_var, A_var


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper