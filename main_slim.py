import collections
import re
from Train import *
from Node import *
import copy
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

import logging
import sys
import os

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)
logger = logging.getLogger("railway")
logger.setLevel(logging.INFO)

'''
initialize stations, sections, trains and train arcs
'''
station_list = []  # 实际车站列表
v_station_list = []  # 时空网车站列表，车站一分为二 # 源节点s, 终结点t
sec_times = {}  # total miles for stations
miles = []
train_list = []
node_list = {}  # 先用车站做key，再用t做key索引到node
start_time = time.time()
sec_times_all = {}
multiplier = {"y_v": {}, "z_jv": {}}


def read_station(path, size):
    global miles, v_station_list, station_list
    df = pd.read_excel(path).sort_values('站名')
    df = df.iloc[:size, :]
    miles = df['里程'].values
    station_list = df['站名'].astype(str).to_list()
    v_station_list.append('_s')
    for sta in station_list:
        if station_list.index(sta) != 0:  # 不为首站，有到达
            v_station_list.append('_' + sta)
        if station_list.index(sta) != len(station_list) - 1:
            v_station_list.append(sta + '_')  # 不为尾站，又出发
    v_station_list.append('_t')


def read_section(path):
    df = pd.read_excel(path).assign(
        interval=lambda dfs: dfs['区间名'].apply(lambda x: tuple(x.split("-")))
    ).set_index("interval")
    global sec_times, sec_times_all
    sec_times = df[350].to_dict()
    sec_times_all = df.to_dict()


def parse_row_to_train(row):
    tr = Train(row['车次ID'], 0, time_span)
    tr.preferred_time = row['偏好始发时间']
    tr.up = row['上下行']
    tr.standard = row['标杆车']
    tr.speed = row['速度']
    # todo, what does -1 mean?
    tr.linePlan = {k: max(row[k], 0) for k in station_list}
    tr.init_traStaList(station_list)
    tr.create_arcs_LR(sec_times, time_span)
    return tr


def read_train(path, size=10):
    df = pd.read_excel(path, dtype={"车次ID": str})
    df = df.rename(columns={k: str(k) for k in df.columns})
    df = df.iloc[:size, :]
    train_series = df.apply(lambda row: parse_row_to_train(row), axis=1)
    global train_list
    train_list = train_series.to_list()


def init_nodes():
    '''
    initialize nodes, associated with incoming nad outgoing train arcs
    '''
    # source node
    node_list['s_'] = {}
    node_list['s_'][-1] = Node('s_', -1)
    # initialize node dictionary with key [sta][t]
    for sta in v_station_list:  # 循环车站
        node_list[sta] = {}
        for t in range(0, time_span):  # 循环时刻t
            node = Node(sta, t)
            node_list[sta][t] = node
    # sink node
    node_list['_t'] = {}
    node_list['_t'][-1] = Node('_t', -1)


def add_arcs_to_nodes_by_flow():
    # associate node with train arcs, add incoming and outgoing arcs to nodes
    for nodes_sta in node_list.values():
        for node in nodes_sta.values():
            for train in train_list:
                node.associate_with_outgoing_arcs(train)
                node.associate_with_incoming_arcs(train)


# 通过列车弧的资源占用特性，将arc与node的关系建立
def associate_arcs_nodes_by_resource_occupation():
    for sta in v_station_list:
        if sta != v_station_list[0] and sta.endswith('_'):  # all section departure stations
            for t in range(0, time_span):
                # 先用车站做key，再用t做key索引到node
                cur_node = node_list[sta][t]

                if len(cur_node.out_arcs) == 0:  # 没有弧
                    continue

                for tra in cur_node.out_arcs.keys():  # 先索引包含的列车
                    for out_arc in cur_node.out_arcs[tra].values():  # 遍历这个列车在这个点的所有弧
                        before_occupy = out_arc.before_occupy_dep
                        after_occupy = out_arc.after_occupy_dep
                        for i in range(0, before_occupy + 1):  # 前面节点占用，在这里把自己加上，可以取到0
                            if t - i >= 0:
                                node_list[sta][t - i].incompatible_arcs.append(out_arc)
                                out_arc.node_occupied.append(node_list[sta][t - i])
                            else:
                                break
                        for i in range(1, after_occupy + 1):  # 后面节点占用，这里就不取0了
                            if t + i < time_span:
                                node_list[sta][t + i].incompatible_arcs.append(out_arc)
                                out_arc.node_occupied.append(node_list[sta][t + i])
                            else:
                                break

        elif sta != v_station_list[-1] and sta.startswith('_'):  # all section arrival stations
            for t in range(0, time_span):
                # 先用车站做key，再用t做key索引到node
                cur_node = node_list[sta][t]

                if len(cur_node.in_arcs) == 0:  # 没有弧
                    continue

                for tra in cur_node.in_arcs.keys():  # 先索引包含的列车
                    for in_arc in cur_node.in_arcs[tra].values():  # 遍历这个列车在这个点的所有弧
                        before_occupy = in_arc.before_occupy_arr
                        after_occupy = in_arc.after_occupy_arr
                        for i in range(0, before_occupy + 1):  # 前面节点占用
                            if t - i >= 0:
                                node_list[sta][t - i].incompatible_arcs.append(in_arc)
                                in_arc.node_occupied.append(node_list[sta][t - i])
                            else:
                                break
                        for i in range(1, after_occupy + 1):  # 后面节点占用
                            if t + i < time_span:
                                node_list[sta][t + i].incompatible_arcs.append(in_arc)
                                in_arc.node_occupied.append(node_list[sta][t + i])
                            else:
                                break


# def get_train_timetable_from_result():
#     for train in trainList:
#         print("===============Tra_" + train.traNo + "======================")
#         for i in range(len(train.v_staList) - 1):
#             curSta = train.v_staList[i]
#             nextSta = train.v_staList[i + 1]
#             for t, arcs_t in train.arcs[curSta, nextSta].items():
#                 for arc_length, arc in arcs_t.items():
#                     if arc.isChosen_LR == 1:
#                         print(curSta + "(" + str(t) + ") => " + nextSta + "(" + str(t + arc_length) + ")")
#                         train.timetable[curSta] = t
#                         train.timetable[nextSta] = t + arc_length
def get_train_timetable_from_result():
    for train in train_list:
        print("===============Tra_" + train.traNo + "======================")
        for node in train.feasible_path.node_passed:
            train.timetable[node[0]] = node[1]


# Labelling Algorithm
class Label(object):
    def __init__(self):
        self.node_passed = []  # 该标记的path，即从source node走到该点的路径，存储一系列node名称(sta, t)，防止deepcopy效率太低！
        self.cost = 0

    def __repr__(self):
        temp = ""
        for node_name in self.node_passed:
            temp += node_name[0] + "," + str(node_name[1])
            if node_name != self.node_passed[-1]:
                temp += " => "
        return temp


'''
labelling_SPPRC: apply labelling correction algorithm to solve SPPRC(shortest path problem with resource constraint)
param:
    Graph: network object
    org: string, source node
    des: string destination node
    rmp_pi: dict, the dual price of each constraint in RMP
'''


def label_correcting_shortest_path(summary_interval, org, des, train):
    '''
    get the shortest path for the specific train
    :param summary_interval:
    :param org: source node name [sta, t]
    :param des: sink node name [sta, t]
    :param train: train to generate train time space network
    :return:
    '''
    # initialize Queue
    # c_time = time.time()
    Queue = collections.deque()
    SE_list = []
    summary_interval = summary_interval
    # create initial label
    label = Label()
    label.node_passed = [org]
    Queue.append(label)  # add initial label into Queue, Queue存储各个label，各个label含各自的路径信息
    Paths = []  # all complete paths
    cnt = 0
    cnt2 = 0
    # main loop of the algorithm
    while len(Queue) > 0:
        current_path = Queue.pop()  # 当前的label
        # extend the label
        last_node_name = current_path.node_passed[-1]
        last_node = node_list[last_node_name[0]][last_node_name[1]]  # 当前点
        if train.traNo in last_node.out_arcs.keys():  # 该节点有该列车的流出弧的话，才进行后续节点的加入
            for out_arc in last_node.out_arcs[train.traNo].values():  # 遍历当前点的流出弧，找到下一节点
                child_node = node_list[out_arc.staBelong_next][out_arc.timeBelong_next].name
                cnt2 += 1
                extended_path = copy.deepcopy(current_path)  # 新label
                extended_path.node_passed.append(child_node)
                extended_path.cost += out_arc.arc_length
                for occupy_node in out_arc.node_occupied:
                    extended_path.cost += occupy_node.multiplier

                Queue.append(extended_path)

        if current_path.node_passed[-1][0] == des[0]:  # 注意不能直接path[-1] == des，引用类型相同是判断地址相同
            Paths.append(current_path)

    # choose optimal solution
    opt_path = None
    min_cost = 10000000
    for path in Paths:
        if path.cost < min_cost:
            min_cost = path.cost
            opt_path = path
    path_cost = opt_path.cost
    # a_time = time.time()
    # print(a_time - c_time)
    return opt_path, path_cost


def label_correcting_shortest_path_with_forbidden(summary_interval, org, des, train):
    '''
    get the shortest path for the specific train with the remained subgraph
    :param summary_interval:
    :param org: source node name [sta, t]
    :param des: sink node name [sta, t]
    :param train: train to generate train time space network
    :return:
    '''
    # initialize Queue
    # c_time = time.time()
    Queue = collections.deque()
    SE_list = []
    summary_interval = summary_interval
    # create initial label
    label = Label()
    label.node_passed = [org]
    Queue.append(label)  # add initial label into Queue, Queue存储各个label，各个label含各自的路径信息
    Paths = []  # all complete paths
    cnt = 0
    cnt2 = 0
    # main loop of the algorithm
    while len(Queue) > 0:
        current_path = Queue.pop()  # 当前的label
        # extend the label
        last_node_name = current_path.node_passed[-1]
        last_node = node_list[last_node_name[0]][last_node_name[1]]  # 当前点

        if train.traNo in last_node.out_arcs.keys():  # 该节点有该列车的流出弧的话，才进行后续节点的加入
            for out_arc in last_node.out_arcs[train.traNo].values():  # 遍历当前点的流出弧，找到下一节点
                if node_list[out_arc.staBelong_next][out_arc.timeBelong_next].isOccupied:  # 若下一节点已经被占用
                    continue
                child_node = node_list[out_arc.staBelong_next][out_arc.timeBelong_next].name
                cnt2 += 1
                extended_path = copy.deepcopy(current_path)  # 新label
                extended_path.node_passed.append(child_node)
                extended_path.cost += out_arc.arc_length
                for occupy_node in out_arc.node_occupied:
                    extended_path.cost += occupy_node.multiplier

                # if child_node in SE_list:
                #     Queue.appendleft(extended_path)
                # else:
                #     SE_list.append(child_node)
                #     Queue.append(extended_path)
                Queue.append(extended_path)

                # if cnt2 % summary_interval == 0:
                #     print('extended_path:', extended_path.__repr__())

        if current_path.node_passed[-1][0] == des[0]:  # 注意不能直接path[-1] == des，引用类型相同是判断地址相同
            Paths.append(current_path)

    # choose optimal solution
    opt_path = None
    min_cost = 10000000
    for path in Paths:
        if path.cost < min_cost:
            min_cost = path.cost
            opt_path = path
    path_cost = opt_path.node_passed[-2][1] - opt_path.node_passed[1][1]

    # a_time = time.time()
    # print(a_time - c_time)
    return opt_path, path_cost


def update_lagrangian_multipliers(alpha):
    total_cost = 0
    for sta in v_station_list:
        if sta == v_station_list[0] or sta == v_station_list[-1]:  # s_和_t并没有乘子
            continue
        for node in node_list[sta].values():  # 这个站的各个t的node
            temp = 0
            for arc in node.incompatible_arcs:
                temp += arc.isChosen_LR

            node.multiplier = max(0, node.multiplier + alpha * (temp - 1))  # 1为node capacity
            total_cost += node.multiplier
    return total_cost


def set_node_occupation(train):
    for node_id in range(1, len(train.feasible_path.node_passed) - 2):
        node = train.feasible_path.node_passed[node_id]
        next_node = train.feasible_path.node_passed[node_id + 1]
        for node in train.arcs[node[0], next_node[0]][node[1]][next_node[1] - node[1]].node_occupied:
            node.isOccupied = True
    train.last_feasible_path = copy.deepcopy(train.feasible_path)


def clear_node_occupation():
    for train in train_list:
        if train.last_feasible_path is not None:
            for node_id in range(1, len(train.last_feasible_path.node_passed) - 2):
                node = train.last_feasible_path.node_passed[node_id]
                next_node = train.last_feasible_path.node_passed[node_id + 1]
                for node in train.arcs[node[0], next_node[0]][node[1]][next_node[1] - node[1]].node_occupied:
                    node.isOccupied = False


if __name__ == '__main__':

    station_size = int(os.environ.get('station_size', 30))
    train_size = int(os.environ.get('train_size', 5))
    time_span = int(os.environ.get('time_span', 500))
    logger.info(f"size: #train,#station,#timespan: {train_size, station_size, time_span}")
    read_station('raw_data/1-station.xlsx', station_size)
    read_section('raw_data/3-section-time.xlsx')
    read_train('raw_data/6-lineplan-down.xlsx', train_size)

    '''
    initialization
    '''
    logger.info("reading finish")
    # init_trains()
    init_nodes()
    logger.info("step 1")
    add_arcs_to_nodes_by_flow()
    logger.info("step 2")
    associate_arcs_nodes_by_resource_occupation()
    logger.info("step 3")

    '''
    Lagrangian relaxation approach
    '''
    LB = []
    UB = []

    minGap = 0.1
    gap = 100
    alpha = 0
    iter = 0
    interval = 10
    while gap > minGap:
        # LR: train sub-problems solving
        path_cost_LR = 0
        for train in train_list:
            train.opt_path_LR, train.opt_cost_LR = label_correcting_shortest_path(20, node_list['s_'][-1].name,
                                                                                  node_list['_t'][-1].name, train)
            train.update_arc_chosen()  # LR中的arc_chosen，用于更新乘子
            path_cost_LR += train.opt_cost_LR

        # feasible solutions
        path_cost_feasible = 0
        for train in train_list:
            train.feasible_path, train.feasible_cost = label_correcting_shortest_path_with_forbidden(20,
                                                                                                     node_list['s_'][
                                                                                                         -1].name,
                                                                                                     node_list['_t'][
                                                                                                         -1].name,
                                                                                                     train)
            set_node_occupation(train)  # 可行解不需要arc_chosen，用opt_path即可
            path_cost_feasible += train.feasible_cost
        clear_node_occupation()  # 清除不能在循环内，会将同一轮次的上一列车的占用给清空了
        UB.append(path_cost_feasible)

        # update lagrangian multipliers
        if iter < 20:
            alpha = 0.5 / (iter + 1)
        else:
            alpha = 0.5 / 20
        multiplier_cost = update_lagrangian_multipliers(alpha)
        LB.append(path_cost_LR - multiplier_cost)

        iter += 1
        gap = (UB[-1] - LB[-1]) / LB[-1]

        if iter % interval == 0:
            print("==================  iteration " + str(iter) + " ==================")
            print("                 current gap: " + str(round(gap * 100, 5)) + "% \n")

    get_train_timetable_from_result()
    print("================== solution found ==================")
    print("                 final gap: " + str(round(gap * 100, 5)) + "% \n")

    '''
    draw timetable
    '''
    plt.rcParams['figure.figsize'] = (12.0, 8.0)
    plt.rcParams["font.family"] = 'Times'
    plt.rcParams["font.size"] = 9
    fig = plt.figure(figsize=(7, 7), dpi=200)
    color_value = {
        '0': 'midnightblue',
        '1': 'mediumblue',
        '2': 'c',
        '3': 'orangered',
        '4': 'm',
        '5': 'fuchsia',
        '6': 'olive'
    }

    xlist = []
    ylist = []
    for i in range(len(train_list)):
        train = train_list[i]
        xlist = []
        ylist = []
        for sta_id in range(len(train.staList)):
            sta = train.staList[sta_id]
            if sta_id != 0:  # 不为首站, 有到达
                if "_" + sta in train.v_staList:
                    xlist.append(train.timetable["_" + sta])
                    ylist.append(miles[station_list.index(sta)])
            if sta_id != len(train.staList) - 1:  # 不为末站，有出发
                if sta + "_" in train.v_staList:
                    xlist.append(train.timetable[sta + "_"])
                    ylist.append(miles[station_list.index(sta)])
        plt.plot(xlist, ylist, color=color_value[str(i % 7)], linewidth=1.5)
        plt.text(xlist[0] + 0.8, ylist[0] + 4, train.traNo, ha='center', va='bottom',
                 color=color_value[str(i % 7)], weight='bold', family='Times', fontsize=9)

    plt.grid(True)  # show the grid
    plt.ylim(0, miles[-1])  # y range

    plt.xlim(0, time_span)  # x range
    plt.xticks(np.linspace(0, time_span, int(time_span / 10 + 1)))

    plt.yticks(miles, station_list, family='Times')
    plt.xlabel('Time (min)', family='Times new roman')
    plt.ylabel('Space (km)', family='Times new roman')
    plt.show()

    end_time = time.time()
    time_elapsed = end_time - start_time
    print(time_elapsed)

    ## plot the bound updates
    font_dic = {
                "family": "Times",
                "style": "oblique",
                "weight": "normal",
                "color": "green",
                "size": 20
                }



    x_cor = range(1, len(LB) + 1)
    plt.plot(x_cor, LB, label='LB')
    plt.plot(x_cor, UB, label='UB')
    plt.legend()
    plt.xlabel('Iteration', fontdict=font_dic)
    plt.ylabel('Bounds update', fontdict=font_dic)
    plt.title('LR: Bounds updates \n', fontsize=23)
    plt.show()
