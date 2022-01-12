import collections
import re
from typing import *
import networkx as nx

from Train import *
from Node import *
import copy
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from collections import defaultdict
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
train_list: List[Train] = []
node_list = {}  # 先用车站做key，再用t做key索引到node
start_time = time.time()
sec_times_all = {}


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
            if "s" not in sta and "t" not in sta:
                multiplier[(sta, t)] = 0.0
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
        if not train.is_feasible:
            continue
        for node in train.feasible_path:
            train.timetable[node[0]] = node[1]


def update_lagrangian_multipliers(alpha, subgradient_dict):
    for node in multiplier.keys():
            multiplier[node] += alpha * max(subgradient_dict[node], 0)


def update_yv_multiplier():
    """
    multiplier for each node (v)
    """
    for (station, t), predecessors in node_prec_map.items():
        yv_multiplier[station, t] = sum(
            multiplier[p] for p in predecessors
        )
    logger.info("multiplier for 'v (nodes)' updated")


def update_subgradient_dict(subgradient_dict, opt_path_LR):
    for node in opt_path_LR[1:-1]:  # z_{j v}
        subgradient_dict[node] += 1


if __name__ == '__main__':

    station_size = int(os.environ.get('station_size', 5))
    train_size = int(os.environ.get('train_size', 100))
    time_span = int(os.environ.get('time_span', 100))
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
    iter_max = 1000
    interval = 1
    while gap > minGap and iter < iter_max:
        # compile adjusted multiplier for each node
        #   from the original Lagrangian
        update_yv_multiplier()
        nonzeros_xa = {k: v for k, v in xa_map.items() if v > 0}
        nonzeros = {k: v for k, v in yv_multiplier.items() if v > 0}
        # LR: train sub-problems solving
        path_cost_LR = 0
        subgradient_dict = defaultdict(lambda: -1)
        for train in train_list:
            train.update_arc_multiplier()
            train.opt_path_LR, train.opt_cost_LR = train.shortest_path()
            train.update_arc_chosen()  # LR中的arc_chosen，用于更新乘子
            path_cost_LR += train.opt_cost_LR
            update_subgradient_dict(subgradient_dict, train.opt_path_LR)
        logger.info("dual subproblems finished")

        # feasible solutions
        path_cost_feasible = 0
        # todo, 他这里不对，不应该排序么
        occupied_nodes = set()
        count = 0
        for idx, train in enumerate(train_list):
            train.update_primal_graph(occupied_nodes)

            train.feasible_path, train.feasible_cost = train.shortest_path_primal()

            if train.feasible_cost == np.inf:
                path_cost_feasible += max(d['weight']for i, j, d in train.subgraph.edges(data=True)) * len(train.v_staList)
                continue
            else:
                count += 1
            occupied_nodes.update(train.feasible_path[1:-1])
            path_cost_feasible += train.feasible_cost
        logger.info(f"maximum cardinality of feasible paths: {count}")

        UB.append(path_cost_feasible)

        # update lagrangian multipliers
        if iter < 20:
            alpha = 0.5 / (iter + 1)
        else:
            alpha = 0.5 / 20
        update_lagrangian_multipliers(alpha, subgradient_dict)
        LB.append(path_cost_LR)

        iter += 1
        gap = (UB[-1] - LB[-1]) / abs(LB[-1])

        if iter % interval == 0:
            print("==================  iteration " + str(iter) + " ==================")
            print("                 current gap: " + str(round(gap * 100, 5)) + "% \n")
            print(f"UB: {path_cost_feasible}; LB: {path_cost_LR}")

    get_train_timetable_from_result()
    print("================== solution found ==================")
    print("                 final gap: " + str(round(gap * 100, 5)) + "% \n")

    '''
    draw timetable
    '''
    plt.rcParams['figure.figsize'] = (10.0, 5.0)
    plt.rcParams["font.family"] = 'Times'
    plt.rcParams["font.size"] = 9
    fig = plt.figure(dpi=200)
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
        if not train.is_feasible:
            continue
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
