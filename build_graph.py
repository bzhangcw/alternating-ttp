# -*- coding: utf-8 -*-
# @author: PuShanwen
# @email: 2019212802@live.sufe.edu.cn
# @date: 2022/01/10
import collections

def build_graph(origin, ):
    Queue = collections.deque()
    Queue.append(origin)  # add initial label into Queue, Queue存储各个label，各个label含各自的路径信息
    # main loop of the algorithm
    while len(Queue) > 0:
        current_node = Queue.pop()  # 当前的label
        # extend the label
        for train, arcs_dict in current_node.out_arcs.items():
            train.subgraph.add_edge()
        if train.traNo in current_node.out_arcs.keys():  # 该节点有该列车的流出弧的话，才进行后续节点的加入
            for out_arc in current_node.out_arcs[train.traNo].values():  # 遍历当前点的流出弧，找到下一节点
            child_node = node_list[out_arc.staBelong_next][out_arc.timeBelong_next].name

