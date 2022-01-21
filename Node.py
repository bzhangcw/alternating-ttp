from collections import defaultdict
from util import *

###########
# todo
# a few parameters to be fixed
###########


class Node():
    def __init__(self, sta, t):
        self.sta_located = sta
        self.t_located = t
        self.in_arcs = {}  # 流入该节点的弧集，以trainNo为key，弧为value
        self.out_arcs = {}  # 流出该节点的弧集，以trainNo为key, 弧为value
        self.incompatible_arcs = []  # 该节点对应资源占用<=1的约束中，不相容弧的集合，以trainNo为索引，子字典以arc_length为key
        self.multiplier = 0  # 该节点对应约束的拉格朗日乘子
        self.name = [self.sta_located, self.t_located]
        self.isOccupied = False  # 可行解中，该节点是否已经被占据
        self.build_precedence_map(epsilon=eps)
        self._str = self.__str__()
        self.is_sink = sta == NODE_SINK_ARR

    def __str__(self):
        return f"node:{self.sta_located}@{self.t_located}"

    def __repr__(self):
        return self._str

    def __hash__(self):
        return self._str.__hash__()

    def build_precedence_map(self, epsilon):
        """
        let self be w
        create a list of nodes that precede the current node:
            i.e., v <= w
        and w.t - v.t <= epsilon
        """
        node_prec_map[self.sta_located, self.t_located] = [
            (self.sta_located, self.t_located - t) for t in
            range(min(self.t_located, epsilon))
        ]

    def associate_with_incoming_arcs(self, train):
        global yv2xa_map
        '''
        associate node with train arcs, add incoming arcs to nodes
        :param train:
        :return:
        '''
        sta_node = self.sta_located
        t_node = self.t_located

        if sta_node not in train.v_staList:  # 若该车不经过该站，直接退出
            return -1

        # associate incoming arcs
        # train arc structure: key：[dep, arr], value为弧集字典(key: [t], value: arc字典, key为arc_length)
        if sta_node != train.v_staList[0]:  # 不为第一站，则拥有上一站
            preSta = train.v_staList[train.v_staList.index(sta_node) - 1]  # 已经考虑列车停站情况的车站集
            curSta = sta_node
            cur_arcs_ = train.new_arcs[preSta, curSta]  # 这个区间/车站的所有弧

            if self.is_sink:  # 若该弧流入该节点
                for t_node in cur_arcs_:
                    for arc_var in cur_arcs_[t_node]:
                        # 若不包含该车的弧列表，则先生成弧列表
                        self._add_in_arcs(train, arc_var)
            elif t_node in cur_arcs_:
                for arc_var in cur_arcs_[t_node]:
                    self._add_in_arcs(train, arc_var)

    def _add_in_arcs(self,train, arc_var):
        if train.traNo not in self.in_arcs.keys():
            self.in_arcs[train.traNo] = {}
        arc_length = arc_var.arc_length
        self.in_arcs[train.traNo][arc_length] = arc_var
        train.subgraph.add_edge((arc_var.staBelong_pre, arc_var.timeBelong_pre),
                                (arc_var.staBelong_next, arc_var.timeBelong_next),
                                weight=arc_length)
        yv2xa_map[(arc_var.staBelong_next, arc_var.timeBelong_next)][(
            arc_var.staBelong_pre, arc_var.timeBelong_pre, arc_var.staBelong_next,
            arc_var.timeBelong_next)] += 1
        xa_map[
            (arc_var.staBelong_pre, arc_var.timeBelong_pre), (
                arc_var.staBelong_next, arc_var.timeBelong_next)
        ][train.traNo] += 1

    def associate_with_outgoing_arcs(self, train):
        '''
        associate node with train arcs, add outgoing arcs to nodes
        :param train:
        :return:
        '''
        sta_node = self.sta_located
        t_node = self.t_located  # 所在点

        if sta_node not in train.v_staList or sta_node == train.v_staList[-1]:  # 列车径路不包含该站 或 该站为最后一站，则都不会有流出弧
            return -1

        curSta = sta_node
        nextSta = train.v_staList[train.v_staList.index(sta_node) + 1]
        cur_arcs = train.arcs[curSta, nextSta]
        if sta_node == train.v_staList[-2]:
            b = 0
        if t_node in cur_arcs.keys():  # 如果点t在列车区间/车站弧集当中，source node就是-1
            self.out_arcs[train.traNo] = {}
            for arc_length, arc_var in cur_arcs[t_node].items():
                self.out_arcs[train.traNo][arc_length] = arc_var
                train.subgraph.add_edge((arc_var.staBelong_pre, arc_var.timeBelong_pre),
                                        (arc_var.staBelong_next, arc_var.timeBelong_next),
                                        weight=arc_length)
