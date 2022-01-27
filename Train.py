# Here defines the info about the trains
import copy
import logging
from typing import Optional

import numpy as np
import sortedcollections

import igraph as ig
import networkx as nx
from util import *


class Train(object):

    def __init__(self, traNo, dep_LB, dep_UB, backend=0):
        '''
        construct
        :param traNo:
        :param dep_LB:
        :param dep_UB:
        '''
        self.traNo = traNo  # 列车车次
        self.dep_LB = dep_LB  # 始发时间窗下界
        self.dep_UB = dep_UB  # 始发时间窗上界
        self.arcs = {}
        self.new_arcs = {}
        self.stop_addTime = {}  # 停车附加时分
        self.start_addTime = {}  # 起车附加时分
        self.min_dwellTime = {}  # 最小停站时分
        self.max_dwellTime = {}  # 最大停站时分
        self.pass_station = {}  # 最大最小停站时分为0的车站
        self.secTimes = {}
        self.right_time_bound = {}  # 各站通过线路时间窗和列车始发时间窗综合确定的右侧边界
        self.depSta = None
        self.arrSta = None
        self.v_staList = []  # dual stations
        self.v_sta_type = {}
        self.staList = []  # actual stations
        self.linePlan = {}  # 开行方案字典
        self.opt_path_LR = None  # LR 中的最短路径
        self.last_opt_path_LR = None
        self.opt_cost_LR = 0
        self.opt_cost_LR_normal = 0
        self.feasible_path = None  # 可行解中的最短路径
        self.last_feasible_path = None  # 上一个可行解的最短路径，用于置0
        self.feasible_cost = 0
        self.timetable = {}  # 以virtual station为key，存int值
        self.speed = None  # 列车速度，300,350
        self.is_feasible = False
        ##################
        # subgraphs for each train
        ##################
        # subgraph for each train, to use in dual optimization (Lagrangian)
        #   the topology is never changed
        self.backend = backend
        if backend == 0:
            self.subgraph = nx.DiGraph(traNo=self.traNo)
        else:
            self.subgraph: Optional[ig.Graph] = None
        # copy the original network, use to create primal solution
        #   change with iteration and occupied nodes and maybe incompatible edges
        self.subgraph_primal = None
        #
        self.source = 's_', -1
        self.sink = '_t', -1
        self.logger = logging.getLogger(f"train#{self.traNo}")

        ###############
        # required by ig backend.
        ###############
        self._ig_nodes = sortedcollections.OrderedSet()
        self._ig_nodes_id = {}
        self._ig_edges = {}
        self._ig_s, self._ig_t = 0, 0
        self._ig_t_primal = 0
        self.max_edge_weight = 0

    def __repr__(self):
        return "train" + str(self.traNo)

    def init_traStaList(self, allStaList):
        '''
        create train staList, include s_, _t， only contains nodes associated with this train
        :param allStaList:
        :return:
        '''
        all_station_list = list(self.linePlan.keys())
        for station in all_station_list:
            if self.linePlan[station] in {-1, 1}:
                depSta = station
                break
        for station in all_station_list[::-1]:
            if self.linePlan[station] in {-1, 1}:
                arrSta = station
                break

        in_rail_flag = 0
        for station in all_station_list:
            if station == depSta:
                in_rail_flag = 1

            if in_rail_flag == 1:
                self.staList.append(station)

            if station == arrSta:
                in_rail_flag = 0

        self.v_staList.append('s_')
        for i, station in enumerate(self.staList):
            if i != 0:
                self.v_staList.append('_' + station)
                self.v_sta_type['_' + station] = "p" if self.linePlan[station] in (0, -1) else "a"
            if i != len(self.staList) - 1:  # 若不为实际车站的最后一站，则加上sta_
                self.v_staList.append(station + '_')
                self.v_sta_type[station + '_'] = "p" if self.linePlan[station] in (0, -1) else "s"

        self.v_staList.append('_t')

    def _subgraph_copy(self):
        return copy.deepcopy(self.subgraph)

    def truncate_train_time_bound(self, TimeSpan):
        right_bound_by_sink = []  # 从总天窗时间右端反推至该站的右侧边界，按运行最快了算
        accum_time = 0
        right_bound_by_sink.append(TimeSpan - accum_time)  # 最后一站的到达
        for sta_id in range(len(self.staList) - 1, 0, -1):
            accum_time += self.secTimes[self.staList[sta_id - 1], self.staList[sta_id]]
            right_bound_by_sink.append(TimeSpan - accum_time)
            if sta_id != 1:  # 最后一站不用加上停站时分了
                if self.linePlan[self.staList[sta_id - 1]] == 1:  # 若停站了则加一个2
                    accum_time += self.min_dwellTime[self.staList[sta_id - 1]]
                else:
                    accum_time += 0
                right_bound_by_sink.append(TimeSpan - accum_time)
        right_bound_by_sink = list(reversed(right_bound_by_sink))

        # 计算从始发站开始，到终点站的最晚时间，计算方法有误，暂时不适用 todo
        right_bound_by_dep = []
        right_bound_by_dep.append(self.dep_UB)  # 第一个站
        accum_time = self.dep_UB
        for sta_id in range(0, len(self.staList) - 1):  # 最后一个站不考虑
            accum_time += self.secTimes[self.staList[sta_id], self.staList[sta_id + 1]]
            right_bound_by_dep.append(accum_time)
            if sta_id != len(self.staList) - 2:  # 最后一个区间，不加停站时分
                accum_time += self.min_dwellTime[self.staList[sta_id]]
                right_bound_by_dep.append(accum_time)

        for sta in self.v_staList:
            if sta == self.v_staList[-1] or sta == self.v_staList[0]:
                continue
            right_bound_dep = right_bound_by_dep[self.v_staList.index(sta) - 1]
            right_bound_sink = right_bound_by_sink[self.v_staList.index(sta) - 1]
            # self.right_time_bound[sta] = min(right_bound_dep, right_bound_sink)
            self.right_time_bound[sta] = right_bound_sink

        assert all([bound >= 0 for bound in self.right_time_bound.values()])

    def create_subgraph(self, secTimes, TimeSpan):
        if self.backend == 0:
            self.create_subgraph_nx(secTimes, TimeSpan)
        elif self.backend == 1:
            self.create_subgraph_ig(secTimes, TimeSpan)
        else:
            raise ValueError(f"backend {self.backend} is not supported")

    def update_arc_multiplier(self):
        """
        rebuild current price/multiplier to cal shortest path
        """
        if self.backend == 0:
            self.update_arc_multiplier_nx()
        elif self.backend == 1:
            self.update_arc_multiplier_ig()
        else:
            raise ValueError(f"backend {self.backend} is not supported")

    def update_primal_graph(self, *args, **kwargs):
        if self.backend == 0:
            self.update_primal_graph_nx(*args, **kwargs)
        elif self.backend == 1:
            self.update_primal_graph_ig(*args, **kwargs)
        else:
            raise ValueError(f"backend {self.backend} is not supported")

    def shortest_path(self, *args, **kwargs):
        if self.backend == 0:
            return self.shortest_path_nx(*args, **kwargs)
        elif self.backend == 1:
            return self.shortest_path_ig(*args, **kwargs)
        else:
            raise ValueError(f"backend {self.backend} is not supported")

    def shortest_path_primal(self):
        """
        """
        self.is_feasible = False
        self.feasible_path = None
        self.feasible_cost = np.inf
        ssp, cost = self.shortest_path(option='primal')
        self.is_feasible = (cost < np.inf)

        return ssp, cost

    def normalize_cost(self):

        self.opt_cost_LR_normal = self.opt_cost_LR / len(self.v_staList)

    #################
    # Networkx
    #################

    def create_subgraph_nx(self, secTimes, TimeSpan):
        """
        create the subgraph for this train
            use the created nodelist
            you should not initialize any new nodes.
        """
        raise ValueError("fix this")

        self.depSta = self.staList[0]
        self.arrSta = self.staList[-1]
        self.secTimes = secTimes
        self.truncate_train_time_bound(TimeSpan)

        minArr = self.dep_LB  # for curSta(judge by dep)
        '''
        create arcs involving node s
        '''
        self.arcs['s_', self.staList[0] + '_'] = {}
        self.new_arcs['s_', self.staList[0] + '_'] = defaultdict(list)
        self.arcs['s_', self.staList[0] + '_'][-1] = {}  # source node流出弧, 只有t=-1，因为source node与时间无关

        for t in range(minArr, self.right_time_bound[self.v_staList[1]]):
            self.arcs['s_', self.staList[0] + '_'][-1][t] = Arc(self.traNo, 's_', self.staList[0] + '_', -1, t, 0)
            self.new_arcs['s_', self.staList[0] + '_'][-1 + t].append(self.arcs['s_', self.staList[0] + '_'][-1][t])
            # 声明弧长为t，实际length为0

        '''
        create arcs between real stations
        '''
        for i in range(len(self.staList) - 1):
            curSta = self.staList[i]
            nextSta = self.staList[i + 1]
            # virtual dual stations
            curSta_dep = curSta + "_"
            nextSta_arr = "_" + nextSta
            nextSta_dep = nextSta + "_"

            secRunTime = secTimes[curSta, nextSta]  # 区间运行时分

            # 创建两个弧, 一个运行弧，一个停站弧
            '''
            curSta_dep-->nextSta_arr区间运行弧
            '''
            self.arcs[curSta_dep, nextSta_arr] = {}
            self.new_arcs[curSta_dep, nextSta_arr] = defaultdict(list)

            if self.linePlan[curSta] == 1:  # 本站停车，加起车附加时分
                secRunTime += self.start_addTime[curSta]
            if self.linePlan[nextSta] == 1:  # 下站停车，加停车附加时分
                secRunTime += self.stop_addTime[nextSta]  # 添加停车附加时分
            # 设置d-a的区间运行弧
            for t in range(minArr, self.right_time_bound[curSta_dep]):
                if t + secRunTime >= self.right_time_bound[nextSta_arr]:  # 范围为0 => TimeSpan - 1
                    break
                self.arcs[curSta_dep, nextSta_arr][t] = {}  # dep-arr在node t的弧集，固定区间运行时分默认只有一个元素
                self.arcs[curSta_dep, nextSta_arr][t][secRunTime] = Arc(self.traNo, curSta_dep, nextSta_arr, t,
                                                                        t + secRunTime, secRunTime)
                self.new_arcs[curSta_dep, nextSta_arr][t + secRunTime].append(
                    self.arcs[curSta_dep, nextSta_arr][t][secRunTime])

            # update cur time window
            minArr += secRunTime

            '''
            nextSta_arr-->nextSta_dep车站停站弧
            '''
            if i + 1 == len(self.staList) - 1:  # 若停站, 但已经是最后一个站了，不需停站弧
                break

            self.arcs[nextSta_arr, nextSta_dep] = {}
            self.new_arcs[nextSta_arr, nextSta_dep] = defaultdict(list)
            if self.linePlan[nextSta] == 1:  # 该站停车，创建多个停站时间长度的停站弧
                for t in range(minArr, self.right_time_bound[nextSta_arr]):
                    if t + self.min_dwellTime[nextSta] >= self.right_time_bound[nextSta_dep]:  # 当前t加上最短停站时分都超了，break掉
                        break
                    self.arcs[nextSta_arr, nextSta_dep][t] = {}
                    my_range = range(self.min_dwellTime[nextSta], self.max_dwellTime[nextSta] + 1)
                    for span in my_range:
                        if t + span >= self.right_time_bound[nextSta_dep]:
                            break
                        self.arcs[nextSta_arr, nextSta_dep][t][span] = Arc(self.traNo, nextSta_arr, nextSta_dep, t,
                                                                           t + span, span)
                        self.new_arcs[nextSta_arr, nextSta_dep][t + span].append(
                            self.arcs[nextSta_arr, nextSta_dep][t][span])

            else:  # 该站不停车，只创建一个竖直弧，长度为0
                for t in range(minArr, self.right_time_bound[nextSta_arr]):
                    self.arcs[nextSta_arr, nextSta_dep][t] = {}
                    self.arcs[nextSta_arr, nextSta_dep][t][0] = Arc(self.traNo, nextSta_arr, nextSta_dep, t, t, 0)
                    self.new_arcs[nextSta_arr, nextSta_dep][t].append(self.arcs[nextSta_arr, nextSta_dep][t][0])

            # update cur time window
            minArr += self.min_dwellTime[nextSta]

        '''
        create arcs involving node t
        '''
        self.arcs['_' + self.staList[-1], '_t'] = {}
        self.new_arcs['_' + self.staList[-1], '_t'] = defaultdict(list)
        for t in range(minArr, self.right_time_bound[self.v_staList[-2]]):
            finale_name = '_' + self.staList[-1]
            self.arcs[finale_name, '_t'][t] = {}  # dep-arr在node t的弧集，固定区间运行时分默认只有一个元素
            self.arcs[finale_name, '_t'][t][0] = Arc(self.traNo, finale_name, '_t', t, -1, 0)
            self.new_arcs[finale_name, '_t'][t].append(self.arcs[finale_name, '_t'][t][0])

    def update_arc_multiplier_nx(self):
        """
        rebuild current price/multiplier to cal shortest path
        """
        subg = self.subgraph
        price = [(i, j, {'price': v['weight'] + xa_map[i, j][self.traNo][self.v_sta_type[j[0]]] * yvc_multiplier[j][self.v_sta_type[j[0]]]}) if j[0] not in ["s_", "_t"] else (i, j, {"price": v["weight"]}) for i, j, v in subg.edges(data=True)]
        subg.update(edges=price)
        self.max_edge_weight = self

    def update_primal_graph_nx(self, *args, **kwargs):
        """
        use occupied nodes to create a updated primal graph
            to compute primal feasible solutions
        """
        self.subgraph_primal = nx.DiGraph(self.subgraph.edges(data=True))

        occupied_nodes, arcs, incompatible_arcs, _safe_int = args
        # step 1,
        # you should remove the union of neighborhood of each node
        # i. get neighborhood size
        radius = {}
        for (i, t), c in occupied_nodes.items():
            _type_affix = c + self.v_sta_type.get(i, '%')
            if _type_affix in _safe_int:
                radius[i, t] = _safe_int[_type_affix][i.strip("_"), self.speed]
            else:
                radius[i, t] = 0  # move the center only.
        # ii. then remove nodes defined by radius
        _all_nodes = ((i, t + dlt) for (i, t), r in radius.items() for dlt in range(-r + 1, r))
        self.subgraph_primal.remove_nodes_from(_all_nodes)

        # step 2,
        # remove any incompatible arcs
        self.subgraph_primal.remove_edges_from(incompatible_arcs)

    def shortest_path_nx(self, option='dual'):
        """
        """
        i, j = self.source, self.sink
        _g = self.subgraph if option == 'dual' else self.subgraph_primal
        _price = 'price' if option == 'dual' else 'weight'
        try:
            ssp = nx.shortest_path(_g, i, j, weight=_price, method='bellman-ford')
            cost = nx.path_weight(_g, ssp, weight=_price)
        except Exception as e:
            # infeasible and unconnected case.
            # you are unable to create a shortest path.
            # self.logger.warning(e)
            # self.logger.warning(f"unable to compute for {self.traNo}: {option}")
            ssp = []
            cost = np.inf
            if option == "dual":
                raise e

        # compute cost
        return ssp, cost

    #################
    # iGraph
    #################
    def _ig_add_vertex_pair(self, _s, _t):
        """
        add vertex pair (an edge) to igraph backend.
        """
        if _t[0] not in ["s_", "_t"]:
            xa_map[_s, _t][self.traNo][self.v_sta_type[_t[0]]] += 1
        if self.backend == 0:
            return -1
        self._ig_nodes.add(_s)
        self._ig_nodes.add(_t)
        self._ig_edges[_s, _t] = max(0, _t[-1] - max(0, _s[-1]))

    def _init_subgraph_ig(self):
        _n_nodes = self._ig_nodes.__len__()
        self._ig_nodes_id = _node_id = dict(zip(self._ig_nodes, range(_n_nodes)))
        _edges = ((_node_id[i], _node_id[j]) for (i, j), v in self._ig_edges.items())
        _weight = list(self._ig_edges.values())
        _name = list(self._ig_edges.keys())
        self.subgraph = ig.Graph(
            directed=True,
            graph_attrs={"trainNo": self.traNo},
            n=_n_nodes,
            edges=_edges,
            edge_attrs={"weight": _weight, "name": _name},
            vertex_attrs={"name": self._ig_nodes}
        )

    def create_subgraph_ig(self, secTimes, TimeSpan):
        """
        create the subgraph for this train
            use the created nodelist
            you should not initialize any new nodes.
        """
        self.depSta = self.staList[0]
        self.arrSta = self.staList[-1]
        self.secTimes = secTimes
        self.truncate_train_time_bound(TimeSpan)

        minArr = self.dep_LB  # for curSta(judge by dep)

        for t in range(minArr, self.right_time_bound[self.v_staList[1]]):
            _s, _t = ('s_', -1), (self.staList[0] + '_', t)
            self._ig_add_vertex_pair(_s, _t)

        '''
        create arcs between real stations
        '''
        for i in range(len(self.staList) - 1):
            curSta = self.staList[i]
            nextSta = self.staList[i + 1]
            # virtual dual stations
            curSta_dep = curSta + "_"
            nextSta_arr = "_" + nextSta
            nextSta_dep = nextSta + "_"

            secRunTime = secTimes[curSta, nextSta]  # 区间运行时分

            # 创建两个弧, 一个运行弧，一个停站弧

            if self.linePlan[curSta] == 1:  # 本站停车，加起车附加时分
                secRunTime += self.start_addTime[curSta]
            if self.linePlan[nextSta] == 1:  # 下站停车，加停车附加时分
                secRunTime += self.stop_addTime[nextSta]  # 添加停车附加时分
            # 设置d-a的区间运行弧
            for t in range(minArr, self.right_time_bound[curSta_dep]):
                if t + secRunTime >= self.right_time_bound[nextSta_arr]:  # 范围为0 => TimeSpan - 1
                    break
                _s, _t = (curSta_dep, t), (nextSta_arr, t + secRunTime)
                self._ig_add_vertex_pair(_s, _t)

            # update cur time window
            minArr += secRunTime

            '''
            nextSta_arr-->nextSta_dep车站停站弧
            '''
            if i + 1 == len(self.staList) - 1:  # 若停站, 但已经是最后一个站了，不需停站弧
                break

            if self.linePlan[nextSta] == 1:  # 该站停车，创建多个停站时间长度的停站弧
                for t in range(minArr, self.right_time_bound[nextSta_arr]):
                    if t + self.min_dwellTime[nextSta] >= self.right_time_bound[nextSta_dep]:  # 当前t加上最短停站时分都超了，break掉
                        break
                    my_range = range(self.min_dwellTime[nextSta], self.max_dwellTime[nextSta] + 1)
                    for span in my_range:
                        if t + span >= self.right_time_bound[nextSta_dep]:
                            break
                        # igraph backend
                        _s, _t = (nextSta_arr, t), (nextSta_dep, t + span)
                        self._ig_add_vertex_pair(_s, _t)
            else:  # 该站不停车，只创建一个竖直弧，长度为0
                for t in range(minArr, self.right_time_bound[nextSta_arr]):
                    # igraph backend
                    _s, _t = (nextSta_arr, t), (nextSta_dep, t)
                    self._ig_add_vertex_pair(_s, _t)
            # update cur time window
            minArr += self.min_dwellTime[nextSta]

        '''
        create arcs involving node t
        '''

        for t in range(minArr, self.right_time_bound[self.v_staList[-2]]):
            finale_name = '_' + self.staList[-1]
            # igraph backend
            _s, _t = (finale_name, t), ('_t', -1)
            self._ig_add_vertex_pair(_s, _t)

        self._init_subgraph_ig()
        self._ig_s = 0
        self._ig_t = self.subgraph.vcount() - 1
        self.max_edge_weight = np.max(self.subgraph.es['weight'])

    def update_arc_multiplier_ig(self):
        """
        rebuild current price/multiplier to cal shortest path
        """
        subg = self.subgraph
        for e in subg.es:
            i, j = e['name']
            w = e['weight']
            p = w + xa_map[i, j][self.traNo][self.v_sta_type[j[0]]] * yvc_multiplier[j][self.v_sta_type[j[0]]] if j[0] not in ["s_", "_t"] else w
            # attr updates.
            e['price'] = p

    def update_primal_graph_ig(self, *args, **kwargs):
        """
        use occupied nodes to create a updated primal graph
            to compute primal feasible solutions
        """
        self.subgraph_primal = self.subgraph.copy()
        occupied_nodes, arcs, incompatible_arcs, _safe_int = args
        # step 1,
        # you should remove the union of neighborhood of each node
        # i. get neighborhood size
        radius = {}
        for (i, t), c in occupied_nodes.items():
            _type_affix = c + self.v_sta_type.get(i, '%')
            if _type_affix in _safe_int:
                radius[i, t] = _safe_int[_type_affix][i.strip("_"), self.speed]
            else:
                radius[i, t] = 0  # move the center only.
        # ii. then remove nodes defined by radius
        _all_nodes = ((i, t + dlt) for (i, t), r in radius.items() for dlt in range(-r, r + 1))
        _ig_all_nodes = [self._ig_nodes_id[n] for n in _all_nodes if n in self._ig_nodes]
        self.subgraph_primal.delete_vertices(_ig_all_nodes)
        self._ig_t_primal = self.subgraph_primal.vcount() - 1

    def shortest_path_ig(self, option='dual'):
        """
        """
        i, j = self.source, self.sink
        _g = self.subgraph if option == 'dual' else self.subgraph_primal
        _price = 'price' if option == 'dual' else 'weight'
        _sink = self._ig_t if option == 'dual' else self._ig_t_primal
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                ssp = _g.get_shortest_paths(self._ig_s, _sink, _price, mode=ig.OUT, output='vpath')[0]
                ssp_literal = [_g.vs[i]['name'] for i in ssp]
                # todo, fix this
                edges = _g.get_eids(path=ssp)
                cost = sum(_g.es[edges][_price])

            except Exception as e:
                # infeasible and unconnected case.
                # you are unable to create a shortest path.
                # self.logger.warning(e)
                # self.logger.warning(f"unable to compute for {self.traNo}: {option}")
                ssp_literal = []
                cost = np.inf
                if option == 'dual':
                    raise e

        # compute cost
        return ssp_literal, cost
