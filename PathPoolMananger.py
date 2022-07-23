from typing import List, Tuple, Union

# graph_tool = "networkx"
graph_tool = "igraph"
if graph_tool == "igraph":
    import igraph as ig
elif graph_tool == "networkx":
    import networkx as nx
else:
    raise ValueError("graph_tool must be either igraph or networkx")


class PathPoolManager:
    def __init__(self, train_list: List, safe_int: dict, up: bool, miv_mode: int = 1):
        self.safe_int = safe_int
        self.up = up
        self.to_path_ids = dict()
        self.train_list = train_list
        self.train_ids_dict = {tr.traNo: tr for tr in self.train_list}
        self.path_pool = {train_id: [] for train_id in self.train_ids_dict}
        self.inverse_path_ids = {}
        self.miv_mode = miv_mode
        if graph_tool == "igraph":
            self.graph = ig.Graph(directed=False)
        else:
            self.graph = nx.Graph()

    def add_path(self, train_id: int, path: Union[List, Tuple]):
        path = tuple(path)
        path_id = len(self.to_path_ids)  # path_id as the order of adding the path
        if (train_id, path) in self.to_path_ids:
            # print(train_id, hash(path), "tuple already in the graph")
            return
        self.to_path_ids[train_id, path] = path_id
        self.path_pool[train_id].append(path_id)
        self.inverse_path_ids[path_id] = train_id, path

        if graph_tool == "igraph":
            self.graph.add_vertex(path_id)
        else:
            self.graph.add_node(path_id)
        self.add_edge_with_conflict(train_id, path)

    def add_edge_with_conflict(self, train_id, path: Tuple):
        path_id = self.to_path_ids[train_id, path]
        train1 = self.train_ids_dict[train_id]

        conflict_path_pairs = []
        for train2_id, tr2_path_pool in self.path_pool.items():
            train2 = self.train_ids_dict[train2_id]
            for path2_id in tr2_path_pool:
                _, path2 = self.inverse_path_ids[path2_id]
                if (path_id != path2_id) and (
                        train_id == train2_id or
                        self.pairwise_path_conflict(train1, train2, path, path2)):  # in the same pool
                    conflict_path_pairs.append((path_id, path2_id))
        for path1_id, path2_id in conflict_path_pairs:
            self.graph.add_edge(path1_id, path2_id)

    def maximal_independent_vertex_sets(self):
        if graph_tool == "igraph":
            return self.graph.maximal_independent_vertex_sets()
        else:
            return nx.maximal_independent_set(self.graph)

    def largest_independent_vertex_sets(self):
        if graph_tool == "igraph":
            # return self.graph.largest_independent_vertex_sets()
            return self.maximal_ivs_with_starts()
        else:
            return nx.algorithms.approximation.maximum_independent_set(self.graph)

    def maximal_ivs_with_starts(self):
        if self.miv_mode == 0:
            # find best path for each train
            vpool = []
            for train_id, ppool in self.path_pool.items():
                pp = sorted(ppool, key=lambda x: self.inverse_path_ids[x][1][-2][-1])
                vpool.append(pp[0])

            _maximum_value = 0
            mis = None
            for v in vpool:
                g = self.graph.copy()
                _v = g.vs.select(name=v)[0]
                _mis = [_v['name']]
                g.delete_vertices([_v, *_v.neighbors()])
                _mis.extend(self._mivs(g))
                _size = len(_mis)
                if _size > _maximum_value:
                    mis = _mis
                    _maximum_value = _size
            assert len(mis) <= len(self.train_list)
            return tuple(mis)
        else:
            # find best path without each train
            mis = []
            _maximum_value = 0
            mis_best_dict = {}
            for train in self.train_list:
                if train.is_feasible:
                    path_id = self.to_path_ids[train.traNo, tuple(train.feasible_path)]
                    mis_best_dict[train.traNo] = path_id
                    _maximum_value += 1
                    mis.append(path_id)
            if _maximum_value < len(self.train_list):
                g = self.graph.copy()
                v_list = list(mis_best_dict.values())
                _mis = self._mivs_for_v_list(g, v_list)
                _size = len(_mis)
                if _size > _maximum_value:
                    mis = _mis
                    _maximum_value = _size
                    if _maximum_value == len(self.train_list):
                        return tuple(mis)

                for train in self.train_list:
                    if train.is_feasible:
                        g = self.graph.copy()
                        path_id = mis_best_dict.pop(train.traNo)
                        v_list = list(mis_best_dict.values())
                        mis_best_dict[train.traNo] = path_id
                        _mis = self._mivs_for_v_list(g, v_list)
                        _size = len(_mis)
                        if _size > _maximum_value:
                            mis = _mis
                            _maximum_value = _size
                            if _maximum_value == len(self.train_list):
                                break
            assert len(mis) <= len(self.train_list)
            return tuple(mis)

    @staticmethod
    def _mivs(g):
        # sequential clique deletion algorithm.
        # cf. https://en.wikipedia.org/wiki/Maximal_independent_set#Finding_a_single_maximal_independent_set
        mis = []
        while g.vcount():
            _v = g.vs[0]
            mis.append(_v['name'])
            g.delete_vertices([_v, *_v.neighbors()])
        return sorted(mis)

    @staticmethod
    def _mivs_for_v_list(g, v_list):
        # sequential clique deletion algorithm.
        # cf. https://en.wikipedia.org/wiki/Maximal_independent_set#Finding_a_single_maximal_independent_set
        mis = []
        while g.vcount():
            if len(v_list) > 0:
                v = v_list.pop(0)
                _v = g.vs.select(name=v)[0]
            else:
                _v = g.vs[0]
            mis.append(_v['name'])
            g.delete_vertices([_v, *_v.neighbors()])
        return tuple(sorted(mis))

    @staticmethod
    def _mivs_without_vs(g, v_list):
        g.delete_vertices(v_list)
        mis = []
        while g.vcount():
            _v = g.vs[0]
            mis.append(_v['name'])
            g.delete_vertices([_v, *_v.neighbors()])
        return mis

    def pairwise_path_conflict(self, train1, train2, path1, path2) -> bool:
        path1_start_station = path1[1][0]
        path2_start_station = path2[1][0]
        assert path1_start_station != "s_" and path2_start_station != "s_"
        common_start_station = path1_start_station \
            if self.station_rear(path1_start_station, path2_start_station) \
            else path2_start_station

        path1_end_station = path1[-2][0]
        path2_end_station = path2[-2][0]
        assert path1_end_station != "_t" and path2_end_station != "_t"
        common_end_station = path1_end_station \
            if self.station_rear(path2_end_station, path1_end_station) \
            else path2_end_station

        if self.station_rear(common_start_station, common_end_station):
            return False

        # get common station list
        path1_start_index = 0
        for i in range(1, len(path1) - 1):
            if self.station_rear(path1[i][0], common_start_station):
                path1_start_index = i
                break
        path1_end_index = len(path1) - 1
        for i in range(len(path1) - 2, 0, -1):
            if self.station_rear(common_end_station, path1[i][0]):
                path1_end_index = i
                break
        assert path1_start_index > 0 and path1_end_index < len(path1) - 1
        new_path1 = list(path1[path1_start_index:path1_end_index + 1])

        path2_start_index = 0
        for i in range(1, len(path2) - 1):
            if self.station_rear(path2[i][0], common_start_station):
                path2_start_index = i
                break
        path2_end_index = len(path2) - 1
        for i in range(len(path2) - 2, 0, -1):
            if self.station_rear(common_end_station, path2[i][0]):
                path2_end_index = i
                break
        assert path2_start_index > 0 and path2_end_index < len(path2) - 1
        new_path2 = list(path2[path2_start_index:path2_end_index + 1])

        if len(new_path1) > 0:
            if new_path1[0][0].startswith("_"):
                new_path1.pop(0)
            if new_path2[0][0].startswith("_"):
                new_path2.pop(0)
            if len(new_path1) > 0 and new_path1[-1][0].endswith("_"):
                new_path1.pop(-1)
            if len(new_path2) > 0 and new_path2[-1][0].endswith("_"):
                new_path2.pop(-1)
            assert len(new_path1) == len(new_path2)
        else:
            assert len(new_path1) == len(new_path2)
            return False

        for node1, node2 in zip(new_path1, new_path2):
            assert node1[0] == node2[0]

            v_sta = node1[0]
            sta = v_sta.replace("_", "")
            front_train, rear_train = (train2, train1) if node1[1] >= node2[1] else (train1, train2)

            headway = self.safe_int[front_train.v_sta_type[v_sta] + rear_train.v_sta_type[v_sta]][sta, rear_train.speed]
            if abs(node1[1] - node2[1]) < headway:  # violation
                return True
        return False

    def station_rear(self, s1: str, s2: str):
        if self.up:
            return self.station_leq(s1, s2)
        else:
            return self.station_geq(s1, s2)

    @staticmethod
    def station_geq(s1: str, s2: str):
        s1_ = int(s1.replace("_", ""))
        s2_ = int(s2.replace("_", ""))
        return s1_ >= s2_

    @staticmethod
    def station_leq(s1: str, s2: str):
        s1_ = int(s1.replace("_", ""))
        s2_ = int(s2.replace("_", ""))
        return s1_ <= s2_
