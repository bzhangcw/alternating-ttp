from typing import List, Tuple, Union


graph_tool = "networkx"
if graph_tool == "igraph":
    import igraph as ig
elif graph_tool == "networkx":
    import networkx as nx
else:
    raise ValueError("graph_tool must be either igraph or networkx")


class PathPoolManager:
    def __init__(self, train_list: List, safe_int: dict):
        self.safe_int = safe_int
        self.path_ids = dict()
        self.train_list = train_list
        self.train_ids_dict = {tr.traNo: tr for tr in self.train_list}
        self.path_pool = {train_id: [] for train_id in self.train_ids_dict}
        self.path_map = {}
        if graph_tool == "igraph":
            self.graph = ig.Graph(directed=False)
        else:
            self.graph = nx.Graph()

    def add_path(self, train_id: int, path: Union[List, Tuple]):
        path = tuple(path)
        path_id = len(self.path_ids)  # path_id as the order of adding the path
        if (train_id, path) in self.path_ids:
            # print(train_id, hash(path), "tuple already in the graph")
            return
        self.path_ids[train_id, path] = path_id
        self.path_pool[train_id].append(path_id)
        self.path_map[path_id] = path

        if graph_tool == "igraph":
            self.graph.add_vertex(path_id)
        else:
            self.graph.add_node(path_id)
        self.add_edge_with_conflict(train_id, path)

    def add_edge_with_conflict(self, train_id, path: Tuple):
        path_id = self.path_ids[train_id, path]
        train1 = self.train_ids_dict[train_id]
        for path2_train_id, path2_pool in self.path_pool.items():
            train2 = self.train_ids_dict[path2_train_id]
            for path2_id in path2_pool:
                path2 = self.path_map[path2_id]
                if (path_id != path2_id) and (
                        train_id == path2_train_id or
                        self.pairwise_path_conflict(train1, train2, path, path2)):  # in the same pool
                    self.graph.add_edge(path_id, path2_id)

    def maximal_independent_vertex_sets(self):
        if graph_tool == "igraph":
            return self.graph.maximal_independent_vertex_sets()
        else:
            return nx.maximal_independent_set(self.graph)

    def largest_independent_vertex_sets(self):
        if graph_tool == "igraph":
            return self.graph.largest_independent_vertex_sets()
        else:
            return nx.algorithms.approximation.maximum_independent_set(self.graph)

    def pairwise_path_conflict(self, train1, train2, path1, path2) -> bool:
        path1_start_station = path1[1][0]
        path2_start_station = path2[1][0]
        assert path1_start_station != "s_" and path2_start_station != "s_"
        common_start_station = path1_start_station \
            if PathPoolManager.station_geq(path1_start_station, path2_start_station) \
            else path2_start_station

        path1_end_station = path1[-2][0]
        path2_end_station = path2[-2][0]
        assert path1_end_station != "_t" and path2_end_station != "_t"
        common_end_station = path1_end_station \
            if PathPoolManager.station_geq(path2_end_station, path1_end_station) \
            else path2_end_station

        new_path1 = [sp_node for sp_node in path1[1:-1] if PathPoolManager.station_geq(sp_node[0], common_start_station)
                     and PathPoolManager.station_geq(common_end_station, sp_node[0])]
        new_path2 = [sp_node for sp_node in path2[1:-1] if PathPoolManager.station_geq(sp_node[0], common_start_station)
                     and PathPoolManager.station_geq(common_end_station, sp_node[0])]
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

    @staticmethod
    def station_geq(s1: str, s2: str):
        s1_ = int(s1.replace("_", ""))
        s2_ = int(s2.replace("_", ""))
        return s1_ >= s2_
