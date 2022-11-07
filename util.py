"""
utility modules, include utility for subgradient method.
"""
import datetime
import logging
import os
import sys
from collections import defaultdict
from itertools import combinations

from gurobipy import GRB

from util_solver import getConstrByPrefix

##############################
# DEFAULTS
##############################
NODE_SINK = 't'
NODE_SINK_ARR = '_t'
##############################
# package-wise global variables
##############################
# flattened yv2xa
# (s', t', s, t) arc : value
xa_map = defaultdict(lambda: defaultdict(lambda: {"a": 0, "s": 0, "p": 0}))
# node precedence map in terms of arrival/departure interval
node_prec_map = defaultdict(list)
# original Lagrangian
# multiplier = defaultdict(lambda: {"aa": 0, "ap": 0, "ss": 0, "sp": 0, "pa": 0, "ps": 0, "pp": 0})  # each (station, t)
multiplier = dict()  # each (station, t)
z_vars = dict()
# node multiplier
yv_multiplier = {}  # the multiplier of each v
yvc_multiplier = defaultdict(lambda: {"a": 0, "s": 0, "p": 0})  # the multiplier of each v with type c
safe_int = {}
# category list
category = ["s", "a", "p"]
safe_int_df = None

##############################
# LOGGER
##############################
_grb_logger = logging.getLogger("gurobipy.gurobipy")
_grb_logger.setLevel(logging.ERROR)

logFormatter = logging.Formatter("%(asctime)s: %(message)s")
logger = logging.getLogger("railway")
logger.setLevel(logging.INFO)

# consoleHandler = logging.StreamHandler(sys.stdout)
# consoleHandler.setFormatter(logFormatter)
# logger.addHandler(consoleHandler)


# global graph generation id.
class GraphCounter(object):
    def __init__(self):
        # number of nodes (unique) created
        self.vc = 0
        # number of edges (unique) created (not used yet)
        self.ec = 0
        self.id_nodes = {}
        self.tuple_id = {}


gc = GraphCounter()


class SysParams(object):
    # todo, support ArgumentParser
    DBG = False
    station_size = 0
    train_size = 0
    time_span = 0
    iter_max = 0
    up = 0

    def __init__(self):
        subdir_result = self.subdir_result = datetime.datetime.now().strftime('%y%m%d-%H%M')
        fdir_result = self.fdir_result = f"result/{subdir_result}"
        os.makedirs(fdir_result, exist_ok=True)
        fileHandler = logging.FileHandler("{0}/{1}.log".format(fdir_result, "out"))
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)


    def parse_environ(self):
        import os
        self.station_size = int(os.environ.get('station_size', 29))
        self.train_size = int(os.environ.get('train_size', 50))
        self.time_span = int(os.environ.get('time_span', 1080))
        self.iter_max = int(os.environ.get('iter_max', 100))
        self.up = int(os.environ.get('up', 0))
        self.log_problem_size(logger)

    def log_problem_size(self, logger):
        logger.info(
            f"size: #train,#station,#timespan,#iter_max: {self.train_size, self.station_size, self.time_span, self.iter_max}"
        )


# subgradient params
class SubgradParam(object):

    def __init__(self):
        self.kappa = 0.2
        self.alpha = 1.0
        self.beta = 1
        self.gamma = 0.1  # parameter for argmin x
        self.changed = 0
        self.num_stuck = 0
        self.eps_num_stuck = 3
        self.iter = 0
        self.lb = 1e-6
        self.lb_arr = []
        self.ub_arr = []
        self.gap = 1
        self.dual_method = "pdhg"  # "lagrange" or "pdhg"
        self.primal_heuristic_method = "seq"  # "jsp" or "seq"
        self.feasible_provider = "seq"  # "jsp" or "seq"
        self.max_number = 1
        self.norms = ([], [], [])
        self.multipliers = ([], [], [])

    def parse_environ(self):
        import os
        self.primal_heuristic_method = os.environ.get('primal', 'seq')
        self.dual_method = os.environ.get('dual', 'pdhg_alm')

    def update_bound(self, lb):
        if lb >= self.lb:
            self.lb = lb
            self.changed = 1
            self.num_stuck = 0
        else:
            self.changed = 0
            self.num_stuck += 1

        if self.num_stuck >= self.eps_num_stuck:
            self.kappa *= 0.5
            self.num_stuck = 0
        self.lb_arr.append(lb)

    def update_incumbent(self, ub):
        self.ub_arr.append(ub)

    def update_gap(self):
        _best_ub = min(self.ub_arr)
        _best_lb = max(self.lb_arr)
        self.gap = (_best_ub - _best_lb) / (abs(_best_lb) + 1e-3)

    def reset(self):
        self.num_stuck = 0
        self.eps_num_stuck = 3
        self.iter = 0
        self.lb = 1e-6
        self.lb_arr = []
        self.ub_arr = []
        self.gap = 1
        self.dual_method = "pdhg"  # "lagrange" or "pdhg"
        self.primal_heuristic_method = "jsp"  # "jsp" or "seq"
        self.feasible_provider = "jsp"  # "jsp" or "seq"
        self.max_number = 1
        self.norms = ([], [], [])  # l1-norm, l2-norm, infty-norm
        self.multipliers = ([], [], [])
        self.parse_environ()


def from_train_path_to_train_order(train_list, method="dual"):
    overtaking_dict = {}
    for train_1, train_2 in combinations(train_list, 2):
        if LR_path_overtaking(train_1, train_2, method):
            overtaking_dict[train_1, train_2] = True
    train_order = defaultdict(list)
    for train in train_list:
        if method == "dual":
            for sta, t in train.opt_path_LR[1:-1]:
                train_order[sta].append((train, t))
        elif method == "primal":
            if train.is_feasible:
                for sta, t in train.feasible_path[1:-1]:
                    train_order[sta].append((train, t))
            # else:
            #     for sta, t in train.opt_path_LR[1:-1]:
            #         train_order[sta].append((train, t))
        else:
            raise TypeError(f"method {method} is wrong")
    train_order = dict(train_order)
    for sta in train_order.keys():
        train_order[sta].sort(key=lambda x: x[1])

    return train_order, overtaking_dict


def fix_train_order_at_station(model, train_order, safe_int, overtaking_dict, theta):
    for v_station, order in train_order.items():
        station = v_station.replace("_", "")
        for i, (train, t) in enumerate(order):
            if v_station.endswith("_"):  # only consider dp dd pd pp
                if train.v_sta_type[v_station] == "s":
                    # dd
                    model.addConstrs(
                        (theta['dd'][station][train][train_after] == 1 for train_after, t_after in order[i:]
                         if train_after.v_sta_type[v_station] == "s"
                         and (train, train_after) not in overtaking_dict
                         and t_after - t >= safe_int["ss"][station, train_after.speed]),
                        name=f"headway_fix_dd[{train}]")
                    # dp
                    model.addConstrs(
                        (theta['dp'][station][train][train_after] == 1 for train_after, t_after in order[i:]
                         if train_after.v_sta_type[v_station] == "p"
                         and (train, train_after) not in overtaking_dict
                         and t_after - t >= safe_int["sp"][station, train_after.speed]),
                        name=f"headway_fix_dp[{train}]")
                elif train.v_sta_type[v_station] == "p":
                    # pd
                    model.addConstrs(
                        (theta['pd'][station][train][train_after] == 1 for train_after, t_after in order[i:]
                         if train_after.v_sta_type[v_station] == "s"
                         and (train, train_after) not in overtaking_dict
                         and t_after - t >= safe_int["ps"][station, train_after.speed]),
                        name=f"headway_fix_pd[{train}]")
                else:
                    raise TypeError(f"train:{train} has the wrong virtual station type: {train.v_sta_type[v_station]}")
            elif v_station.startswith("_"):  # only consider ap aa pa
                if train.v_sta_type[v_station] == "a":
                    # aa
                    model.addConstrs(
                        (theta['aa'][station][train][train_after] == 1 for train_after, t_after in order[i:]
                         if train_after.v_sta_type[v_station] == "a"
                         and (train, train_after) not in overtaking_dict
                         and t_after - t >= safe_int["aa"][station, train_after.speed]),
                        name=f"headway_fix_aa[{train}]")
                    # ap
                    model.addConstrs(
                        (theta['ap'][station][train][train_after] == 1 for train_after, t_after in order[i:]
                         if train_after.v_sta_type[v_station] == "p"
                         and (train, train_after) not in overtaking_dict
                         and t_after - t >= safe_int["ap"][station, train_after.speed]),
                        name=f"headway_fix_ap[{train}]")
                elif train.v_sta_type[v_station] == "p":
                    # pa
                    model.addConstrs(
                        (theta['pa'][station][train][train_after] == 1 for train_after, t_after in order[i:]
                         if train_after.v_sta_type[v_station] == "a"
                         and (train, train_after) not in overtaking_dict
                         and t_after - t >= safe_int["pa"][station, train_after.speed]),
                        name=f"headway_fix_pa[{train}]")
                    # pp
                    model.addConstrs(
                        (theta['pp'][station][train][train_after] == 1 for train_after, t_after in order[i:]
                         if train_after.v_sta_type[v_station] == "p"
                         and (train, train_after) not in overtaking_dict
                         and t_after - t >= safe_int["pp"][station, train_after.speed]),
                        name=f"headway_fix_pp[{train}]")
                else:
                    raise TypeError(f"train:{train} has the wrong virtual station type: {train.v_sta_type[v_station]}")
            else:
                raise TypeError(f"virtual station:{v_station} has the wrong type")


def fix_train_at_station(model, x_var, feasible_train_list):
    model.update()
    x_var_feas = [x_var[train] for train in feasible_train_list]
    return model.addConstrs((x_i == 1 for x_i in x_var_feas), name="feasible_fix_x")


def IIS_resolve(model, iter_max=5):
    headway_fix_constrs = getConstrByPrefix(model, "headway_fix")
    iter = 0
    while iter < iter_max and model.status == GRB.INFEASIBLE:
        model.computeIIS()
        zipped = [(i, constr) for i, constr in enumerate(headway_fix_constrs) if constr.IISConstr]
        remove_indice, incompatible_constrs = zip(*zipped)
        model.remove(incompatible_constrs)
        for i in remove_indice:
            headway_fix_constrs.pop(i)
        model.optimize()
        iter += 1


def LR_path_overtaking(train_1, train_2, method="dual"):
    if method == "dual":
        path_1 = train_1.opt_path_LR
        path_2 = train_2.opt_path_LR
    elif method == "primal":
        path_1 = train_1.feasible_path if train_1.is_feasible else train_1.opt_path_LR
        path_2 = train_2.feasible_path if train_2.is_feasible else train_2.opt_path_LR
    else:
        raise ValueError(f"method {method} is not supported")
    max_dep = max(int(train_1.depSta), int(train_2.depSta))
    min_arr = min(int(train_1.arrSta), int(train_2.arrSta))
    if max_dep >= min_arr:
        return False
    train_1_path_LR = [elem for elem in path_1[1:-1] if max_dep <= int(elem[0].replace("_", "")) <= min_arr]
    train_2_path_LR = [elem for elem in path_2[1:-1] if max_dep <= int(elem[0].replace("_", "")) <= min_arr]
    if train_2_path_LR[0][0].startswith("_"):
        train_2_path_LR.pop(0)
    if train_2_path_LR[-1][0].endswith("_"):
        train_2_path_LR.pop(-1)
    if train_1_path_LR[0][0].startswith("_"):
        train_1_path_LR.pop(0)
    if train_1_path_LR[-1][0].endswith("_"):
        train_1_path_LR.pop(-1)
    assert all(node_trn_1[0] == node_trn_2[0] for node_trn_1, node_trn_2 in zip(train_1_path_LR, train_2_path_LR))

    for i, (node_trn_1, node_trn_2) in enumerate(zip(train_1_path_LR[:-1], train_2_path_LR[:-1])):
        if node_trn_1[0].endswith("_"):
            assert node_trn_2[0].endswith("_")
            node_next_trn_1 = train_1_path_LR[i + 1]
            node_next_trn_2 = train_2_path_LR[i + 1]
            if (node_trn_1[1] <= node_trn_2[1] and node_next_trn_1[1] >= node_next_trn_2[1]) \
                    or (node_trn_1[1] >= node_trn_2[1] and node_next_trn_1[1] <= node_next_trn_2[1]):
                return True

    return False
