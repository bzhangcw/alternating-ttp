"""
utility modules
"""
from collections import defaultdict
from itertools import combinations
from gurobipy import GRB

from constr_verification import getConstrByPrefix

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
xa_map = defaultdict(lambda: defaultdict(int))
xa_map_dbg = defaultdict(lambda: defaultdict(int))
# node precedence map in terms of arrival/departure interval
node_prec_map = defaultdict(list)
# original Lagrangian
multiplier = defaultdict(int)  # each (station, t)
# node multiplier
yv_multiplier = {}  # the multiplier of each v
safe_int = {}
# arrival-arrival headway and departure-departure headway
eps = 5


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


# subgradient params
class SubgradParam(object):

    def __init__(self):
        self.kappa = 0.2
        self.alpha = 1.0
        self.changed = 0
        self.num_stuck = 0
        self.eps_num_stuck = 3
        self.iter = 0
        self.lb = 1e-6
        self.gap = 1

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


def from_train_path_to_train_order(train_list, method="dual"):
    overtaking_dict = {}
    for train_1, train_2 in combinations(train_list, 2):
        if LR_path_overtaking(train_1, train_2, method):
            overtaking_dict[train_1.traNo, train_2.traNo] = True
    train_order = defaultdict(list)
    for train in train_list:
        if method == "dual":
            for sta, t in train.opt_path_LR[1:-1]:
                train_order[sta].append((train, t))
        elif method == "primal":
            if train.is_feasible:
                for sta, t in train.feasible_path[1:-1]:
                    train_order[sta].append((train, t))
            else:
                for sta, t in train.opt_path_LR[1:-1]:
                    train_order[sta].append((train, t))
        else:
            raise TypeError(f"method {method} is wrong")
    train_order = dict(train_order)
    for sta in train_order.keys():
        train_order[sta].sort(key=lambda x: x[1])

    return train_order, overtaking_dict


def fix_train_order_at_station(model, train_order, safe_int, overtaking_dict, theta_aa, theta_ap, theta_pa, theta_pp, theta_dd, theta_dp, theta_pd):
    for v_station, order in train_order.items():
        station = v_station.replace("_", "")
        for i, (train, t) in enumerate(order):
            if v_station.endswith("_"):  # only consider dp dd pd pp
                if train.v_sta_type[v_station] == "s":
                    # dd
                    model.addConstrs((theta_dd[station][train.traNo][train_after.traNo] == 1 for train_after, t_after in order[i:]
                                      if train_after.v_sta_type[v_station] == "s"
                                      and (train.traNo, train_after.traNo) not in overtaking_dict
                                      and t_after - t >= safe_int["ss"][station, train_after.speed]), name=f"headway_fix_dd[{train.traNo}]")
                    # dp
                    model.addConstrs((theta_dp[station][train.traNo][train_after.traNo] == 1 for train_after, t_after in order[i:]
                                      if train_after.v_sta_type[v_station] == "p"
                                      and (train.traNo, train_after.traNo) not in overtaking_dict
                                      and t_after - t >= safe_int["sp"][station, train_after.speed]), name=f"headway_fix_dp[{train.traNo}]")
                elif train.v_sta_type[v_station] == "p":
                    # pd
                    model.addConstrs((theta_pd[station][train.traNo][train_after.traNo] == 1 for train_after, t_after in order[i:]
                                      if train_after.v_sta_type[v_station] == "s"
                                      and (train.traNo, train_after.traNo) not in overtaking_dict
                                      and t_after - t >= safe_int["ps"][station, train_after.speed]), name=f"headway_fix_pd[{train.traNo}]")
                else:
                    raise TypeError(f"train:{train} has the wrong virtual station type: {train.v_sta_type[v_station]}")
            elif v_station.startswith("_"):  # only consider ap aa pa
                if train.v_sta_type[v_station] == "a":
                    # aa
                    model.addConstrs((theta_aa[station][train.traNo][train_after.traNo] == 1 for train_after, t_after in order[i:]
                                      if train_after.v_sta_type[v_station] == "a"
                                      and (train.traNo, train_after.traNo) not in overtaking_dict
                                      and t_after - t >= safe_int["aa"][station, train_after.speed]), name=f"headway_fix_aa[{train.traNo}]")
                    # ap
                    model.addConstrs((theta_ap[station][train.traNo][train_after.traNo] == 1 for train_after, t_after in order[i:]
                                      if train_after.v_sta_type[v_station] == "p"
                                      and (train.traNo, train_after.traNo) not in overtaking_dict
                                      and t_after - t >= safe_int["ap"][station, train_after.speed]), name=f"headway_fix_ap[{train.traNo}]")
                elif train.v_sta_type[v_station] == "p":
                    # pa
                    model.addConstrs((theta_pa[station][train.traNo][train_after.traNo] == 1 for train_after, t_after in order[i:]
                                      if train_after.v_sta_type[v_station] == "a"
                                      and (train.traNo, train_after.traNo) not in overtaking_dict
                                      and t_after - t >= safe_int["pa"][station, train_after.speed]), name=f"headway_fix_pa[{train.traNo}]")
                    # pp
                    model.addConstrs((theta_pp[station][train.traNo][train_after.traNo] == 1 for train_after, t_after in order[i:]
                                      if train_after.v_sta_type[v_station] == "p"
                                      and (train.traNo, train_after.traNo) not in overtaking_dict
                                      and t_after - t >= safe_int["pp"][station, train_after.speed]), name=f"headway_fix_pp[{train.traNo}]")
                else:
                    raise TypeError(f"train:{train} has the wrong virtual station type: {train.v_sta_type[v_station]}")
            else:
                raise TypeError(f"virtual station:{v_station} has the wrong type")


def fix_train_at_station(model, x_var, feasible_train_list):
    model.update()
    x_var_feas = [x_var[train.traNo] for train in feasible_train_list]
    return model.addConstrs((x_i == 1 for x_i in x_var_feas), name="feasible_fix_x")


def IIS_resolve(model, iter_max=30):
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
