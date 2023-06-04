"""
functional interface module for bcd
% consider the model:
%   min c'x
%     s.t. Ax<=b, Bx<=d, x \in {0,1}
%       - A: binding part
%       - B: block diagonal decomposed part
% ALM:
%   min c'x+rho*\|max{Ax-b+lambda/rho,0}\|^2
%     s.t. Bx<=d, x \in {0,1}
% implement the BCD to solve ALM (inc. indefinite proximal version),
% - ordinary linearized proximal BCD
% - indefinite proximal BCD which includes an extrapolation step.
% - restart utilities
"""
import copy
import functools
from typing import Dict
import time
import numpy as np
import scipy
import scipy.sparse.linalg as ssl
import tqdm
from gurobipy import *

import data as ms
import util_output as uo
import util_solver as su
from Train import *


# BCD params
class BCDParams(object):

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
        self.primal_heuristic_method = "jsp"  # "jsp" or "seq"
        self.feasible_provider = "jsp"  # "jsp" or "seq"
        self.sspbackend = "grb"
        self.dualobjtype = 1
        self.verbosity = 2
        self.max_number = 1
        self.norms = ([], [], [])
        self.multipliers = ([], [], [])
        self.itermax = 200
        self.linmax = 1

    def parse_environ(self):
        import os
        self.primal_heuristic_method = os.environ.get('primal', 'jsp')
        self.dual_method = os.environ.get('dual', 'pdhg_alm')
        self.sspbackend = os.environ.get('sspbackend', 'grb')
        self.dualobjtype = int(os.environ.get('dualobj', 1))
        self.verbosity = int(os.environ.get('verbosity', 1))

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

    def show_log_header(self):
        headers = ["k", "t", "f*", "c'x", "L(λ)", "aL(λ)", "|Ax - b|", "error", "ρ", "τ", "kl", "kls"]
        slots = ["{:^3s}", "{:^7s}", "{:^9s}", "{:^9s}", "{:^9s}", "{:^9s}", "{:^10s}", "{:^10s}", "{:^9s}", "{:^9s}",
                 "{:4s}",
                 "{:4s}"]
        _log_header = " ".join(slots).format(*headers)
        lt = _log_header.__len__()
        print("*" * lt)
        print(("{:^" + f"{lt}" + "}").format("BCD for MILP"))
        print(("{:^" + f"{lt}" + "}").format("(c) Chuwen Zhang, Shanwen Pu, Rui Wang"))
        print(("{:^" + f"{lt}" + "}").format("2022"))
        print("*" * lt)
        print(("{:^" + f"{lt}" + "}").format(f"backend: {self.sspbackend}"))
        print("*" * lt)
        print(_log_header)
        print("*" * lt)


def _Ax(block, x):
    return block['A'] @ x


# @np.vectorize
# def _nonnegative(x):
#     return max(x, 0)
def _nonnegative(x):
    a = x >= 0
    return x * a


class Result(object):
    # last iterate
    xk = None
    cx = None
    # best iterate
    xb = None
    cb = None
    # average iterate
    xv = None


def optimize(bcdpar: BCDParams, mat_dict: Dict):
    """

    Args:
        bcdpar: BCDParam
        mat_dict:  matlab dict storing bcd-styled ttp instance

    Returns:

    """
    # data
    start = time.time()
    blocks = mat_dict['trains']
    b = mat_dict['b']
    m, _ = b.shape
    # A = scipy.sparse.hstack([blk['A'] for idx, blk in enumerate(blocks)])
    # A1 = blocks[0]['A']
    # Anorm = scipy.sparse.linalg.norm(A1)
    # tau0 = 1 / (Anorm * rho)
    # Anorm = np.linalg.norm(A1)

    # alias
    rho = rho0 = 6e-1
    tau = tau0 = 1e-2
    tsig = 1
    sigma = 1.2
    cb = 1e6
    cx = 1e6
    np.random.seed(1)
    xb = [np.zeros((blk['n'], 1)) for idx, blk in enumerate(blocks)]
    xk = [np.random.random((blk['n'], 1)) for idx, blk in enumerate(blocks)]
    xv = [np.zeros((blk['n'], 1)) for idx, blk in enumerate(blocks)]
    lbd = rho * np.random.random(b.shape)
    r = Result()
    r.cx, r.cb, r.xb, r.xv, r.xk = cx, cb, xb, xv, xk
    # logger
    # - k: outer iteration num
    # - it: inner iteration num
    # - idx: 1-n block idx
    #       it may not be the train no
    # A_k x_k
    _vAx = {idx: blk['A'] @ xk[idx] for idx, blk in enumerate(blocks)}
    _Ax  = sum(_vAx.values())
    # c_k x_k
    _vcx = {idx: (blk['c'].T @ xk[idx]).trace() for idx, blk in enumerate(blocks)}
    # x_k - x_k* (fixed point error)
    _eps_fix_point = {idx: 0 for idx, blk in enumerate(blocks)}
    _xnorm = {idx: 0 for idx, _ in enumerate(blocks)}
    _grad = {idx: 0 for idx, _ in enumerate(blocks)}
    if bcdpar.sspbackend == 'grb':
        print("creating models")
        for idx, blk in enumerate(blocks):
            train: Train = blk['train']
            _c = blk['c']
            blk['mx'] = train.create_shortest_path_model(blk)
        print("creating models finished")

    alobj = (
            sum(_vcx.values())
            + (lbd.T * (sum(_vAx.values()) - b)).sum()
            + (_nonnegative(sum(_vAx.values()) - b) ** 2).sum() * rho / 2
    )

    bcdpar.show_log_header()
    kc = 0
    for k in range(bcdpar.itermax):
        for it in range(bcdpar.linmax):
            # linearization loop
            # idx: A[idx]@x[idx]
            tau = tau0
            for idx_tau in range(2):
                alobjz = alobj
                # for idx, blk in tqdm.tqdm(enumerate(blocks)):
                for idx, blk in enumerate(blocks):
                    train_no = blk['id']
                    train: Train = blk['train']
                    # update gradient
                    Ak = blk['A']


                    if bcdpar.dualobjtype == 1:
                        _c = (
                                blk['c']
                                + Ak.T @ (lbd + rho * _nonnegative(_Ax - Ak @ xk[idx] - b / 2))
                                + (- xk[idx] + 0.5) / tau
                        )
                    elif bcdpar.dualobjtype == 2:
                        # todo, implement _c
                        # _c = blk['c'] \
                        #      + rho * Ak.T @ _nonnegative(_Ax - b + lbd / rho) \
                        #      + (0.5 - xk[idx]) / tau
                        pass
                    else:
                        raise ValueError(f"cannot recognize type {bcdpar.dualobjtype}")

                    # compute shortest path
                    _x = train.vectorize_shortest_path(
                        _c, blk=blk, backend=bcdpar.sspbackend, model_and_x=blk.get('mx')
                    ).reshape(_c.shape)

                    # Note the subproblem is not a strict shortest path problem
                    #   you can choose not to visit any node
                    # if it is solved by ssp, then accept if it has negative cost
                    _v_sp = (_c.T @ _x).trace()
                    if (_v_sp > 0) and (bcdpar.sspbackend != 'grb'): _x = np.zeros(_c.shape)

                    _eps_fix_point[idx] = np.linalg.norm(xk[idx] - _x)
                    _xnorm[idx] = np.linalg.norm(xk[idx]) ** 2
                    _grad[idx] = np.linalg.norm(_c) ** 2

                    # update this block
                    xk[idx] = _x
                    xv[idx] = _x if kc == 0 else _x * 1 / (kc + 1) + kc / (kc + 1) * xv[idx]
                    _vAx[idx] = Ak @ _x
                    _vcx[idx] = _cx = (blk['c'].T @ _x).trace()
                _Ax = sum(_vAx.values())
                lobj = (
                        sum(_vcx.values())
                        + (lbd.T * (sum(_vAx.values()) - b)).sum()
                )
                alobj = (
                        lobj
                        + (_nonnegative(sum(_vAx.values()) - b) ** 2).sum() * rho / 2
                )

                # do line search
                if alobj - alobjz <= tau * sum(_eps_fix_point.values()):
                    break
                else:
                    tau /= tsig

            relerr = sum(_eps_fix_point.values()) / max(1, sum(_xnorm[idx] for idx,_ in enumerate(blocks)))
            if bcdpar.verbosity > 1:
                print("{:01d} cx: {:.1f} al_func:{:+.2f} grad_func:{:+.2e} relerr:{:+.2f} int:{:01d}".format(
                    it, sum(_vcx.values()), alobj, sum(_grad[idx] for idx,_ in enumerate(blocks)), relerr, idx_tau))

            # fixed-point eps
            if sum(_eps_fix_point.values()) < 1e-4:
                break
        _iter_time = time.time() - start
        _Ax = sum(_vAx.values())
        _vpfeas = _nonnegative(_Ax - b)
        eps_pfeas = np.linalg.norm(_vpfeas)
        cx = sum(_vcx.values())

        # lobj = cx + (_nonnegative(_Ax - b + lbd / rho) ** 2).sum() * rho / 2 - np.linalg.norm(lbd) ** 2 / 2 / rho
        # lobj = cx + (lbd.T * (_Ax - b)).sum() + (_nonnegative(_Ax - b) ** 2).sum() * rho / 2
        eps_fp = sum(_eps_fix_point.values())
        _log_line = "{:03d} {:.1e} {:+.2e} {:+.2e} {:+.2e} {:+.2e} {:+.3e} {:+.3e} {:+.3e} {:.2e} {:04d} {:04d}".format(
            k, _iter_time, r.cb, cx, lobj, alobj, eps_pfeas, eps_fp, rho, tau, it + 1, idx_tau
        )
        print(_log_line)
        ######################################################
        # primal dual restart
        ######################################################
        if eps_pfeas == 0:
            print(f":restarting epoch\n:kc = {kc}")
            if cx < r.cb:
                r.xb = copy.deepcopy(xk)
                r.cb = cx
            rho = rho0
            xk = copy.deepcopy([(1 - _xx) for _xx in xv])
            continue

        lbd = _nonnegative((_Ax - b) * rho + lbd)
        rho *= sigma

        train_list = [blk['train'] for blk in blocks]
        for tr, opt_lr in zip(train_list, _vcx):
            tr.opt_cost_LR = opt_lr
        pri_best_count, pri_best_obj = primal_heuristic(train_list, bcdpar.safe_int, target='obj')
        print("primal heuristic: best count: {}, best obj: {}".format(pri_best_count, pri_best_obj))

        bcdpar.iter += 1
        kc += 1
        if kc > 50:
            kc = 0

    return r


def primal_heuristic(
        train_list,
        safe_int,
        target='obj',
):
    """
    produce path: iterable of (station, time)

    Args:
        method: the name of primal feasible algorithm.
    """
    iter_max = 1

    best_count = count = -1
    best_obj = -1e6
    new_train_order = sorted(
        train_list, key=lambda tr: (tr.standard, tr.opt_cost_LR), reverse=True
    )

    for j in range(iter_max):
        path_cost_feasible = 0
        occupied_nodes = {}
        occupied_arcs = defaultdict(lambda: set())
        incompatible_arcs = set()
        count = 0
        not_feasible_trains = []
        train_order = new_train_order

        for idx, train in enumerate(train_order):
            train.update_primal_graph(
                occupied_nodes, occupied_arcs, incompatible_arcs, safe_int
            )
            # the heuristic only works for 1, but we may summarize using 0
            train.feasible_path, train.feasible_cost = train.shortest_path_primal(
                objtype=1
            )
            if not train.is_feasible:
                # path_cost_feasible += train.max_edge_weight * (len(train.staList) - 1)
                not_feasible_trains.append((idx, train.traNo))
                continue
            else:
                count += 1
            _this_occupied_nodes = get_occupied_nodes(train)
            occupied_nodes.update(_this_occupied_nodes)
            (
                _this_occupied_arcs,
                _this_new_incompatible_arcs,
            ) = get_occupied_arcs_and_clique(train.feasible_path)
            for k, v in _this_occupied_arcs.items():
                occupied_arcs[k].add(v)
            incompatible_arcs.update(_this_new_incompatible_arcs)
            path_cost_feasible += train.feasible_cost

        if (count > best_count) and (target == 'count'):
            print("better feasible number: {}".format(count))
            best_count = count
            best_obj = path_cost_feasible
            if best_count == len(train_list):
                print("all trains in the graph!")
                break
        elif (path_cost_feasible > best_obj) and (target == 'obj'):
            best_obj = path_cost_feasible
            best_count = count
        else:
            first_train_idx = not_feasible_trains[0][0]
            train_order.insert(0, train_order.pop(first_train_idx))
            print(
                "only feasible number: {}, incumbent: {}".format(count, best_count)
            )

    print(f"seq maximum cardinality: {best_count}, obj: {-best_obj}")

    #################################################################################
    # SUMMARIZATION

    return best_count, -best_obj


def get_occupied_nodes(train):
    _centers = train.feasible_path[1:-1]
    _nodes = {}
    for vs, t in _centers:
        _nodes[vs, t] = train.v_sta_type[vs]

    return _nodes


def get_occupied_arcs_and_clique(feasible_path):
    _this_occupied_arcs = {
        (i, j): (t1, t2)
        for (i, t1), (j, t2) in zip(feasible_path[1:-2], feasible_path[2:-1])
        if i[0] != j[1]  # if not same station
    }
    # edges starting later yet arriving earlier
    _this_new_incompatible_arcs = {
        ((i, st), (j, et))
        for (i, j), (t1, t2) in _this_occupied_arcs.items()
        for st in range(t1 + 1, t2)
        for et in range(st, t2)
    }
    # edges starting earlier yet arriving later
    _this_new_incompatible_arcs.update(
        ((i, st), (j, et))
        for (i, j), (t1, t2) in _this_occupied_arcs.items()
        for st in range(t1 - 10, t1)
        for et in range(t2, t2 + 10)
    )

    return _this_occupied_arcs, _this_new_incompatible_arcs