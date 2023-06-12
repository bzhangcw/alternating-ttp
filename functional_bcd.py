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
from enum import IntEnum
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


class DualObjType(IntEnum):
    ALMGP = 1
    ALMGC = 0


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
        self.linmax = 10

    def parse_environ(self):
        import os

        self.primal_heuristic_method = os.environ.get("primal", "jsp")
        self.dual_method = os.environ.get("dual", "pdhg_alm")
        self.sspbackend = os.environ.get("sspbackend", "grb")
        self.dualobjtype = DualObjType(
            int(os.environ.get("dualobj", DualObjType.ALMGP))
        )
        self.verbosity = int(os.environ.get("verbosity", 1))
        self.itermax = int(os.environ.get("iter_max", 1))
        self.timelimit = int(os.environ.get("time_max", 200))

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
        headers = [
            "k",
            "t",
            "f*",
            "c'x",
            "L(λ)",
            "aL(λ)",
            "|Ax - b|",
            "error",
            "ρ",
            "τ",
            "kl",
            "kls",
        ]
        slots = [
            "{:^3s}",
            "{:^7s}",
            "{:^9s}",
            "{:^9s}",
            "{:^9s}",
            "{:^9s}",
            "{:^10s}",
            "{:^10s}",
            "{:^9s}",
            "{:^9s}",
            "{:4s}",
            "{:4s}",
        ]
        self._log_header = _log_header = " ".join(slots).format(*headers)
        lt = _log_header.__len__()
        print("*" * lt)
        print(("{:^" + f"{lt}" + "}").format("BCD for MILP"))
        print(("{:^" + f"{lt}" + "}").format("(c) Chuwen Zhang, Shanwen Pu, Rui Wang"))
        print(("{:^" + f"{lt}" + "}").format("2022"))
        print("*" * lt)
        print(("{:" + f"{lt}" + "}").format(f"- backend : {self.sspbackend}"))
        print(("{:" + f"{lt}" + "}").format(f"- dual    : {self.dualobjtype.name}"))
        print(
            ("{:" + f"{lt}" + "}").format(
                f"- limit   : {self.itermax}/{self.timelimit}"
            )
        )
        print("*" * lt)
        print(_log_header)
        print("*" * lt)


def _Ax(block, x):
    return block["A"] @ x


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
    blocks = mat_dict["trains"]
    b = mat_dict["b"]
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
    lobj = -1e6
    np.random.seed(1)
    xb = [np.zeros((blk["n"], 1)) for idx, blk in enumerate(blocks)]
    xk = [np.random.random((blk["n"], 1)) for idx, blk in enumerate(blocks)]
    xv = [np.zeros((blk["n"], 1)) for idx, blk in enumerate(blocks)]
    lbd = rho * np.random.random(b.shape)
    r = Result()
    r.cx, r.cb, r.xb, r.xv, r.xk = cx, cb, xb, xv, xk
    # logger
    # - k: outer iteration num
    # - it: inner iteration num
    # - idx: 1-n block idx
    #       it may not be the train no
    # A_k x_k
    _vAx = {idx: blk["A"] @ xk[idx] for idx, blk in enumerate(blocks)}
    _Ax = sum(_vAx.values())
    # c_k x_k
    _vcx = {idx: (blk["c"].T @ xk[idx]).trace() for idx, blk in enumerate(blocks)}
    # x_k - x_k* (fixed point error)
    _eps_fix_point = {idx: 0 for idx, blk in enumerate(blocks)}
    _xnorm = {idx: 0 for idx, _ in enumerate(blocks)}
    _grad = {idx: 0 for idx, _ in enumerate(blocks)}
    if bcdpar.sspbackend == "grb":
        print("creating models")
        for idx, blk in enumerate(blocks):
            train: Train = blk["train"]
            _c = blk["c"]
            blk["mx"] = train.create_shortest_path_model(blk)
        print("creating models finished")

    alobj = (
            sum(_vcx.values())
            + (lbd.T @ (sum(_vAx.values()) - b)).sum()
            + (_nonnegative(sum(_vAx.values()) - b) ** 2).sum() * rho / 2
    )

    bcdpar.show_log_header()
    kc = 0

    bst_pri_best_xks = None
    for k in range(bcdpar.itermax):
        try:
            # packing args
            args = (
                bcdpar,
                alobj,
                tau0,
                rho,
                blocks,
                lbd,
                xk,
                b,
                _eps_fix_point,
                _xnorm,
                _grad,
                xv,
                kc,
                _vAx,
                _vcx,
                tsig,
            )

            ######################################################
            # once the linearization finished,
            # run lagrangian relaxation to find a LB
            ######################################################
            _lb, *_ = lagrangian_relax(bcdpar, blocks, lbd, mat_dict["b"])
            if _lb > lobj:
                lobj = _lb

            ######################################################
            # the bcd k-th iteration
            ######################################################
            it, idx_tau, alobj, *_ = block_coord_descent(*args)

            _iter_time = time.time() - start
            _Ax = sum(_vAx.values())
            _vpfeas = _nonnegative(_Ax - b)
            eps_pfeas = np.linalg.norm(_vpfeas)
            cx = sum(_vcx.values())

            # lobj = cx + (_nonnegative(_Ax - b + lbd / rho) ** 2).sum() * rho / 2 - np.linalg.norm(lbd) ** 2 / 2 / rho
            # lobj = cx + (lbd.T * (_Ax - b)).sum() + (_nonnegative(_Ax - b) ** 2).sum() * rho / 2
            eps_fp = sum(_eps_fix_point.values())


            ######################################################
            # primal heuristics
            ######################################################
            bool_use_primal = 1
            if bool_use_primal:
                train_list = [blk["train"] for blk in blocks]
                for tr, opt_lr in zip(train_list, _vcx.values()):
                    tr.opt_cost_LR = opt_lr
                pri_best_count, pri_best_obj, pri_best_xks = primal_heuristic(
                    train_list, b.copy(), blocks, target="obj"
                )
                print(
                    "       ⎣primal heuristic: best count: {}, best obj: {}".format(
                        pri_best_count, pri_best_obj
                    )
                )
                if r.cb > pri_best_obj:
                    r.xb = pri_best_xks
                    r.cb = pri_best_obj

            _log_line = "{:03d} {:.1e} {:+.2e} {:+.2e} {:+.2e} {:+.2e} {:+.3e} {:+.3e} {:+.3e} {:.2e} {:04d} {:04d}".format(
                k,
                _iter_time,
                r.cb,
                cx,
                lobj,
                alobj,
                eps_pfeas,
                eps_fp,
                rho,
                tau,
                it + 1,
                idx_tau,
            )
            print(_log_line)

            if _iter_time > bcdpar.timelimit:
                break

            ######################################################
            # primal dual restart
            ######################################################
            if eps_pfeas == 0:
                print(f":restarting epoch\n:kc = {kc}")
                print(bcdpar._log_header)
                rho = rho0
                if cx <= r.cb:
                    r.cb = cx
                    r.xb = {blk['train'].traNo: xk[idx] for idx, blk in enumerate(blocks)}
                # primal restart at an acc mean
                xk = copy.deepcopy([(1 - _xx) for _xx in xv])
                continue

            lbd = _nonnegative((_Ax - b) * rho + lbd)
            rho *= sigma
            bcdpar.iter += 1
            kc += 1
            if kc > 50:
                kc = 0
        except KeyboardInterrupt as e:
            print("terminated early")
            break
    print(bcdpar._log_header)
    print(_log_line)
    return r, r.xb


def block_coord_descent(*args):
    (
        bcdpar,
        alobj,
        tau0,
        rho,
        blocks,
        lbd,
        xk,
        b,
        _eps_fix_point,
        _xnorm,
        _grad,
        xv,
        kc,
        _vAx,
        _vcx,
        tsig,
        *_,
    ) = args
    # for linter only
    idx_tau = 0
    for it in range(bcdpar.linmax):
        # linearization loop
        # idx: A[idx]@x[idx]
        tau = tau0
        alobjz = alobj
        for idx_tau in range(2):
            # line search loop

            for idx, blk in enumerate(blocks):
                train_no = blk["id"]
                train: Train = blk["train"]
                # update gradient
                Ak = blk["A"]

                if bcdpar.dualobjtype == DualObjType.ALMGP:
                    # bcd style ALM-GP
                    # each time recalculate feasibility
                    # this is not ADMM
                    _Ax = sum(_vAx.values())
                    _c = (
                            blk["c"]
                            + Ak.T @ (lbd + rho * _nonnegative(_Ax - Ak @ xk[idx] - b / 2))
                            + (-xk[idx] + 0.5) / tau
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
                    _c, blk=blk, backend=bcdpar.sspbackend, model_and_x=blk.get("mx")
                ).reshape(_c.shape)

                # Note the subproblem is not a strict shortest path problem
                #   you can choose not to visit any node
                # if it is solved by ssp, then accept if it has negative cost
                _v_sp = (_c.T @ _x).trace()
                if (_v_sp > 0) and (bcdpar.sspbackend != "grb"):
                    _x = np.zeros(_c.shape)

                _eps_fix_point[idx] = np.linalg.norm(xk[idx] - _x)
                _xnorm[idx] = np.linalg.norm(xk[idx]) ** 2
                _grad[idx] = np.linalg.norm(_c) ** 2

                # update this block
                xk[idx] = _x
                xv[idx] = _x if kc == 0 else _x * 1 / (kc + 1) + kc / (kc + 1) * xv[idx]
                _vAx[idx] = Ak @ _x
                _vcx[idx] = _cx = (blk["c"].T @ _x).trace()

            lobj = sum(_vcx.values()) + (lbd.T @ (sum(_vAx.values()) - b)).sum()
            alobj = lobj + (_nonnegative(sum(_vAx.values()) - b) ** 2).sum() * rho / 2

            if bcdpar.verbosity > 1:
                relerr = sum(_eps_fix_point.values()) / max(
                    1, sum(_xnorm[idx] for idx, _ in enumerate(blocks))
                )

                print(
                    "   ⎣--{:01d} cx: {:.1f} al_func:{:+.2f} grad_func:{:+.2e} relerr:{:+.2f} int:{:01d}".format(
                        it,
                        sum(_vcx.values()),
                        alobj,
                        sum(_grad[idx] for idx, _ in enumerate(blocks)),
                        relerr,
                        idx_tau,
                    )
                )
            # do line search
            if alobj - alobjz <= tau * sum(_eps_fix_point.values()):
                break
            else:
                tau /= tsig

        # fixed-point eps
        if sum(_eps_fix_point.values()) < 1e-4:
            break

    return it, idx_tau, alobj


def lagrangian_relax(bcdpar, blocks, lbd, b):
    _lobj = 0
    for idx, blk in enumerate(blocks):
        train: Train = blk["train"]
        # update gradient
        Ak = blk["A"]
        _c = blk["c"] + Ak.T @ lbd

        # compute shortest path
        _x = train.vectorize_shortest_path(
            _c, blk=blk, backend=bcdpar.sspbackend, model_and_x=blk.get("mx")
        ).reshape(_c.shape)

        # Note the subproblem is not a strict shortest path problem
        #   you can choose not to visit any node
        # if it is solved by ssp, then accept if it has negative cost
        _lobj += (_c.T @ _x).trace()
    _lobj -= (lbd.T @ b)[0, 0]
    return _lobj, _x


def primal_heuristic(
        train_list,
        b,
        blocks,
        target="obj",
):
    """
    produce path: iterable of (station, time)

    Args:
        method: the name of primal feasible algorithm.
    """
    iter_max = 3

    best_count = count = -1
    best_obj = 1e6
    train_order = sorted(
        zip(train_list, blocks),
        key=lambda pair: (pair[0].standard, pair[0].opt_cost_LR),
        reverse=True,
    )

    for j in range(iter_max):
        path_cost_feasible = 0
        count = 0
        inf_trs = []
        feas4all = True

        xks = {}
        b_bar = b.copy()
        for idx, (train, blk) in enumerate(train_order):
            assert train.traNo == blk["id"]
            # the heuristic only works for 1, but we may summarize using 0
            _x, optimal, b_bar = train.vectorize_heur_grb(
                blk, b_bar, model_and_x=blk["mx"]
            )

            xks[blk["id"]] = _x
            train.feasible_cost = blk["c"].T @ _x
            train.feasible_path = get_path_from_vec(_x, train)
            # print("Train {}: {}".format(train.traNo, train.feasible_path))
            if _x.sum() > 0.5:
                train.is_feasible = True
            else:
                # not feasible, so put this tr, blk to the first of train_order
                inf_trs.append(idx)
                feas4all = False

            if train.is_feasible:
                count += 1

            path_cost_feasible += train.feasible_cost

        if not feas4all:
            # put all infeasible trs to the first
            feas_trs = [idx for idx in range(len(train_order)) if idx not in inf_trs]
            new_idx = inf_trs + feas_trs
            train_order = [train_order[idx] for idx in new_idx]

        if (count > best_count) and (target == "count"):
            print("     better feasible number: {}".format(count))
            best_count = count
            best_obj = path_cost_feasible
            best_xks = xks.copy()
            assert (
                    abs(best_obj - sum(blk["c"].T @ xks[blk["id"]] for blk in blocks))
                    < 1e-2
            )
            if best_count == len(train_list):
                print("     all trains in the graph!")
                break
        elif (path_cost_feasible < best_obj) and (target == "obj"):
            best_obj = path_cost_feasible
            best_xks = xks.copy()
            assert (
                    abs(best_obj - sum(blk["c"].T @ xks[blk["id"]] for blk in blocks))
                    < 1e-2
            )
            best_count = count
        else:
            print(
                "     ⎣only feasible number: {}, incumbent: {}, best_obj: {}".format(
                    count, best_count, best_obj
                )
            )

    # print(f"seq maximum cardinality: {best_count}, obj: {best_obj}")

    #################################################################################
    # SUMMARIZATION

    return best_count, best_obj[0], best_xks


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


def get_path_from_vec(x, tr):
    # x is a vector of 0 and 1, and it's created by the shortest path algorithm in the graph
    # the index of x is the index of the edge in the graph
    # the value of x is 1 if the edge is in the path, otherwise 0
    # the path is a list of (station, time)
    path = [("s_", -1)]
    assert len(x) == len(tr.subgraph.es)
    for i, _e in enumerate(x):
        if _e > 0.5:
            i, o = tr.subgraph.es[i]["name"]  # FIXME
            path.append(o)
    return path
