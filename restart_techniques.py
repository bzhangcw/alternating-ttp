# -*- coding: utf-8 -*-
# @author: PuShanwen
# @email: 2019212802@live.sufe.edu.cn
# @date: 2022/06/06
import numpy as np


def naive_restart(x, _c):
    if (x > 0).any():
        _c[x > 0] += 1
    else:
        _c -= 1
    return _c


def perturbed_restart(train, _c, itermax, alm):
    obj_bst = 1e+10
    _c_bst = 0
    for _ in range(itermax):
        _c_new = np.random.rand(_c.shape)
        # save to price
        train.vectorize_update_arc_multiplier(_c_new.flatten() - _c_new.min())
        # compute shortest path
        _x = train.vectorize_shortest_path().reshape(_c_new.shape)





def path_generating_restart():
    pass


def xk_not_change(xk, xk_old):
    eps = 1e-8
    return all((xk[idx] - xk_old[idx]).max() < eps for idx in range(len(xk)))
