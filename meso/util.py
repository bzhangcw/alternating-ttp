"""
utility modules
"""
from functools import wraps
from time import time

timeit = True


def timing(f):
    if timeit:
        @wraps(f)
        def wrap(*args, **kw):
            ts = time()
            result = f(*args, **kw)
            te = time()
            print(f'Function {f.__name__} took {te - ts:2.4f} seconds')
            return result
    else:
        wrap = f
    return wrap


class SysParams(object):
    # todo, support ArgumentParser
    DBG = False
    station_size = 0
    train_size = 0
    time_span = 0
    iter_max = 0
    down = 0
    fix_preferred_time = True

    def parse_environ(self):
        import os
        self.station_size = int(os.environ.get('station_size', 29))
        self.train_size = int(os.environ.get('train_size', 4))
        self.time_span = int(os.environ.get('time_span', 1080))
        self.iter_max = int(os.environ.get('iter_max', 100))
        self.down = int(os.environ.get('down', -1))
        self.fix_preferred_time = int(os.environ.get('fix_preferred_time', True))


# subgradient params
class SubgradParam(object):

    def __init__(self):
        self.kappa = 0.2
        self.alpha = 1.0
        self.gamma = 0.1  # parameter for argmin x
        self.changed = 0
        self.num_stuck = 0
        self.eps_num_stuck = 3
        self.iter = 0
        self.lb = 1e-6
        self.lb_arr = [-1e6]
        self.ub_arr = [1e6]
        self.gap = 1
        self.dual_method = "pdhg"  # "lagrange" or "pdhg"
        self.primal_heuristic_method = "jsp"  # "jsp" or "seq"
        self.feasible_provider = "jsp"  # "jsp" or "seq"
        self.max_number = 1

    def parse_environ(self):
        import os
        self.primal_heuristic_method = os.environ.get('primal', 'seq')
        self.dual_method = os.environ.get('dual', 'pdhg')

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
