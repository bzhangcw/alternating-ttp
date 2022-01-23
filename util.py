"""
utility modules
"""
from collections import defaultdict

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
# node multiplier
yv_multiplier = {}  # the multiplier of each v
yvc_multiplier = defaultdict(lambda: {"a": 0, "s": 0, "p": 0})  # the multiplier of each v with type c
safe_int = {}
# category list
category = ["s", "a", "p"]


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
