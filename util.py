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
# (s,t) node -> (s', t', s, t) arc : value
yv2xa_map = defaultdict(lambda: defaultdict(int))
# flattened yv2xa
# (s', t', s, t) arc : value
xa_map = defaultdict(int)
# node precedence map in terms of arrival/departure interval
node_prec_map = defaultdict(list)
# original Lagrangian
multiplier = defaultdict(int)  # each (station, t)
# node multiplier
yv_multiplier = {}  # the multiplier of each v
safe_int = {}
