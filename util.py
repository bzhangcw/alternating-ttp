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
xa_map = defaultdict(lambda: {"a": 0, "s": 0, "p": 0})
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
