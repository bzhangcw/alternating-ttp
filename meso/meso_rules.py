


class MesoRules(object):
    # define the intervals
    # inteval(INT) of something X will be written as:
    #    "INV_X"
    # I追:同方向列车区间追踪间隔时间（3min）。
    INT_CHASING = 3
    # I发发：同方向列车车站连续发车间隔时间。上海虹桥5min，北京南12—15道4分，其余5分;京沪高其余各站4min。(全部4min)
    # INT - departure-departure (D-D)
    INT_DD = 3
    # I到到：同方向列车连续到达间隔时间（4min）。
    # INT - arrival-arrival (A-A)
    INT_AA = 4
    # I到通：同方向列车先到后通间隔时间（3min）。
    # INT - arrival-passing (A-P)
    INT_AP = 4
    # I通发：同方向列车先通后发间隔时间（2min/1min）。
    INT_PD = 2
    # I发通：同方向列车先发后通间隔时间，前车为300km/h列车（5min），前车为350km/h列车（6min）。
    INT_DP = 5
    # 发到：同方向列车同股道接车间隔时间：6min。
    INT_DA = 6
    # 敌对进路相对方向列车不同时到发(先到后发）间隔时间“τ敌到发”（2min）。
    INT_OPPO_AD = 2
    # 敌对进路相对方向列车不同时发到(先发后到）间隔时间“τ敌发到”：北京南8min，其他7min。
    INT_OPPO_DA = 8
    # 列车在站折返时间：动车组列车20min。
    # INT - arrival-passing (A-P)
    INT_ZIG = 4
    # 沿途停车办理客运业务停站时间最小2min。
    INT_SERVICE = 2

    def __init__(self):
        pass