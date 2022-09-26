"""
functional interface module for model builder via gurobi and so forth
"""
import logging
import numpy as np

import scipy
from scipy.sparse import csr_matrix
import tqdm
from gurobipy import *

import data as ms
import util_output as uo
import util_solver as su
from Train import *

_logger = logging.getLogger("model-builder")
_logger.setLevel(logging.DEBUG)
# get the prefix and affix of a multiway constr
# ['aa', 'ap', 'ss', 'sp', 'pa', 'ps', 'pp']
# for example,
#   'aa' => (0, 0), which means you add constr to 2 inbound virtual stations,
#   say, for station v,
#   then you should add to (_v, t) and (_v, t)
# for arrival, it should add to inbound,
# for departure, ...         to outbound,
# for passing, does not matter.
bool_affix_safe_int_map = {
    'aa': (0, 0),  # 0 - inbound, 1 - outbound
    'ap': (0, 0),
    'ss': (1, 1),
    'sp': (1, 1),
    'pa': (0, 0),
    'ps': (1, 1),
    'pp': (0, 0),
}


def create_neighborhood(v_station_after, t, interval):
    """
    Args:
        node: current node
        speed: speed of the train ahead
        _type: type of the safe interval considered

    Returns:
        a set of nodes being the neighborhood of v
    """

    return {(v_station_after, t + dlt) for dlt in range(1, interval)}


##############################
# all-in-one milp model
##############################

def optimize(model, zjv):
    model.optimize()
    import pandas as pd
    zsol = pd.Series(model.getAttr("X", zjv))
    zsol = zsol[zsol > 0]
    for tr in ms.train_list:
        tr.is_best_feasible = False
        try:
            tb = zsol[tr.traNo].index
            if tb.size > 2:
                for node in tb:
                    tr.timetable[node[0]] = node[1]
                    tr.is_best_feasible = True
        except:
            print(tr.traNo, "not available")
    uo.plot_timetables_h5(ms.train_list, ms.miles, ms.station_list, param_sys=params_sys, param_subgrad=params_subgrad)


def create_milp_model():
    model = Model("quadratic-proximal-ssp")

    univ_nodes = tuplelist(
        set(v['name'] for tr in ms.train_list for v in tr.subgraph.vs)
    )
    # zjv[j, v]
    zjv = model.addVars(
        ((tr.traNo, *v['name']) for tr in ms.train_list for v in tr.subgraph.vs),
        lb=0, ub=1.0, name='zjv'
    )
    # starting arcs
    s_arcs = model.addVars(
        (tr.traNo for tr in ms.train_list), vtype=GRB.BINARY, name='s_arcs'
    )

    for tr in tqdm.tqdm(ms.train_list):
        # note xe are actually different for each train: j
        g = tr.subgraph
        xe = model.addVars((e['name'] for e in g.es), lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f'e@{tr.traNo}')
        for v in g.vs:
            in_edges = v.in_edges()
            out_edges = v.out_edges()

            if v.index == tr._ig_s:
                model.addConstr(quicksum(xe[e['name']] for e in out_edges) == s_arcs[tr.traNo],
                                name=f'start_arcs[{tr.traNo}]')
                model.addConstr(s_arcs[tr.traNo] <= 1, name=f'sk_{(tr.traNo, *v["name"])}')
                model.addConstr(
                    quicksum(xe[e['name']] for e in out_edges) == zjv[(tr.traNo, *v['name'])],
                    name=f'zjv{(tr.traNo, *v["name"])}'
                )
                continue
            if v.index == tr._ig_t:
                model.addConstr(quicksum(xe[e['name']] for e in in_edges) <= 1, name=f'sk_{(tr.traNo, *v["name"])}')
                model.addConstr(
                    quicksum(xe[e['name']] for e in in_edges) == zjv[(tr.traNo, *v['name'])],
                    name=f'zjv{(tr.traNo, *v["name"])}'
                )
                continue
            model.addConstr(quicksum(xe[e['name']] for e in in_edges) - quicksum(xe[e['name']] for e in out_edges) == 0,
                            name=f'sk_{(tr.traNo, *v["name"])}')
            model.addConstr(quicksum(xe[e['name']] for e in in_edges) == zjv[(tr.traNo, *v['name'])],
                            name=f'zjv{(tr.traNo, *v["name"])}')

    train_class = defaultdict(list)
    for tr in ms.train_list:
        for station, tp in tr.v_sta_type.items():
            train_class[station, tr.speed, tp].append(tr.traNo)

    # binding constraints by multiway~
    # ['aa', 'ap', 'ss', 'sp', 'pa', 'ps', 'pp']

    for _tp, (bool_affix_ahead, bool_affix_after) in tqdm.tqdm(bool_affix_safe_int_map.items()):
        ahead, after = _tp
        sub_safe_int = ms.safe_int[_tp]
        for (station, speed), interval in sub_safe_int.items():
            v_station_ahead = f"{station}_" if bool_affix_ahead else f"_{station}"
            v_station_after = f"{station}_" if bool_affix_after else f"_{station}"
            list_train_head = train_class[v_station_ahead, speed, ahead]
            list_train_after = train_class[v_station_after, speed, after]
            if len(list_train_head) == 0 or len(list_train_after) == 0:
                continue
            for v in univ_nodes.select(v_station_ahead, '*'):
                _, t = v
                ahead_expr = quicksum(zjv.select("*", v_station_ahead, t))
                after_expr = quicksum(
                    quicksum(zjv.select("*", v_station_after, t_after)) for _, t_after in
                    create_neighborhood(v_station_after, t, interval))

                model.addConstr(
                    LinExpr(ahead_expr + after_expr) <= 1,
                    f"multiway_{_tp}@{speed}@{v_station_ahead}:{v_station_after}@{t}"
                )

    # maximum number of trains, by add up z
    obj_expr = -quicksum(s_arcs.values())

    model.setObjective(obj_expr, sense=GRB.MINIMIZE)
    model.setParam("LogToConsole", 1)
    model.setParam("Threads", 1)

    return model, zjv


def split_model(m: Model):
    m.update()

    # get constrs and vars for single train
    train_index_dict = {tr.traNo: {"vars": None, "constrs": None} for tr in ms.train_list}
    for traNo in tqdm.tqdm(train_index_dict):
        train_constrs = su.getConstrByPrefix(m, [f"sk_({traNo},", f"zjv({traNo},", f"start_arcs[{traNo}]"])
        train_vars = su.getVarByPrefix(m, (f"zjv[{traNo},", f"e@{traNo}[", f"s_arcs[{traNo}]"))
        train_index_dict[traNo]["constrs"] = [constr.index for constr in train_constrs]
        train_index_dict[traNo]["vars"] = [var.index for var in train_vars]

        # assert all vars and constrs are in model and not removed
        assert all(ind >= 0 for ind in train_index_dict[traNo]["constrs"]) and all(
            ind >= 0 for ind in train_index_dict[traNo]["vars"])

    # get coupling constraints
    coupling_constr_index = [constr.index for constr in su.getConstrByPrefix(m, "multi")]

    # sanity check for constrs
    all_constrs = coupling_constr_index.copy()
    for traNo in train_index_dict:
        all_constrs.extend(train_index_dict[traNo]["constrs"])
    assert len(all_constrs) == len(set(all_constrs)) == len(m.getConstrs())

    # sanity check for vars
    all_vars = []
    for traNo in train_index_dict:
        all_vars.extend(train_index_dict[traNo]["vars"])
    assert len(all_vars) == len(set(all_vars)) == len(m.getVars())

    # save coupling constraints
    train_index_dict["coupling"] = coupling_constr_index
    return train_index_dict


##############################
# decomposed-multiblock milp model
##############################
def getA_b_c(m: Model, model_index, binding_size):
    m.update()

    # get coupling constraints
    non_coupling_constr_index = []
    coupling_constr_index = []
    for constr in m.getConstrs():
        if constr.ConstrName.startswith("multi"):
            coupling_constr_index.append(constr.index)
        else:
            non_coupling_constr_index.append(constr.index)

    A = m.getA()
    sense = np.array(m.getAttr('Sense')).reshape((m.NumConstrs, 1))
    b = np.array(m.getAttr('RHS')).reshape((m.NumConstrs, 1))
    _, n = A.shape
    # build
    B_k = A[non_coupling_constr_index, :]
    b_k = b[non_coupling_constr_index, :]
    A_k = _initialize_csr(model_index, coupling_constr_index, A, shape=(binding_size, n))
    b = np.ones((binding_size, 1))
    sense_B_k = sense[non_coupling_constr_index, :]
    sense_A_k = sense[coupling_constr_index, :]
    obj = m.getObjective()
    c_k = np.zeros((m.NumVars, 1))
    for i in range(obj.size()):
        c_k[obj.getVar(i).index] = obj.getCoeff(i)

    return A_k, B_k, b_k, c_k, b, sense_B_k, ['<'] * binding_size


def _initialize_csr(model_index, coupling_constr_index, A, shape):
    """
    initialize csr matrix of the big coupling matrix A_k
        total size of A_k in shape
        A_k[model_index, :] = A[coupling_constr_index, :]
    """
    # @note: change this.
    # A_k = np.zeros((binding_size, n))
    # A_k[model_index, :] = A[coupling_constr_index, :].todense()
    # _logger.debug(A_k[12, :].nonzero())
    A1 = A[coupling_constr_index, :].tocoo()
    new_rows = np.array([model_index[idx] for idx in A1.row])
    A_k = scipy.sparse.coo_matrix((A1.data, (new_rows, A1.col)), shape=shape)
    return A_k.tocsr()


def create_decomposed_models():
    # @update:
    #   - remove s_arcs
    #   - remote zjv
    #   - now xe only
    train_class = defaultdict(list)
    for tr in ms.train_list:
        for station, tp in tr.v_sta_type.items():
            train_class[station, tr.speed, tp].append(tr.traNo)

    univ_nodes = tuplelist(
        set(v['name'] for tr in ms.train_list for v in tr.subgraph.vs)
    )

    # query global binding constraints index info
    global_index = {}
    idx = 0
    for _tp, (bool_affix_ahead, bool_affix_after) in bool_affix_safe_int_map.items():
        ahead, after = _tp
        sub_safe_int = ms.safe_int[_tp]
        for (station, speed), interval in sub_safe_int.items():
            v_station_ahead = f"{station}_" if bool_affix_ahead else f"_{station}"
            v_station_after = f"{station}_" if bool_affix_after else f"_{station}"
            list_train_head = train_class[v_station_ahead, speed, ahead]
            list_train_after = train_class[v_station_after, speed, after]
            if len(list_train_head) == 0 or len(list_train_after) == 0:
                continue
            for _, t in sorted(univ_nodes.select(v_station_ahead, '*')):
                global_index[_tp, station, speed, v_station_ahead, t] = idx
                idx += 1

    model_dict = {}
    model_index = {}
    for tr in tqdm.tqdm(ms.train_list):
        model = Model(f"quadratic-proximal-ssp-{tr.traNo}")

        # note xe are actually different for each train: j
        g = tr.subgraph
        # node name to index
        node_to_index = {
            n['name']: n.index for n in g.vs
        }
        # xe = model.addVars((e['name'] for e in g.es), lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f'e@{tr.traNo}')
        xe = model.addVars((e.index for e in g.es), lb=0.0, ub=1.0, vtype=GRB.BINARY, name=f'e@{tr.traNo}')
        # use index
        node_in_edges = g.get_inclist(ig.IN)
        for v in g.vs:
            in_edges = v.in_edges()
            out_edges = v.out_edges()

            if v.index == tr._ig_s:
                model.addConstr(quicksum(xe[e.index] for e in out_edges) <= 1, name=f'sk_{(tr.traNo, *v["name"])}')
                continue
            if v.index == tr._ig_t:
                model.addConstr(quicksum(xe[e.index] for e in in_edges) <= 1, name=f'sk_{(tr.traNo, *v["name"])}')
                continue
            model.addConstr(quicksum(xe[e.index] for e in in_edges) - quicksum(xe[e.index] for e in out_edges) == 0,
                            name=f'sk_{(tr.traNo, *v["name"])}')

        index_array = []
        for _tp, (bool_affix_ahead, bool_affix_after) in bool_affix_safe_int_map.items():
            ahead, after = _tp
            sub_safe_int = ms.safe_int[_tp]
            for (station, speed), interval in sub_safe_int.items():
                v_station_ahead = f"{station}_" if bool_affix_ahead else f"_{station}"
                v_station_after = f"{station}_" if bool_affix_after else f"_{station}"
                list_train_head = train_class[v_station_ahead, speed, ahead]
                list_train_after = train_class[v_station_after, speed, after]
                if len(list_train_head) == 0 or len(list_train_after) == 0:
                    continue
                for node in g.vs.select(station=v_station_ahead):
                    _, t = node['name']

                    # ahead_expr = quicksum(xe[e['name']] for e in node.in_edges())
                    ahead_expr = quicksum(xe.select(node_in_edges[node.index]))
                    # after_expr = quicksum(
                    #     xe[e['name']]
                    #     for vv in g.vs.select(name_in=create_neighborhood(v_station_after, t, interval))
                    #     for e in vv.in_edges()
                    # )
                    after_expr = quicksum(
                        quicksum(xe.select(node_in_edges[vv.index]))
                        for vv in g.vs[[node_to_index[n] for n in create_neighborhood(v_station_after, t, interval) if
                                        n in node_to_index]]
                    )
                    model.addConstr(
                        LinExpr(ahead_expr + after_expr) <= 1,
                        f"multiway_{_tp}@{speed}@{v_station_ahead}:{v_station_after}@{t}"
                    )
                    index_array.append(global_index[_tp, station, speed, v_station_ahead, t])

        # maximum number of trains, by add up z
        obj_expr = -quicksum(xe[e.index] for e in g.vs[tr._ig_s].out_edges())

        model.setObjective(obj_expr, sense=GRB.MINIMIZE)
        # model.setParam("LogToConsole", 1)
        # model.setParam("Threads", 1)

        model_dict[tr.traNo] = model
        model_index[tr.traNo] = index_array

    return model_dict, global_index, model_index
