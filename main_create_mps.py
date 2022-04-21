import tqdm
import pickle
from gurobipy import *

import util_output as uo
import main_slim as ms
import solver_utils as su
from train import *

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
        v['name'] for tr in ms.train_list for v in tr.subgraph.vs
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
                model.addConstr(quicksum(xe[e['name']] for e in out_edges) == s_arcs[tr.traNo], name=f'start_arcs[{tr.traNo}]')
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
            model.addConstr(quicksum(xe[e['name']] for e in in_edges) == zjv[(tr.traNo, *v['name'])], name=f'zjv{(tr.traNo, *v["name"])}')

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
            for v in univ_nodes.select(v_station_ahead):
                _, t = v
                ahead_expr = quicksum(zjv.select("*", v_station_ahead, t))
                after_expr = quicksum(
                    quicksum(zjv.select("*", v_station_after, t_after)) for _, t_after in
                    create_neighborhood(v_station_after, t, interval))

                model.addConstr(
                    ahead_expr + after_expr <= 1,
                    f"multiway_{_tp}@{v_station_ahead}:{v_station_after}"
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
        assert all(ind >= 0 for ind in train_index_dict[traNo]["constrs"]) and all(ind >= 0 for ind in train_index_dict[traNo]["vars"])

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


if __name__ == '__main__':
    params_sys = SysParams()
    params_subgrad = SubgradParam()

    params_sys.parse_environ()
    params_subgrad.parse_environ()

    ms.setup(params_sys, params_subgrad)

    model, zjv = create_milp_model()

    train_index_dict = split_model(model)

    with open(f"index_dict_{params_sys.train_size}_{params_sys.station_size}_{params_sys.time_span}.pkl", "wb") as file:
        pickle.dump(train_index_dict, file)

    model.write(f"test_{params_sys.train_size}_{params_sys.station_size}_{params_sys.time_span}.lp")
    model.write(f"test_{params_sys.train_size}_{params_sys.station_size}_{params_sys.time_span}.rlp")
    # model.write(f"test_{params_sys.train_size}_{params_sys.station_size}_{params_sys.time_span}.mps")
    optimize(model, zjv)
