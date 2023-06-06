"""
use matrix based API to run Augmented Lagrangians.
"""
import copy

import scipy.io
import functional_bcd as bcd
import pickle as pk
import data as ms

from functional_model_builder import *
from util import SysParams

if __name__ == '__main__':
    # example usage.
    params_sys = SysParams()
    params_subgrad = SubgradParam()
    params_bcd = bcd.BCDParams()

    params_sys.parse_environ()
    params_subgrad.parse_environ()
    params_bcd.parse_environ()

    _, safe_int = ms.setup(params_sys)
    params_bcd.safe_int = safe_int

    data_name = f"ttp_{params_sys.train_size}_{params_sys.station_size}_{params_sys.time_span}"
    try:
        mat_dict = pk.load(open(f"{data_name}.pk", 'rb'))
        print("loaded a generated matlab style dict")
    except Exception as _:
        print("generate matlab style dict")
        model_dict, global_index, model_index = create_decomposed_models(obj_type=params_sys.obj)
        mat_dict = generate_matlab_dict(model_dict, global_index, model_index)
        # mat_dict_for_mat = generate_matlab_dict(model_dict, global_index, model_index, False)
        # scipy.io.savemat(open(f"{data_name}.mat", 'wb'), mat_dict_for_mat, do_compression=True)
        with open(f"{data_name}.pk", 'wb') as f:
            pk.dump(mat_dict, f)

    params_bcd.sspbackend = "grb"  # FIXME
    params_bcd.itermax = 50  # FIXME
    r, pri_best_xks = bcd.optimize(bcdpar=params_bcd, mat_dict=mat_dict)

    # sanity check
    model, zjv, xes, s_arcs = create_milp_model(obj_type=params_sys.obj)
    model.write(f"{data_name}.mps.gz")
    model.optimize()
    fb = model.objval
    zb = quicksum(s_arcs.values()).getValue()
    # r.xb = copy.deepcopy(r.xk)
    for k, trs in tqdm.tqdm(enumerate(mat_dict['trains'])):
        tr = trs['train']
        for e in tr.subgraph.es:

            (r.xb[k][e.index] > 0.5) and print(e, pri_best_xks[tr.traNo][e.index])
            xes[tr.traNo][e['name']].setAttr(GRB.Attr.LB, pri_best_xks[tr.traNo][e.index])
            xes[tr.traNo][e['name']].setAttr(GRB.Attr.UB, pri_best_xks[tr.traNo][e.index])

    model.optimize()
    fx = model.objval

    print(f"""
:gurobi validation
- best objective: {fb}#{zb}
- bcd: {r.cb}
    """)
# - bcd: {r.cb}  # {quicksum(s_arcs.values()).getValue()}