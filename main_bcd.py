"""
use matrix based API to run Augmented Lagrangians.
"""
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

    ms.setup(params_sys)

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

    r = bcd.optimize(bcdpar=params_bcd, mat_dict=mat_dict)

    # sanity check
    model, zjv, xes = create_milp_model(obj_type=params_sys.obj)
    model.write(f"{data_name}.mps.gz")
    model.optimize()
    fb = model.objval
    for k, trs in tqdm.tqdm(enumerate(mat_dict['trains'])):
        tr = trs['train']
        for e in tr.subgraph.es:
            if tr.traNo == 2:
                (r.xb[k][e.index] > 0.5) and print(e, r.xb[k][e.index])
            xes[tr.traNo][e['name']].setAttr(GRB.Attr.LB, r.xb[k][e.index])
            xes[tr.traNo][e['name']].setAttr(GRB.Attr.UB, r.xb[k][e.index])
    model.optimize()
    fx = model.objval

    print(f"""
:gurobi validation
- best objective: {fb}
- bcd           : {fx} (gurobi validated: {fx}) 
    """)