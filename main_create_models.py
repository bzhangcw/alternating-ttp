"""
create a railway ttp instance by gurobi
    also save to matlab & mps.
"""
from scipy.io import savemat
import pickle as pk
from functional_model_builder import *

bool_create_mat = False
bool_sanity_check = True
if __name__ == '__main__':
    params_sys = SysParams()
    params_subgrad = SubgradParam()

    params_sys.parse_environ()
    params_subgrad.parse_environ()

    ms.setup(params_sys)
    # temporary path
    tmp = "./tmp"
    if not os.path.exists("./tmp/"):
        os.mkdir("./tmp")
    data_name = f"{tmp}/ttp_{params_sys.train_size}_{params_sys.station_size}_{params_sys.time_span}"
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
            
    if bool_create_mat:
        model_dict, global_index, model_index = create_decomposed_models(obj_type=params_sys.obj)

        mat_dict = generate_matlab_dict(model_dict, global_index, model_index, dump=True)
        savemat(f"ttp_{params_sys.train_size}_{params_sys.station_size}_{params_sys.time_span}.mat", mat_dict,
                do_compression=True)

    # sanity check by create an all-in-one model
    #   compare results to decomposed model.
    if bool_sanity_check:
        model, zjv, xes, s_arcs = create_milp_model(obj_type=params_sys.obj)
        model.write(f"{data_name}.mps.gz")
