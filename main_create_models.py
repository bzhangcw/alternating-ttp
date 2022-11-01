"""
create a railway ttp instance by gurobi
    also save to matlab & mps.
"""
from scipy.io import savemat

from functional_model_builder import *

if __name__ == '__main__':
    params_sys = SysParams()
    params_subgrad = SubgradParam()

    params_sys.parse_environ()
    params_subgrad.parse_environ()

    ms.setup(params_sys)

    model_dict, global_index, model_index = create_decomposed_models(obj_type=params_sys.obj)

    mat_dict = generate_matlab_dict(model_dict, global_index, model_index, dump=True)
    savemat(f"ttp_{params_sys.train_size}_{params_sys.station_size}_{params_sys.time_span}.mat", mat_dict,
            do_compression=True)

    # sanity check by create an all-in-one model
    #   compare results to decomposed model.
    sanity_check = True
    if sanity_check:
        model, zjv = create_milp_model(obj_type=params_sys.obj)
        optimize(model)
