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

    model_dict, global_index, model_index = create_decomposed_models(max_train_size=True)

    mat_dict = generate_matlab_dict(model_dict, global_index, model_index)
    savemat(f"ttp_{params_sys.train_size}_{params_sys.station_size}_{params_sys.time_span}.mat", mat_dict,
            do_compression=True)

    # sanity check by create an all-in-one model
    #   compare results to decomposed model.
    sanity_check = False
    # if sanity_check:
    #     model, zjv = create_milp_model()
    #     train_index_dict = split_model(model)
    #
    #     non_sum = 0
    #     var_sum = 0
    #     full_model_coup_constr_names = [constr.ConstrName for constr in su.getConstrByPrefix(model, "multi")]
    #     assert len(full_model_coup_constr_names) == len(set(full_model_coup_constr_names))
    #     full_model_coup_constr_names = set(full_model_coup_constr_names)
    #     for traNo in modelDict:
    #         sub_model_coup_constr_names = [constr.ConstrName for constr in
    #                                        su.getConstrByPrefix(modelDict[traNo], "multi")]
    #         assert len(sub_model_coup_constr_names) == len(set(sub_model_coup_constr_names))
    #         sub_model_coup_constr_names = set(sub_model_coup_constr_names)
    #
    #         assert len(full_model_coup_constr_names) == len(sub_model_coup_constr_names)
    #
    #         non_sum += modelDict[traNo].NumConstrs - len(full_model_coup_constr_names)
    #         var_sum += modelDict[traNo].NumVars
    #     assert model.NumConstrs == non_sum + len(full_model_coup_constr_names)
    #     assert var_sum == model.NumVars
