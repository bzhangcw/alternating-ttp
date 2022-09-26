"""
create a railway ttp instance by gurobi
    also save to matlab & mps.
"""
from scipy.io import savemat
from functional_model_builder import *
from functional_bcd import *

if __name__ == '__main__':
    params_sys = SysParams()
    params_subgrad = SubgradParam()

    params_sys.parse_environ()
    params_subgrad.parse_environ()

    ms.setup(params_sys)

    model_dict, global_index, model_index = create_decomposed_models()

    mat_dict = dict()
    mat_dict['trains'] = []
    mat_dict['b'] = 0
    for traNo in sorted(model_dict.keys()):
        A_k, B_k, b_k, c_k, b, sense_B_k, sense_A_k = getA_b_c(model_dict[traNo], model_index[traNo], len(global_index))
        struct = {
            "A": A_k,
            "B": B_k,
            "b": b_k,
            "c": c_k,
            "sense_A_k": sense_A_k,
            "sense_B_k": sense_B_k,
            "binding": len(model_index[traNo])
        }
        #
        mat_dict['b'] = b
        mat_dict['trains'].append(struct)
        print(f"traNo:{traNo}, A_k: {A_k.shape}, B_k: {B_k.shape}, b_k: {b_k.shape}, c_k: {c_k.shape}\n"
              f"real binding size: {len(model_index[traNo])}")
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