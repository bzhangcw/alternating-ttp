"""
main for bcd
"""
from functional_model_builder import *
from functional_bcd import *
import data as ms



if __name__ == '__main__':
    # example usage.
    params_sys = SysParams()
    params_sys.parse_environ()
    params_bcd = BCDParams()
    params_bcd.parse_environ()

    ms.setup(params_sys)

    model_dict, global_index, model_index = create_decomposed_models()
    mat_dict = dict()
    mat_dict['trains'] = []
    mat_dict['b'] = 0
    for idx, traNo in enumerate(sorted(model_dict.keys())):
        A_k, B_k, b_k, c_k, b, sense_B_k, sense_A_k = getA_b_c(model_dict[traNo], model_index[traNo], len(global_index))
        _, n = B_k.shape
        struct = {
            "A": A_k,
            "B": B_k,
            "b": b_k,
            "c": c_k,
            "sense_A_k": sense_A_k,
            "sense_B_k": sense_B_k,
            "binding": len(model_index[traNo]),
            "id": traNo,
            "train": ms.train_list[idx],
            "n": n,
        }
        #
        mat_dict['b'] = b
        mat_dict['trains'].append(struct)
        print(f"traNo:{traNo}, A_k: {A_k.shape}, B_k: {B_k.shape}, b_k: {b_k.shape}, c_k: {c_k.shape}\n"
              f"real binding size: {len(model_index)}")

    # scipy.io.savemat(f"ttp_{params_sys.train_size}_{params_sys.station_size}_{params_sys.time_span}.mat", mat_dict,
    #         do_compression=True)
    optimize(bcdpar=params_bcd, mat_dict=mat_dict)