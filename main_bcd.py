"""
main for bcd
"""
from functional_model_builder import *
from functional_bcd import *
import data as ms
from util import SysParams




if __name__ == '__main__':
    # example usage.
    params_sys = SysParams()
    params_subgrad = SubgradParam()
    params_bcd = BCDParams()

    params_sys.parse_environ()
    params_subgrad.parse_environ()
    params_bcd.parse_environ()

    ms.setup(params_sys)

    model_dict, global_index, model_index = create_decomposed_models()


    mat_dict = generate_matlab_dict(model_dict, global_index, model_index)

    # scipy.io.savemat(f"ttp_{params_sys.train_size}_{params_sys.station_size}_{params_sys.time_span}.mat", mat_dict,
    #         do_compression=True)
    optimize(bcdpar=params_bcd, mat_dict=mat_dict)