import sys

import dataLoader as tppdl
from Meso import Meso
from meso_rules import MesoRules
from util import SysParams

# python main *.csv

if __name__ == "__main__":
    params_sys = SysParams()
    params_sys.parse_environ()
    params_sys.show()
    tppdl.setup(params_sys)


    # todo: the meso should be defined on a station?
    # in this script, use 0 - 北京南
    station_num = 0
    station_name = tppdl.station_name_list[station_num]
    fp_macro_sol = sys.argv[1]
    tppdl.read_local_macro_solution(fp_macro_sol, station_name=station_name)
    train_list_selected = [tr for tr in tppdl.train_list if tr.macro_time is not None]

    # start using a fixed macro solution
    meso_rules = MesoRules()
    tr_delta = [3] * len(train_list_selected)  # TODO: different delta for each train
    meso = Meso(
        train_list_selected,
        tr_delta,
        tppdl.station_boundaries,
        tppdl.tracks,
        tppdl.routes,
        tppdl.route_conflicts,
        rules=meso_rules,
        station=station_num
    )

    meso.run()
