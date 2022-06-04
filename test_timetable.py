"""
A script to test if the timetable if valid.
"""

import main_subgrad as ms
from main_subgrad import *
from util_output import *

import sys

START_PERIOD = 360
if __name__ == '__main__':
    fp = sys.argv[1]

    params_sys = SysParams()
    station_size = params_sys.station_size = int(os.environ.get('station_size', 29))
    train_size = params_sys.train_size = int(os.environ.get('train_size', 292))
    time_span = params_sys.time_span = int(os.environ.get('time_span', 1080))
    iter_max = params_sys.iter_max = int(os.environ.get('iter_max', 100))
    primal = os.environ.get('primal', 'jsp')
    dual = os.environ.get('dual', 'pdhg')
    # create result-folder.
    subdir_result = params_sys.subdir_result = datetime.datetime.now().strftime('%y%m%d-%H%M')
    fdir_result = params_sys.fdir_result = f"result/{subdir_result}"
    os.makedirs(fdir_result, exist_ok=True)
    logger.info(f"size: #train,#station,#timespan,#iter_max: {train_size, station_size, time_span, iter_max}")
    read_station('raw_data/1-station.xlsx', station_size)
    read_station_stop_start_addtime('raw_data/2-station-extra.xlsx')
    read_section('raw_data/3-section-time.xlsx')
    read_dwell_time('raw_data/4-dwell-time.xlsx')
    read_train('raw_data/7-lineplan-up.xlsx', train_size, time_span)
    read_intervals('raw_data/5-safe-intervals.xlsx')

    '''
    initialization
    '''
    logger.info("reading finish")
    logger.info("step 1")
    # initialize_node_precedence(time_span)
    logger.info(f"maximum estimate of active nodes {gc.vc}")

    # for tr in train_list:
    #     tr.create_subgraph(sec_times_all[tr.speed], time_span)

    logger.info("step 2")
    logger.info(f"actual train size {len(train_list)}")

    df, train_paths = read_timetable_csv(fp, st=START_PERIOD, station_name_map=ms.station_name_map)
    for tr in ms.train_list:
        _p = train_paths[tr.traNo]
        tr.is_best_feasible = True
        tr.timetable = dict(_p)

    # params_subgrad = SubgradParam()
    # params_subgrad.dual_method = dual
    # params_subgrad.primal_heuristic_method = primal
    # params_subgrad.max_number = len(train_paths)
    #
    # selective = None
    #     ms.train_list, ms.miles, ms.station_list,
    #     param_sys=params_sys,
    #     param_subgrad=params_subgrad,
    #     selective=selective
    # )
