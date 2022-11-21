from dataLoader import read_train, read_route_conflict_info, filter_meso_info
from util import *
from Meso import Meso

if __name__ == "__main__":
    params_sys = SysParams()
    params_subgrad = SubgradParam()

    params_sys.parse_environ()
    params_subgrad.parse_environ()

    station_size = params_sys.station_size
    train_size = params_sys.train_size
    time_span = params_sys.time_span
    iter_max = params_sys.iter_max

    train_list = read_train('raw_data/7-lineplan-all.xlsx', train_size, time_span, up=params_sys.down)
    route_index, route_conflicts, _, _, routes, reachable_boundary_track_route_pairs = \
        read_route_conflict_info('raw_data/RouteConflictInfo.txt')
    station_boundaries = {'站界:B8', '站界:B9'}
    tracks = {'站线:8', '站线:9', '站线:10', '站线:11', '站线:XII', '站线:XIII', '站线:XIV', '站线:XV', '站线:16',
              '站线:17', '站线:18', '站线:19'}
    station_platform_dict = {'1': tracks}
    route_index, \
    route_conflicts, \
    routes, \
    reachable_boundary_track_route_pairs = filter_meso_info(route_index,
                                                            route_conflicts,
                                                            routes,
                                                            reachable_boundary_track_route_pairs,
                                                            station_boundaries,
                                                            tracks)
    tr_delta = [3] * len(train_list)   # TODO: different delta for each train
    meso = Meso(train_list, tr_delta, station_boundaries, tracks, routes, route_conflicts)
    meso.run()
