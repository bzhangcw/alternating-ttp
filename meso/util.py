"""
utility modules
"""
from functools import wraps
from pprint import pprint
from time import time

timeit = True


def timing(f):
    if timeit:
        @wraps(f)
        def wrap(*args, **kw):
            ts = time()
            result = f(*args, **kw)
            te = time()
            print(f'Function {f.__name__} took {te - ts:2.4f} seconds')
            return result
    else:
        wrap = f
    return wrap


class SysParams(object):
    # todo, support ArgumentParser
    # currently stay with env.
    DBG = False
    station_size = 0
    train_size = 0
    time_span = 0
    iter_max = 0
    down = 0
    fix_preferred_time = True

    def parse_environ(self):
        import os
        self.station_size = int(os.environ.get('station_size', 29))
        self.train_size = int(os.environ.get('train_size', 4))
        self.time_span = int(os.environ.get('time_span', 1080))
        self.iter_max = int(os.environ.get('iter_max', 100))
        self.down = int(os.environ.get('down', -1))
        self.fix_preferred_time = int(os.environ.get('fix_preferred_time', True))

    def show(self):

        pprint(self.__dict__)
