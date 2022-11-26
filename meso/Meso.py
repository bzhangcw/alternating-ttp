import logging

from Train import Train
from meso_rules import MesoRules
from util import timing

solver = "copt"
if solver == "copt":
    from coptpy import Envr, tupledict, tuplelist, quicksum, COPT as CONST
else:
    from gurobipy import Model, tupledict, tuplelist, quicksum, GRB as CONST

logging.basicConfig()
_grb_logger = logging.getLogger("gurobipy.gurobipy")
_grb_logger.setLevel(logging.ERROR)

logFormatter = logging.Formatter("%(asctime)s: %(message)s")
_logger = logging.getLogger("meso-system-scheduler")
_logger.setLevel(logging.INFO)


class Meso:
    """
    The meso model of a single station
    """
    def __init__(
            self,
            trains,
            tr_delta,
            bounds,
            platforms,
            routes,
            route_conflicts,
            rules: MesoRules,
            station: int = 0
    ):
        self.trains = trains
        assert len(tr_delta) == len(trains)
        self.tr_delta = tr_delta
        self.bounds = bounds
        self.platforms = platforms
        self.routes = sorted(routes, key=lambda x: x[1])
        self.route_conflicts = route_conflicts
        self.route_names = list(self.route_conflicts.keys())
        self.r2name = {r: n for r, n in enumerate(self.route_names)}
        self.name2r = {n: r for r, n in enumerate(self.route_names)}
        self.r_conf = {r: {self.name2r[r2_name] for r2_name in self.route_conflicts[self.route_names[r]]}
                       for r in range(len(self.routes))}

        self.F = range(len(self.trains))
        self.Fi = [i for i in self.F if self.trains[i].bool_in_station]  # FIXME
        self.Fo = [i for i in self.F if not self.trains[i].bool_in_station]  # FIXME

        self.R = range(len(self.routes))
        self.Ri = [i for i in self.R if 'B8' in self.routes[i][1]]  # FIXME
        self.Ro = [i for i in self.R if 'B8' not in self.routes[i][1]]  # FIXME

        self.J = tuplelist()
        self.Ji = tuplelist()
        self.Jo = tuplelist()

        self.m = self.create_model()

        self.x = tupledict()

        self.matching = tupledict()
        self.phyconf = tupledict()
        self.inin = tupledict()
        self.outout = tupledict()
        self.inout = tupledict()
        self.outin = tupledict()

        self.station = station
        self.rules: MesoRules = rules
        self.show()

    def show(self):
        _logger.info(f"problem basic description:")
        _logger.info(f"we have:")
        _logger.info(f"- # of trains: {self.trains.__len__()}")

    @staticmethod
    def create_model(*args, **kwargs):
        if solver == "copt":
            return Envr().createModel(*args, **kwargs)
        else:
            return Model(*args, **kwargs)

    @timing
    def addVars(self):
        J_set = set()
        Ji_set = set()
        Jo_set = set()

        for f in self.F:
            tr:Train = self.trains[f]
            R = self.Ri if tr.bool_in_station else self.Ro  # TODO: assert in_sta
            for r in R:
                t0 = tr.macro_time  # TODO: assert macro_time
                delta = self.tr_delta[f]
                region = [t for t in range(t0 - delta, t0 + delta + 1)]  # FIXME: push to (0, timespan)
                for t in region:
                    assert (r, t, f) not in self.x
                    self.x[r, t, f] = self.m.addVar(vtype=CONST.BINARY, name=f"x[{r},{t},{f}]")

                    J_set.add((r, t))
                    if r in self.Ri:
                        Ji_set.add((r, t))
                    else:
                        assert r in self.Ro
                        Jo_set.add((r, t))

        self.J = tuplelist(J_set)
        self.Ji = tuplelist(Ji_set)
        self.Jo = tuplelist(Jo_set)

    @timing
    def addConstrs(self):
        self.addConstrsMatching()
        self.addConstrsPhyConflict()
        self.addConstrsInin()
        self.addConstrsOutout()
        self.addConstrsInOut()
        self.addConstrsOutin()
        self.addConstrsOccup()

    @timing
    def addConstrsMatching(self):
        for f in self.F:
            self.matching = self.m.addConstr(
                self.x.sum('*', '*', f) <= 1,
                name=f"Match[{f}]"
            )

    @timing
    def addConstrsPhyConflict(self):
        delta_1 = 5  # FIXME: delta_1
        for r, t in self.J:
            conflicts = list(self.r_conf[r])
            neighbour = conflicts + [r]

            nodes = [self.x.sum(nei, t_nei, '*')
                     for nei in neighbour
                     for t_nei in range(t, t + delta_1)]
            self.phyconf[r, t] = self.m.addConstr(
                quicksum(nodes) <= 1,
                name=f"Phy[{r},{t}]"
            )

    @timing
    def addConstrsInin(self):
        """
        arrival-arrival constraints
        到到约束，包括同向/异向，
        Returns:

        """

        delta_2 = self.rules.INT_AA

        for r, t in self.Ji:
            nodes = [self.x.sum(nei, t_nei, '*')
                     for nei in self.Ri
                     for t_nei in range(t, t + delta_2)]
            self.inin[r, t] = self.m.addConstr(
                quicksum(nodes) <= 1,
                name=f"Inin[{r},{t}]"
            )

    @timing
    def addConstrsOutout(self):
        # D - D
        delta_3 = self.rules.INT_DD

        for r, t in self.Jo:
            nodes = [self.x.sum(nei, t_nei, '*')
                     for nei in self.Ro
                     for t_nei in range(t, t + delta_3)]
            self.outout[r, t] = self.m.addConstr(
                quicksum(nodes) <= 1,
                name=f"Outout[{r},{t}]"
            )

    @timing
    def addConstrsInOut(self):
        # A - D
        delta_4 = self.rules.INT_OPPO_AD

        for r, t in self.Ji:
            nodes = [self.x.sum(nei, t_nei, '*')
                     for nei in self.Ro
                     for t_nei in range(t, t + delta_4)]
            self.inout[r, t] = self.m.addConstr(
                quicksum(nodes) <= 1,
                name=f"Inout[{r},{t}]"
            )

    @timing
    def addConstrsOutin(self):
        # D - A
        delta_5 = self.rules.INT_DA

        for r, t in self.Jo:
            nodes = [self.x.sum(nei, t_nei, '*')
                     for nei in self.Ri
                     for t_nei in range(t, t + delta_5)]
            self.outin[r, t] = self.m.addConstr(
                quicksum(nodes) <= 1,
                name=f"Outin[{r},{t}]"
            )

    @timing
    def addConstrsOccup(self):
        pass

    def setObjective(self):
        self.m.setObjective(self.x.sum(), sense=CONST.MAXIMIZE)

    @timing
    def optimize(self):
        if solver == "copt":
            self.m.solve()
        else:
            self.m.optimize()

    @timing
    def run(self):
        self.addVars()
        self.addConstrs()
        self.setObjective()
        self.optimize()

    def resolveIIS(self):
        self.m.computeIIS()
        for constr in self.m.getConstrs():
            if constr.IISConstr:
                _logger.info(constr.ConstrName)
