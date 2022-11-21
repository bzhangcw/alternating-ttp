solver = "gurobi"
if solver == "gurobi":
    from gurobipy import Model, tupledict, GRB, quicksum, tuplelist
else:
    pass


class Meso:
    def __init__(self, trains, tr_delta, bounds, platforms, routes, route_conflicts):
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
        self.Fi = [i for i in self.F if self.trains[i].in_sta]  # FIXME
        self.Fo = [i for i in self.F if not self.trains[i].in_sta]  # FIXME

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

    @staticmethod
    def create_model(*args, **kwargs):
        if solver == "gurobi":
            return Model(*args, **kwargs)

    def addVars(self):
        J_set = set()
        Ji_set = set()
        Jo_set = set()

        for f in self.F:
            tr = self.trains[f]
            R = self.Ri if tr.in_sta else self.Ro  # TODO: assert in_sta
            for r in R:
                t0 = tr.macro_time  # TODO: assert macro_time
                delta = self.tr_delta[f]
                region = [t for t in range(t0 - delta, t0 + delta + 1)]  # FIXME: push to (0, timespan)
                for t in region:
                    assert (r, t, f) not in self.x
                    self.x[r, t, f] = self.m.addVar(vtype=GRB.BINARY, name=f"x[{r},{t},{f}]")

                    J_set.add((r, t))
                    if r in self.Ri:
                        Ji_set.add((r, t))
                    else:
                        assert r in self.Ro
                        Jo_set.add((r, t))

        self.J = tuplelist(J_set)
        self.Ji = tuplelist(Ji_set)
        self.Jo = tuplelist(Jo_set)

    def addConstrs(self):
        self.addConstrsMatching()
        self.addConstrsPhyConflict()
        self.addConstrsInin()
        self.addConstrsOutout()
        self.addConstrsInOut()
        self.addConstrsOutin()
        self.addConstrsOccup()

    def addConstrsMatching(self):
        self.matching = self.m.addConstrs(
            (self.x.sum('*', '*', f) <= 1
             for f in self.F
             ),
            name="Match"
        )

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

    def addConstrsInin(self):
        delta_2 = 5  # FIXME

        for r, t in self.Ji:
            nodes = [self.x.sum(nei, t_nei, '*')
                     for nei in self.Ri
                     for t_nei in range(t, t + delta_2)]
            self.inin[r, t] = self.m.addConstr(
                quicksum(nodes) <= 1,
                name=f"Inin[{r},{t}]"
            )

    def addConstrsOutout(self):
        delta_3 = 5  # FIXME

        for r, t in self.Jo:
            nodes = [self.x.sum(nei, t_nei, '*')
                     for nei in self.Ro
                     for t_nei in range(t, t + delta_3)]
            self.outout[r, t] = self.m.addConstr(
                quicksum(nodes) <= 1,
                name=f"Outout[{r},{t}]"
            )

    def addConstrsInOut(self):
        delta_4 = 5  # FIXME

        for r, t in self.Ji:
            nodes = [self.x.sum(nei, t_nei, '*')
                     for nei in self.Ro
                     for t_nei in range(t, t + delta_4)]
            self.inout[r, t] = self.m.addConstr(
                quicksum(nodes) <= 1,
                name=f"Inout[{r},{t}]"
            )

    def addConstrsOutin(self):
        delta_5 = 5  # FIXME

        for r, t in self.Jo:
            nodes = [self.x.sum(nei, t_nei, '*')
                     for nei in self.Ri
                     for t_nei in range(t, t + delta_5)]
            self.outin[r, t] = self.m.addConstr(
                quicksum(nodes) <= 1,
                name=f"Outin[{r},{t}]"
            )

    def addConstrsOccup(self):
        pass

    def setObjective(self):
        self.m.setObjective(self.x.sum(), sense=GRB.MAXIMIZE)

    def optimize(self):
        if solver == "gurobi":
            self.m.optimize()

    def run(self):
        self.addVars()
        self.addConstrs()
        self.setObjective()
        self.optimize()
