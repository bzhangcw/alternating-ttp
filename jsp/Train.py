class Train:
    def __init__(self, traNo):
        self.up = None
        self.traNo = traNo  # 列车车次
        self.dep_LB = None  # 始发时间窗下界
        self.dep_UB = None  # 始发时间窗上界
        self.depSta = None  # 出发站
        self.arrSta = None  # 到达站
        self.stopSta = []  # 经停站
        self.passSta = []  # 通过站
        self.staList = []  # 经停站或通过站
        self.delta = {}  # 运行附加时字典，键为(station, next station)
        self.linePlan = {}  # 开行方案字典
        self.nodeOccupancy = {}  # 是否占用某节点，key为sta => t
        self.timetable = {}  # 以virtual station为key，存int值
        self.speed = None  # 列车速度，300,350

    def decode_line_plan(self, g, h):
        all_station_list = list(self.linePlan.keys())
        for station in all_station_list:
            if self.linePlan[station] in {-1, 1}:
                self.depSta = station
                break
        for station in all_station_list[::-1]:
            if self.linePlan[station] in {-1, 1}:
                self.arrSta = station
                break

        in_rail_flag = 0
        for station in all_station_list:
            if station == self.depSta:
                in_rail_flag = 1

            if self.linePlan[station] == 1:
                self.stopSta.append(station)
                self.staList.append(station)
            if in_rail_flag == 1 and self.linePlan[station] in {0, -1}:
                self.passSta.append(station)
                self.staList.append(station)

            if station == self.arrSta:
                in_rail_flag = 0

        for station in self.staList:
            nextStation = str(int(station) + int(2 * self.up - 1))
            if nextStation not in self.staList:
                break

            if self.linePlan[station] in {0, -1} and self.linePlan[nextStation] in {0, -1}:
                self.delta[(station, nextStation)] = 0
            elif self.linePlan[station] in {0, -1} and self.linePlan[nextStation] == 1:
                self.delta[(station, nextStation)] = g[self.speed][nextStation]
            elif self.linePlan[station] == 1 and self.linePlan[nextStation] in {0, -1}:
                self.delta[(station, nextStation)] = h[self.speed][station]
            elif self.linePlan[station] == 1 and self.linePlan[nextStation] == 1:
                self.delta[(station, nextStation)] = h[self.speed][station] + g[self.speed][nextStation]
