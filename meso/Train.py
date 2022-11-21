class Train(object):
    def __init__(self, traNo, in_sta, macro_time):
        self.traNo = traNo  # 列车车次
        self.in_sta = in_sta  # into or outside station
        self.macro_time = macro_time

    def __hash__(self):
        return self.traNo.__hash__()

    def __eq__(self, other):
        return self.traNo == other.traNo

    def __repr__(self):
        return "train" + str(self.traNo)
