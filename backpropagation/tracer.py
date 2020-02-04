import numpy as np

class SimpleTracer:
    def __init__(self):
        self.container = {}
        self.count = 0

    def addTrace(self, key, result):
        if key not in self.container:
            self.container[key] = []
        self.container[key].append(result)
        return self

    def getElement(self, key, count=-1):
        if count == -1:
            count = self.count
        return self.container[key][count]

    def getColumns(self, *keys):
        results = []
        for c in range(self.count - 1):
            this_row = []
            for k in keys:
                if c >= len(self.container[k]):
                    return np.array(results)
                else:
                    this_row.append(self.container[k][c])
            results.append(this_row)
        return np.array(results)

    def getCurrentCount(self):
        return self.count

    def endTrace(self):
        for key in self.container:
            self.container[key].append('-')
        self.count = self.count + 1

    def nextLevel(self):
        self.count = self.count + 1

    def __repr__(self):
        out = ''
        for key in self.container:
            out = out + str(key) + '\t'
        out = out + "\n\r"
        for i in range(self.count):
            for key in self.container:
                if i >= len(self.container[key]):
                    out = out + "???" + '\t'
                else:
                    out = out + str(self.container[key][i]) + '\t'
            out = out + "\n\r"
        out = out + "\n\r"
        return out


