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


class ComplexTracer(SimpleTracer):
    def __init__(self):
        super().__init__()
        self._iterations = []
        self._curr_iter = 0
        self._iterations.append([self.container, self.count])


    def nextIteration(self):
        self._iterations[self._curr_iter][1] = self.count # save current count
        self.container = {}
        self.count = 0
        self._iterations.append([self.container, self.count])
        self._curr_iter = len(self.container)
        return self

    def loadIteration(self, index=-1):
        self._iterations[self._curr_iter][1] = self.count # save current count
        self.container = self._iterations[index][0]
        self.count = self._iterations[index][1]
        self._curr_iter = index
        return self

    def __repr__(self):
        out = ''
        for index in range(len(self._iterations)):
            self.loadIteration(index)
            out = out + str(super.__repr__())
        return out