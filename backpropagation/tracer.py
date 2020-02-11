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

'''
deprecated
incomplete
'''
class ComplexTracer:
    def __init__(self):
        self.iterations = []
        self.container = {}
        self.iterations.append(self.container)
        self._run_key = "iteration"
        self._layer_key = "layer"
        self._count_key = "count"
        self.addTrace(self._run_key, 0)
        self.addTrace(self._count_key, 0)

    def _top_off(self, array, length):
        return array + [None] * (n - len(array))



    def addTrace(self, key, result):
        if key not in self.container:
            self.container[key] = true, [None] * self.getElement(self._count_key)
        self.container[key][1].append(result)
        if not self.container[key][0]:
            self.container[key] = true, self.container[key][1]
        else:
            count = self.getElement(self._count_key)
            for okey in self.container and okey != key:
                self.container[okey] = false, self._top_off(self.container[okey][1])

        return self

    def getElement(self, key, count=-1, iteration=-1):
        return self.iterations[iteration][key][1][count]

    def getColumns(self, *keys, iteration=-1):
        results = []
        for k in keys:
            results.append(self.iterations[iteration][key][1])
        return np.transpose(results)

    def getCurrentCount(self):
        return self.getElement(self._count_key)

    def endTrace(self):
        for key in self.container:
            self.container[key][1].append('-')
        self.addTrace(self._count_key, self.getElement(self._count_key) + 1)

    def nextLevel(self):
        self.count = self.count + 1

    def __repr__(self):
        for key in self.container:
            out = out + str(key) + ',\t'
        out = out + "\n\r"
        for iteration in self.iterations:
            