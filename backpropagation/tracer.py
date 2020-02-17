import numpy as np

class SimpleTracer:
    def __init__(self):
        self.container = {}
        # self.count = 0
        self._longest_key = None
        self._val_unavailable = "???"

    def addTrace(self, key, result):
        if key not in self.container:
            self.container[key] = []
        self.container[key].append(result)
        if (self._longest_key is None or len(self.container[key]) > len(self.container[self._longest_key])):
            self._longest_key = key
        return self

    def getElement(self, key, count=-1):
        return self.container[key][count]

    def _fill_key(self, key):
        if self._longest_key is None:
            if key in self.container:
                self._longest_key = key
        elif key in self.container:
            key_len = len(self.container[key])
            longest = len(self.container[self._longest_key])
            if longest > key_len:
                self.container[key] = self.container[key] + [self._val_unavailable] * (longest - key_len)

    def getColumns(self, *keys):
        results = []
        for k in keys:
            self._fill_key(k)
            results.append(self.container[k])
        return np.transpose(results)

    def endTrace(self):
        for key in self.container:
            self.addTrace(key, '-')

    def __repr__(self):
        out = ''
        for key in self.container:
            out = out + str(key) + '\t'
        out = out + "\n\r"
        for i in range(len(self.container[self._longest_key])):
            for key in self.container:
                if i >= len(self.container[key]):
                    out = out + self._val_unavailable + '\t'
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
        self._iterations.append([self.container, self._longest_key])


    def nextIteration(self):
        self._iterations[self._curr_iter][1] = self._longest_key # save current count
        self.container = {}
        self._longest_key = None
        self._iterations.append([self.container, self._longest_key])
        self._curr_iter = len(self.container)
        return self

    def loadIteration(self, index=-1):
        self._iterations[self._curr_iter][1] = self._longest_key # save current count
        self.container = self._iterations[index][0]
        self._longest_key = self._iterations[index][1]
        self._curr_iter = index
        return self

    # @classmethod
    def __repr__(self):
        out = ''
        for index in range(len(self._iterations)):
            self.loadIteration(index)
            out = out + str(super().__repr__())
        return out
