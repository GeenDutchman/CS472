import numpy as np

class SimpleTracer:
    def __init__(self):
        self.active = True
        self.container = {}
        # self.count = 0
        self._longest_key = None
        self._val_unavailable = "???"

    def setActive(self, active=False):
        self.active = active
        return self

    def addTrace(self, key, result):
        if self.active:
            if key not in self.container:
                fill_len =  0 if self._longest_key is None else len(self.container[self._longest_key]) - 1
                self.container[key] = [self._val_unavailable] * fill_len
            self.container[key].append(result)
            key_len = len(self.container[key])
            if self._longest_key is None or key_len > len(self.container[self._longest_key]):
                self._longest_key = key
        return self

    def getElement(self, key, count=-1):
        return self.container[key][count]

    def _fill_key(self, key):
        if self.active:
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

    def nextLevel(self):
        if self.active:
            for k in self.container:
                self._fill_key(k)

    def endTrace(self):
        if self.active:
            for key in self.container:
                self.addTrace(key, '-')

    def __repr__(self):
        if self._longest_key is None:
            return 'Trace is empty from here\r\n'
        out = ''
        tabs = '|'
        newline = '\n\r'
        if not self.active:
            out = out + "inactive" + tabs + newline
        for key in self.container:
            out = out + str(key) + tabs
        out = out + newline
        for i in range(len(self.container[self._longest_key])):
            for key in self.container:
                if i >= len(self.container[key]):
                    out = out + self._val_unavailable + tabs
                else:
                    out = out + str(self.container[key][i]).replace('\n', '') + tabs
            out = out + newline
        out = out + newline
        return out
        


class ComplexTracer(SimpleTracer):
    def __init__(self):
        super().__init__()
        self._iterations = []
        self._curr_iter = 0
        self._iterations.append([self.container, self._longest_key])


    def nextIteration(self):
        if self.active:
            self._iterations[self._curr_iter][1] = self._longest_key # save current count
            self._iterations.append([dict(), None])
            self.container = self._iterations[-1][0]
            self._longest_key = self._iterations[-1][1]
            self._curr_iter = len(self._iterations) - 1
        return self

    def loadIteration(self, index=-1):
        self._iterations[self._curr_iter][1] = self._longest_key # save current count
        self.container = self._iterations[index][0]
        self._longest_key = self._iterations[index][1]
        self._curr_iter = index
        return self

    def iteration_to_string(self, index=-1):
        out = ''
        old_index = self._curr_iter
        self.loadIteration(index)
        out = out + str(super().__repr__())
        self.loadIteration(old_index)
        return out

    # @classmethod
    def __repr__(self):
        out = ''
        for index in range(len(self._iterations)):
            self.loadIteration(index)
            out = out + str(super().__repr__())
        return out

        
