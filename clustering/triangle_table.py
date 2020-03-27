class TriangleTable():
    _first_coord = {}

    def directional_add(self, key1, key2, value):
        if key1 not in self._first_coord.keys():
            self._first_coord[key1] = {}
        self._first_coord[key1][key2] = value

    def add(self, key1, key2, value):
        if key1 == key2:
            raise KeyError("Keys cannot be the same")
        self.directional_add(key1, key2, value)
        self.directional_add(key2, key1, value)

    def has_entry(self, key1, key2):
        if key1 not in self._first_coord.keys():
            return False, key1
        if key2 not in self._first_coord[key1].keys():
            return False, key2
        return True

    def get(self, key1, key2):
        return self._first_coord[key1][key2]

    def __repr__(self):
        return str(self._first_coord)