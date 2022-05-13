class Machine:
    def __init__(self, index, arr):
        self.index = index
        self.arr = tuple(arr)
        self.parent = None

    def __eq__(self, other):
        return self.arr == other.arr and self.index == other.index and type(self) == type(other)

    def __hash__(self):
        return hash(tuple((self.arr, self.index, self.__class__)))

    def __repr__(self):
        return f'Row(elements={self.arr}, index={self.index})'

    def __getitem__(self, item):
        return self.arr[item]
