from itertools import count
from .machine import Machine


class Cell:
    _ids = count(0)

    def __init__(self):
        self.machines = set()
        self.parts = set()
        self.index = f'x_{next(self._ids)}'
        self.priority = 0

    def __repr__(self):
        return f'Cell({self.get_machine_indices()}, {self.get_part_indices()})'

    def __eq__(self, other):
        return sorted(self.get_machine_indices()) == sorted(other.get_machine_indices()) and \
               sorted(self.get_part_indices()) == sorted(other.get_part_indices())

    def machines_count(self):
        return len(self.machines)

    def parts_count(self):
        return len(self.parts)

    def get_machine_indices(self):
        return sorted([row.index for row in self.machines])

    def get_part_indices(self):
        return sorted([col.index for col in self.parts])

    def add(self, obj):
        if obj.parent:
            obj.parent.remove(obj)
        obj.parent = self
        if type(obj) == Machine:
            self.machines.add(obj)
        else:
            self.parts.add(obj)

    def remove(self, obj):
        obj.parent = None
        if type(obj) == Machine:
            self.machines.remove(obj)
        else:
            self.parts.remove(obj)

    def info(self):
        col_indices = self.get_part_indices()
        n0 = 0
        n1 = 0
        for row in self.machines:
            elems = [row[index] for index in col_indices]
            for elem in elems:
                if elem:
                    n1 += 1
                else:
                    n0 += 1
        return n1, n0
