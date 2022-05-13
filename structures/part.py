from .machine import Machine


class Part(Machine):
    def __repr__(self):
        return f'Column(elements={self.arr}, index={self.index})'
