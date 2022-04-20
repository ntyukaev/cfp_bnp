import random
from itertools import count
import numpy as np
import copy


class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix
        self.rows_count = len(matrix)
        self.columns_count = len(matrix[0])
        self.rows = set([Row(index, row) for index, row in enumerate(self.matrix)])
        self.columns = set([Column(index, col) for index, col in enumerate(np.transpose(self.matrix))])
        self.flat = self.matrix.flatten().astype(float)
        self.n0 = len(list(filter(lambda x: x == 0, self.flat)))
        self.n1 = len(self.flat) - self.n0
        self.efficacy = 0
        self.cells = set()

    def get_efficacy(self, cells):
        n1_in = 0
        n0_in = 0
        for cell in cells:
            n1_in_cell, n0_in_cell = cell.info()
            n1_in += n1_in_cell
            n0_in += n0_in_cell
        return n1_in / (self.n1 + n0_in)

    def get_pool(self):
        pool = set()
        pool.update([row for row in self.rows])
        pool.update([col for col in self.columns])
        return pool

    def populate_cells(self):
        # the max amount of cells is equal to the min of rows & cols
        cells = [Cell() for _ in range(min([self.rows_count, self.columns_count]))]
        # distribute rows and columns between cells
        for row in self.rows:
            cell = random.choice(cells)
            cell.add(row)

        for col in self.columns:
            cell = random.choice(cells)
            cell.add(col)
        # calculate efficacy
        efficacy = self.get_efficacy(cells)

        # try improving efficacy by moving rows and columns between cells
        pool = self.get_pool()
        while pool:
            element = random.sample(pool, 1)[-1]
            # remove element from parent
            parent = element.parent
            random.shuffle(cells)
            for cell in cells:
                cell.add(element)
                # calculate efficacy
                current_efficacy = self.get_efficacy(cells)
                # if it's better than previous than redo
                if current_efficacy > efficacy:
                    efficacy = current_efficacy
                    pool = self.get_pool()
                    break
            else:
                parent.add(element)
                pool.remove(element)

        self.efficacy = efficacy
        cells = [cell for cell in cells if cell.rows or cell.columns]
        # create a cell containing all rows and columns
        self.cells.update(cells)


class Cell:
    _ids = count(0)

    def __init__(self):
        self.rows = set()
        self.columns = set()
        self.index = next(self._ids)

    def __repr__(self):
        return f'Cell({self.get_rows_indices()}, {self.get_columns_indices()})'

    def get_rows_indices(self):
        return [row.index for row in self.rows]

    def get_columns_indices(self):
        return [col.index for col in self.columns]

    def add(self, obj):
        if obj.parent:
            obj.parent.remove(obj)
        obj.parent = self
        if type(obj) == Row:
            self.rows.add(obj)
        else:
            self.columns.add(obj)

    def remove(self, obj):
        if type(obj) == Row:
            self.rows.remove(obj)
        else:
            self.columns.remove(obj)

    def info(self):
        col_indices = self.get_columns_indices()
        n0 = 0
        n1 = 0
        for row in self.rows:
            elems = [row[index] for index in col_indices]
            for elem in elems:
                if elem:
                    n1 += 1
                else:
                    n0 += 1
        return n1, n0


class Row:
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


class Column(Row):
    def __repr__(self):
        return f'Column(elements={self.arr}, index={self.index})'


def main():
    from utils import read
    example = read('examples/1.txt')
    matrix = Matrix(example)
    matrix.populate_cells()


if __name__ == '__main__':
    main()
