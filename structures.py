import numpy as np
from numpy import dot
from numpy.linalg import norm
import random


class Status:
    UNVISITED = 0
    VISITED = 1


class Types:
    ROW = 0
    COLUMN = 1


class CellList:
    def __init__(self, matrix):
        self.matrix = matrix
        self.flat = matrix.flatten().astype(float)
        self.n0 = len(list(filter(lambda x: x == 0, self.flat)))
        self.n1 = len(self.flat) - self.n0
        self.rows_count = len(matrix)
        self.column_count = len(matrix[0])
        self.cells = set()
        self.efficacy = self.get_efficacy()

    def generate_cells(self):
        self.get_initial_cells()
        self.local_search()

    def get_efficacy(self):
        n1_in = 0
        n0_in = 0
        for cell in self.cells:
            n1_in_cell, n0_in_cell = cell.info()
            n1_in += n1_in_cell
            n0_in += n0_in_cell
        return n1_in / (self.n1 + n0_in)

    def local_search(self):
        # pick a random element and try to put it somewhere else
        # calculate efficacy if it improves efficacy then good
        # do it until we reach the max efficacy

        # now try adding it to the cells
        pool = self.get_pool()
        while pool:
            element = random.sample(pool, 1)[-1]
            for cell in self.cells:
                cell.add(element)
                # calculate efficacy
                current_efficacy = self.get_efficacy()
                # if it's better than previous than redo
                if current_efficacy > self.efficacy:
                    self.efficacy = current_efficacy
                    # change element's parent
                    element.parent = cell
                    pool = self.get_pool()
                else:
                    cell.remove(element)
            element.parent.add(element)
            pool.remove(element)

    def get_pool(self):
        pool = set()
        for cell in self.cells:
            pool.update([row for row in cell.rows])
            pool.update([col for col in cell.columns])
        return pool

    def get_initial_cells(self):
        # get an initial set of cells
        rows = set()
        for index, row in enumerate(self.matrix):
            rows.add(Row(index, row))
        columns = set()
        for index, column in enumerate(np.transpose(self.matrix)):
            columns.add(Column(index, column))
        while rows:
            # create the first cell
            cell = Cell()
            self.cells.add(cell)
            # pick the best row
            best_row = max(rows)
            best_row_ones_indices = [index for index, val in enumerate(best_row) if val]
            cell.add(best_row)
            rows.remove(best_row)
            for column in columns.copy():
                if column.index in best_row_ones_indices:
                    cell.add(column)
                    columns.remove(column)
            # try adding more rows
            for row in rows.copy():
                a = [best_row[index] for index in best_row_ones_indices]
                b = [row[index] for index in best_row_ones_indices]
                cos_sim = dot(a, b) / (norm(a) * norm(b))
                if cos_sim >= 0.85:
                    rows.remove(row)
                    cell.add(row)
            self.cells.add(cell)
        # calculate efficacy
        self.efficacy = self.get_efficacy()


class Cell:
    def __init__(self):
        self.rows = set()
        self.columns = set()
        self.status = Status.UNVISITED

    def info(self):
        col_indices = [col.index for col in self.columns]
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

    def zeros(self):
        return sum([row.zeros for row in self.rows]) + sum([col.zeros for col in self.columns])

    def add(self, obj):
        if not obj.parent:
            obj.parent = self
        if obj.__class__ == Row:
            if obj.parent and obj in obj.parent.rows:
                obj.parent.rows.remove(obj)
            self.rows.add(obj)
        else:
            if obj.parent and obj in obj.parent.columns:
                obj.parent.columns.remove(obj)
            self.columns.add(obj)

    def remove(self, obj):
        if obj.__class__ == Row:
            self.rows.remove(obj)
        else:
            self.columns.remove(obj)


class Row:
    t = Types.ROW

    def __init__(self, index, arr):
        self.index = index
        self.arr = arr
        self.ones = sum(self.arr)
        self.zeros = len(self.arr) - self.ones
        self.status = Status.UNVISITED
        self.parent = None

    def __gt__(self, other):
        return self.ones > other.ones

    def __repr__(self):
        return f'Row({self.arr})'

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        return self.arr.__iter__()

    def __getitem__(self, index):
        return self.arr[index]


class Column(Row):
    t = Types.COLUMN

    def __repr__(self):
        return f'Column({self.arr})'
