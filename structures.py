import random
from itertools import count
import numpy as np
from copy import copy


def get_efficacy(n1):
    def _inner(cells):
        n1_in = 0
        n0_in = 0
        for cell in cells:
            n1_in_cell, n0_in_cell = cell.info()
            n1_in += n1_in_cell
            n0_in += n0_in_cell
        return n1_in / (n1 + n0_in)
    return _inner


def get_violation_metric(n1_in, n0_in, rows_weights, cols_weights):
    def _inner(cells):
        cell_weights = list()
        for cell in cells:
            cell_n1, cell_n0 = cell.info()
            cell_weight = 0
            for row in copy(cell.rows):
                row_weight = rows_weights[row.index]
                cell_weight += row_weight
            for col in copy(cell.columns):
                col_weight = cols_weights[col.index]
                cell_weight += col_weight
            d_k = (n1_in * cell_n1 + n0_in * cell_n0)
            cell_weight_result = cell_weight - d_k
            cell.priority = cell_weight_result
            cell_weights.append(cell_weight_result)
        min_weight = min(cell_weights)
        if min_weight < 0:
            return abs(min_weight)
        return 0
    return _inner


class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix
        self.transposed_matrix = np.transpose(self.matrix)
        self.rows_count = len(matrix)
        self.columns_count = len(matrix[0])
        self.flat = self.matrix.flatten().astype(float)
        self.n0 = len(list(filter(lambda x: x == 0, self.flat)))
        self.n1 = len(self.flat) - self.n0
        self.efficacy_fn = get_efficacy(self.n1)
        self.efficacy = 0
        self.get_violation_metric = get_violation_metric
        self.cells = set()

    def __getitem__(self, item):
        return self.matrix[item]

    def create_row(self, index):
        return Row(index, self.matrix[index])

    def create_column(self, index):
        return Column(index, self.transposed_matrix[index])

    def get_pool(self, rows, columns):
        pool = set()
        pool.update([row for row in rows if row.parent])
        pool.update([col for col in columns if col.parent])
        return pool

    def populate_cells(self, efficacy_fn, n_cells=2):
        # generate rows and columns
        rows = set([Row(index, row) for index, row in enumerate(self.matrix)])
        columns = set([Column(index, col) for index, col in enumerate(np.transpose(self.matrix))])
        # the max amount of cells is equal to the min of rows & cols
        cells = [Cell() for _ in range(n_cells)]
        # distribute rows and columns between cells
        for row in rows:
            cell = random.choice(cells)
            cell.add(row)

        for col in columns:
            cell = random.choice(cells)
            cell.add(col)
        # calculate efficacy
        efficacy = efficacy_fn(cells)

        # try improving efficacy by moving rows and columns between cells
        pool = self.get_pool(rows, columns)
        while pool:
            element = random.sample(pool, 1)[-1]
            # remove element from parent
            parent = element.parent
            if not parent:
                pool.remove(element)
                continue
            random.shuffle(cells)
            for cell in cells:
                cell.add(element)
                # calculate efficacy
                current_efficacy = efficacy_fn(cells)
                # if it's better than previous than redo
                if current_efficacy > efficacy:
                    efficacy = current_efficacy
                    pool = self.get_pool(rows, columns)
                    break
            else:
                parent.add(element)
                pool.remove(element)

        cells = [cell for cell in cells if cell.rows or cell.columns]
        return cells, efficacy

    def populate_cells_for_violation(self, total_cells, efficacy_fn, n_cells=2):
        # generate rows and columns
        rows = set([Row(index, row) for index, row in enumerate(self.matrix)])
        columns = set([Column(index, col) for index, col in enumerate(np.transpose(self.matrix))])
        # the max amount of cells is equal to the min of rows & cols
        cells = [Cell() for _ in range(2)]
        # distribute rows and columns between cells
        for row in rows:
            cell = random.choice(cells)
            cell.add(row)

        for col in columns:
            cell = random.choice(cells)
            cell.add(col)
        # calculate efficacy
        efficacy = 0

        # try improving efficacy by moving rows and columns between cells
        pool = self.get_pool(rows, columns)
        while pool:
            element = random.sample(pool, 1)[-1]
            # remove element from parent
            parent = element.parent
            if not parent:
                pool.remove(element)
                continue
            random.shuffle(cells)
            for cell in cells:
                cell.add(element)
                # calculate efficacy
                current_efficacy = efficacy_fn(cells)
                # if it's better than previous than redo
                if current_efficacy > efficacy:
                    efficacy = current_efficacy
                    if cell not in total_cells:
                        return cell, efficacy
                    pool = self.get_pool(rows, columns)
                    break
            else:
                parent.add(element)
                pool.remove(element)
        cells = sorted(cells, key=lambda x: x.priority, reverse=True)
        return cells[0], efficacy


class Cell:
    _ids = count(0)

    def __init__(self):
        self.rows = set()
        self.columns = set()
        self.index = next(self._ids)
        self.priority = 0

    def __repr__(self):
        return f'Cell({self.get_rows_indices()}, {self.get_columns_indices()})'

    def __eq__(self, other):
        return sorted(self.get_rows_indices()) == sorted(other.get_rows_indices()) and \
               sorted(self.get_columns_indices()) == sorted(other.get_columns_indices())

    def get_rows_indices(self):
        return sorted([row.index for row in self.rows])

    def get_columns_indices(self):
        return sorted([col.index for col in self.columns])

    def add(self, obj):
        if obj.parent:
            obj.parent.remove(obj)
        obj.parent = self
        if type(obj) == Row:
            self.rows.add(obj)
        else:
            self.columns.add(obj)

    def remove(self, obj):
        obj.parent = None
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
