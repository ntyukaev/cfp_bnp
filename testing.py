import sys

import numpy as np
from utils import read
from numpy import dot
from numpy.linalg import norm


def swap(cells):
    return [[col, row] for row, col in cells]


def get_cells(matrix):
    cells = []
    r, c = matrix.shape
    r_sum = [0] * r
    c_sum = [0] * c
    rows_count = set(range(r))

    for index, row in enumerate(matrix):
        r_sum[index] = sum(row)

    for index, col in enumerate(np.transpose(matrix)):
        c_sum[index] = sum(col)

    while rows_count:
        # pick the row which has the largest sum
        best_row = np.argmax(r_sum)
        r_sum[best_row] = -sys.maxsize
        # drop the row
        rows_count.remove(best_row)
        included_rows = list()
        included_rows.append(best_row)
        included_columns = []
        cell = [included_rows, included_columns]
        # pick the columns which have 1
        cols_with_ones = []
        for index, col in enumerate(matrix[best_row]):
            if col:
                cols_with_ones.append(index)
        # count the number of selected ones
        ones_in_row = len(cols_with_ones)
        ones_threshold = int(ones_in_row // 2)
        included_columns.extend(cols_with_ones)
        # now try adding more rows to the cell
        for row in rows_count.copy():
            # sum_of_ones_in_this_row = sum([matrix[row][c] for c in cols_with_ones])
            a = [matrix[best_row][c] for c in cols_with_ones]
            b = [matrix[row][c] for c in cols_with_ones]
            cos_sim = dot(a, b)/(norm(a)*norm(b))
            if cos_sim >= 0.85:
                included_rows.append(row)
                r_sum[row] = -sys.maxsize
                rows_count.remove(row)
        cells.append(cell)
    return cells


def main():
    example_matrix = np.array(read('examples/1.txt'))
    cells = get_cells(example_matrix)
    print(cells)
    cells.extend(swap(get_cells(np.transpose(example_matrix))))
    print(cells)


if __name__ == '__main__':
    main()
