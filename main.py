import sys
import numpy as np
from numpy import dot
from numpy.linalg import norm
from cplex import Cplex


class CellFormation:
    def __init__(self, matrix):
        self.matrix = matrix
        self.rows_count = len(self.matrix)
        self.cols_count = len(self.matrix[0])
        self.flat = matrix.flatten().astype(float)
        self.n0 = len(list(filter(lambda x: x == 0, self.flat)))
        self.n1 = len(self.flat) - self.n0
        self.n0_in = self.n0
        self.n1_in = self.n1
        self.cells = self.generate_cells()
        self.problem = self.construct_problem()

    def construct_problem(self):
        problem = Cplex()
        problem.set_log_stream(None)
        problem.set_results_stream(None)
        problem.set_warning_stream(None)
        problem.set_error_stream(None)
        problem.objective.set_sense(problem.objective.sense.maximize)

        # set cell variables
        cell_count = len(self.cells)

        # a = n1 + n_0_in_l
        # b = n_1_l
        for index, cell in enumerate(self.cells):
            n1_k = cell[2][0]
            n0_k = cell[2][1]
            obj_expression = (self.n1 + self.n0) * n1_k - self.n1 * n0_k
            obj_expression = float(obj_expression)
            # n0_in
            problem.variables.add(obj=[obj_expression],
                                  names=[f'x_{index}'],
                                  types=[problem.variables.type.continuous])

        # set row constraints
        row_variables_names = list()
        row_variables_ub = list()
        row_variables_lb = list()
        row_variables_types = list()
        for k in range(cell_count):
            for i in range(self.rows_count):
                row_variables_names.append(f'r_{i}_{k}')
                row_variables_ub.append(1.0)
                row_variables_lb.append(0.0)
                row_variables_types.append(problem.variables.type.continuous)
        problem.variables.add(
            ub=row_variables_ub,
            lb=row_variables_lb,
            types=row_variables_types,
            names=row_variables_names
        )

        # set col constraints
        col_variables_names = list()
        col_variables_ub = list()
        col_variables_lb = list()
        col_variables_types = list()
        for k in range(cell_count):
            for j in range(self.cols_count):
                col_variables_names.append(f'c_{j}_{k}')
                col_variables_ub.append(1.0)
                col_variables_lb.append(0.0)
                col_variables_types.append(problem.variables.type.continuous)
        problem.variables.add(
            ub=col_variables_ub,
            lb=col_variables_lb,
            types=col_variables_types,
            names=col_variables_names
        )

        # make helper variable for r_i_k * x_k <= 1
        # z_i_k = r_i_k * x_k
        # z_i_k <= x_k
        # z_i_k <= r_i_k
        # z_i_k >= x_k + r_i_k - 1
        for k in range(cell_count):
            x_var = f'x_{k}'
            for i in range(self.rows_count):
                z_var = f'z_{i}_{k}'
                r_var = f'r_{i}_{k}'
                problem.variables.add(
                    ub=[1.0],
                    lb=[0.0],
                    types=[problem.variables.type.continuous],
                    names=[z_var]
                )
                # z_i_k <= x_k
                problem.linear_constraints.add(lin_expr=[[[z_var, x_var], [1.0, -1.0]]],
                                               senses=['L'],
                                               rhs=[0.0],
                                               names=[f'{z_var} less than {x_var}'])
                # z_i_k <= r_i_k
                problem.linear_constraints.add(lin_expr=[[[z_var, r_var], [1.0, -1.0]]],
                                               senses=['L'],
                                               rhs=[0.0],
                                               names=[f'{z_var} less than {r_var}'])

                # x_k + r_i_k - z_i_k <= 1
                problem.linear_constraints.add(lin_expr=[[[x_var, r_var, z_var], [1.0, 1.0, -1.0]]],
                                               senses=['L'],
                                               rhs=[1.0],
                                               names=[f'{x_var} plus {r_var} minus {z_var} less than 1'])
        # sum of z_i by k less than 1
        for i in range(self.rows_count):
            z_vars = list()
            for k in range(cell_count):
                z_vars.append(f'z_{i}_{k}')
            problem.linear_constraints.add(lin_expr=[[z_vars, [1.0] * len(z_vars)]],
                                           senses=['L'],
                                           rhs=[1.0],
                                           names=[f'sum of z_{i} less than one'])

        # make helper variable for c_j_k * x_k <= 1
        # t_j_k = c_j_k * x_k
        # t_j_k <= x_k
        # t_j_k <= c_j_k
        # t_j_k >= x_k + c_j_k - 1
        for k in range(cell_count):
            x_var = f'x_{k}'
            for j in range(self.cols_count):
                t_var = f't_{j}_{k}'
                c_var = f'c_{j}_{k}'
                problem.variables.add(
                    ub=[1.0],
                    lb=[0.0],
                    types=[problem.variables.type.continuous],
                    names=[t_var]
                )
                # t_j_k <= x_k
                problem.linear_constraints.add(lin_expr=[[[t_var, x_var], [1.0, -1.0]]],
                                               senses=['L'],
                                               rhs=[0.0],
                                               names=[f'{t_var} less than {x_var}'])
                # t_j_k <= c_j_k
                problem.linear_constraints.add(lin_expr=[[[t_var, c_var], [1.0, -1.0]]],
                                               senses=['L'],
                                               rhs=[0.0],
                                               names=[f'{t_var} less than {c_var}'])

                # x_k + c_j_k - t_i_k <= 1
                problem.linear_constraints.add(lin_expr=[[[x_var, c_var, t_var], [1.0, 1.0, -1.0]]],
                                               senses=['L'],
                                               rhs=[1.0],
                                               names=[f'{x_var} plus {c_var} minus {t_var} less than 1'])

        # sum of t_i by k less than 1
        for j in range(self.cols_count):
            t_vars = list()
            for k in range(cell_count):
                t_vars.append(f't_{j}_{k}')
            problem.linear_constraints.add(lin_expr=[[t_vars, [1.0] * len(t_vars)]],
                                           senses=['L'],
                                           rhs=[1.0],
                                           names=[f'sum of t_{j} less than one'])

        return problem

    def generate_cells(self):
        cells = list()
        cells.extend(self._generate_cells(self.matrix))
        cells.extend(self.swap(self._generate_cells(np.transpose(self.matrix))))
        return cells

    def swap(self, cells):
        for cell in cells:
            cell[0], cell[1] = cell[1], cell[0]
        return cells

    def _generate_cells(self, matrix):
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
            ones_zeros = [0, 0]
            cell = [included_rows, included_columns, ones_zeros]
            # pick the columns which have 1
            cols_with_ones = []
            for index, col in enumerate(matrix[best_row]):
                if col:
                    cols_with_ones.append(index)
            # count the number of selected ones
            ones_in_row = len(cols_with_ones)
            ones_zeros[0] += ones_in_row
            ones_threshold = int(ones_in_row // 2)
            included_columns.extend(cols_with_ones)
            # now try adding more rows to the cell
            for row in rows_count.copy():
                # sum_of_ones_in_this_row = sum([matrix[row][c] for c in cols_with_ones])
                a = [matrix[best_row][c] for c in cols_with_ones]
                b = [matrix[row][c] for c in cols_with_ones]
                cos_sim = dot(a, b) / (norm(a) * norm(b))
                if cos_sim >= 0.85:
                    ones_zeros[0] += sum(b)
                    ones_zeros[1] += len(b) - sum(b)
                    included_rows.append(row)
                    r_sum[row] = -sys.maxsize
                    rows_count.remove(row)
            cells.append(cell)
        # cells.append([list(range(r)), list(range(c)), [self.n1, self.n0]])
        return cells


if __name__ == '__main__':
    from utils import read
    matrix = read('examples/1.txt')
    cf = CellFormation(matrix)
    cf.problem.solution.get_values()
