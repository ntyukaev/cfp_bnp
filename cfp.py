import random
import numpy as np
from cplex import Cplex
from structures import Matrix, Row, Column, Cell
from utils import read


class CFP:
    def __init__(self, path):
        self.example = read(path)
        self.matrix = Matrix(self.example)
        self.cells, self.efficacy = self.matrix.populate_cells(self.matrix.efficacy_fn,
                                                               n_cells=min(self.matrix.rows_count, self.matrix.columns_count),
                                                               )
        self.current_branch = 0
        self.n1 = self.matrix.n1
        self.n0 = self.matrix.n0
        self.n1_in = self.matrix.n1
        self.n0_in = self.matrix.n0
        self.var_mapping = None
        self.master_problem = self.construct_master_problem()
        self.master_problem.solve()

    def get_var_names_for_slave_problem(self):
        var_names = [f'yr_{r}' for r in range(self.matrix.rows_count)]
        var_names.extend([f'yc_{c}' for c in range(self.matrix.columns_count)])
        return var_names

    def construct_slave_problem(self):
        problem = Cplex()
        problem.set_log_stream(None)
        problem.set_results_stream(None)
        problem.set_warning_stream(None)
        problem.set_error_stream(None)

        problem.objective.set_sense(problem.objective.sense.minimize)

        dual_solution = self.master_problem.solution.get_dual_values()
        rows_weights = dual_solution[:self.matrix.rows_count]
        columns_weights = dual_solution[self.matrix.rows_count:]

        total_vars = []
        for index, weight in enumerate(rows_weights):
            var = f'x_{index}'
            total_vars.append(var)
            problem.variables.add(
                obj=[weight],
                names=[var],
                types=['B']
            )

        for index, weight in enumerate(columns_weights):
            var = f'y_{index}'
            total_vars.append(var)
            problem.variables.add(
                obj=[weight],
                names=[var],
                types=['B']
            )

        # define z variables
        for row_index in range(self.matrix.rows_count):
            for col_index in range(self.matrix.columns_count):
                z_variable = f'z_{row_index}_{col_index}'
                matrix_value = self.matrix[row_index][col_index]
                # define z variable
                objective_value = -float((self.n1 + self.n0_in) * matrix_value + self.n1_in * (1 - matrix_value))
                problem.variables.add(
                    obj=[objective_value],
                    names=[z_variable],
                    types=['B']
                )

                # linearize z = x * y
                # z <= x
                # z <= y
                # x + y - z <= 1
                problem.linear_constraints.add(
                    lin_expr=[[[z_variable, f'x_{row_index}'], [1.0, -1.0]]],
                    senses=['L'],
                    rhs=[0.0]
                )

                problem.linear_constraints.add(
                    lin_expr=[[[z_variable, f'y_{col_index}'], [1.0, -1.0]]],
                    senses=['L'],
                    rhs=[0.0]
                )

                problem.linear_constraints.add(
                    lin_expr=[[[f'x_{row_index}', f'y_{col_index}', z_variable], [1.0, 1.0, -1.0]]],
                    senses=['L'],
                    rhs=[1.0]
                )

        problem.solve()
        rows_cols_sln = problem.solution.get_values()[:len(dual_solution)]
        new_cell = self.matrix.create_cell(rows_cols_sln)
        if new_cell not in self.cells:
            return new_cell

    def construct_master_problem(self):
        self.var_mapping = dict()
        problem = Cplex()
        problem.set_log_stream(None)
        problem.set_results_stream(None)
        problem.set_warning_stream(None)
        problem.set_error_stream(None)
        problem.objective.set_sense(problem.objective.sense.maximize)

        # define x variables
        # a = n1 + n_0_in_l
        # b = n_1_l
        for cell in self.cells:
            n1_k, n0_k = cell.info()
            expression = (self.n1 + self.n0_in) * n1_k - self.n1_in * n0_k
            expression = float(expression)
            var_name = f'x_{cell.index}'
            self.var_mapping[var_name] = cell
            problem.variables.add(obj=[expression],
                                  names=[var_name]
                                  # types=[problem.variables.type.continuous]
                                  )

        # n1_in * n1 constant
        problem.objective.set_offset(-self.n1_in * self.n1)

        # define row constraints
        for i in range(self.matrix.rows_count):
            vars = []
            for cell in self.cells:
                if i in cell.get_rows_indices():
                    vars.append(f'x_{cell.index}')
            problem.linear_constraints.add(lin_expr=[[vars, [1.0] * len(vars)]],
                                           senses=['L'],
                                           rhs=[1.0],
                                           names=[f'row_{i}'])
        # define column constraints
        for i in range(self.matrix.columns_count):
            vars = []
            for cell in self.cells:
                if i in cell.get_columns_indices():
                    vars.append(f'x_{cell.index}')
            problem.linear_constraints.add(lin_expr=[[vars, [1.0] * len(vars)]],
                                           senses=['L'],
                                           rhs=[1.0],
                                           names=[f'column_{i}'])

        return problem

    def calculate_solution_efficacy(self):
        solution_values = self.master_problem.solution.get_values()
        solution_vars = self.master_problem.variables.get_names()
        solution_mapping = list(filter(lambda x: x[1] == 1.0, zip(solution_vars, solution_values)))
        cells = [self.var_mapping[sm[0]] for sm in solution_mapping]
        return self.matrix.efficacy_fn(cells)

    def solve(self):
        # find violation cell
        self.dkb()
        violation_cell = None

        for i in range(15):
            violation_cell = self.get_violation_cell()
            if violation_cell:
                break
        while violation_cell and violation_cell not in self.cells:
            self.cells.append(violation_cell)
            self.master_problem = self.construct_master_problem()
            self.master_problem.solve()
            # self.dkb()
            for i in range(15):
                violation_cell = self.get_violation_cell()
                if violation_cell:
                    break

        efficacy = self.calculate_solution_efficacy()
        if efficacy > self.efficacy:
            self.efficacy = efficacy
            master_solution = self.master_problem.solution.get_values()
            bvar = max(list(filter(lambda x: not x[1].is_integer(), enumerate(master_solution))),
                       key=lambda x: x[1], default=(None, None))[0]

            if bvar is None:
                # solve slave problem
                test = self.construct_slave_problem()
                return efficacy

            self.current_branch += 1
            branch_name = 'b_{}'.format(self.current_branch)
            self.master_problem.linear_constraints.add(lin_expr=[[[bvar], [1.0]]],
                                                       senses=['E'],
                                                       rhs=[0.0],
                                                       names=[branch_name])
            branch_1 = self.solve()

            self.master_problem.linear_constraints.delete(branch_name)

            self.master_problem.linear_constraints.add(lin_expr=[[[bvar], [1.0]]],
                                                       senses=['E'],
                                                       rhs=[1.0],
                                                       names=[branch_name])

            branch_2 = self.solve()

            return max(branch_1, branch_2)

        cell = self.construct_slave_problem()
        return float('-Inf')

    def get_violation_cell(self):
        # эвристика для поиска самой нарушенной ячейки
        dual_solution = self.master_problem.solution.get_dual_values()
        rows_weights = dual_solution[:self.matrix.rows_count]
        columns_weights = dual_solution[self.matrix.rows_count:]
        cell, efficacy = self.matrix.populate_cells_for_violation(self.cells,
            self.matrix.get_violation_metric(self.n1_in, self.n0_in, rows_weights, columns_weights), n_cells=2)

        if efficacy > 1.0 and cell.priority < 0:
            if cell in self.cells:
                reorganized_cell = self.reorganize_cell(cell, rows_weights, columns_weights)
                return reorganized_cell
            return cell
        return None

    def reorganize_cell(self, cell, rows_weights, cols_weights):
        calculate_violation = self.matrix.get_violation_metric(self.n1_in, self.n0_in, rows_weights, cols_weights)
        pool = list()
        pool.extend(cell.rows)
        pool.extend(cell.columns)
        # try to remove rows and columns and calculate violation
        random.shuffle(pool)
        while pool:
            # take random rows/columns
            element = pool.pop()
            cell.remove(element)
            violation = calculate_violation([cell])
            if violation > 0 and cell not in self.cells:
                break
            else:
                cell.add(element)
        return cell

    def get_dual_solution_mapping(self):
        dual_solution = self.master_problem.solution.get_dual_values()
        solution_vars = [f'yr_{r}' for r in range(self.matrix.rows_count)]
        solution_vars.extend([f'yc_{c}' for c in range(self.matrix.columns_count)])
        return dict(zip(solution_vars, dual_solution))

    def dkb(self):
        self.master_problem.solve()
        objective_value = self.master_problem.solution.get_objective_value()
        if objective_value > 0:
            n1_inl = 0
            n0_inl = 0
            solution_values = self.master_problem.solution.get_values()
            solution_names = self.master_problem.variables.get_names()
            solution_dict = dict(zip(solution_names, solution_values))
            for var_name, var_val in solution_dict.items():
                if var_val:
                    cell = self.var_mapping[var_name]
                    n1, n0 = cell.info()
                    n1_inl += n1
                    n0_inl += n0
            self.n1_in = n1_inl
            self.n0_in = n0_inl
            self.master_problem = self.construct_master_problem()
            self.dkb()


def main():
    example = 'examples/1.txt'
    cfp = CFP(example)
    cfp.solve()
    print(cfp.efficacy)


if __name__ == '__main__':
    main()
