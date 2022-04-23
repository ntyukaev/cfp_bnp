from cplex import Cplex
from structures import Matrix
from utils import read


class CFP:
    def __init__(self, path):
        self.example = read(path)
        self.matrix = Matrix(self.example)
        self.cells, self.efficacy = self.matrix.populate_cells(self.matrix.efficacy_fn,
                                                               n_cells=min(self.matrix.rows_count, self.matrix.columns_count),
                                                               )
        self.n1 = self.matrix.n1
        self.n0 = self.matrix.n0
        self.n1_in = self.matrix.n1
        self.n0_in = self.matrix.n0
        self.var_mapping = None
        self.master_problem = self.construct_master_problem()
        self.master_problem.solve()
        self.slave_problem = self.construct_slave_problem()

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

        dual_values = self.master_problem.solution.get_dual_values()
        var_names = self.get_var_names_for_slave_problem()
        # for cell in self.cells:
        #     rows_indices = cell.get_rows_indices()
        problem.variables.add(
            obj=dual_values,
            ub=[1.0] * len(dual_values),
            lb=[0.0] * len(dual_values),
            names=var_names
        )

        for cell in self.cells:
            rows_vars = [f'yr_{r}' for r in cell.get_rows_indices()]
            problem.linear_constraints.add(
                lin_expr=[[rows_vars, [1.0] * len(rows_vars)]],
                senses=['L'],
                rhs=[1.0],
                names=[f'cell_{cell.index}_rows']
            )
            cols_vars = [f'yc_{c}' for c in cell.get_columns_indices()]
            problem.linear_constraints.add(
                lin_expr=[[cols_vars, [1.0] * len(cols_vars)]],
                senses=['L'],
                rhs=[1.0],
                names=[f'cell_{cell.index}_cols']
            )

        return problem

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

    def solve(self):
        # find violation cell
        self.dkb()
        violation_cell = self.get_violation_cell()
        for i in range(100):
            if violation_cell and violation_cell not in self.cells:
                break
            violation_cell = self.get_violation_cell()
        while violation_cell and violation_cell not in self.cells:
            self.cells.append(violation_cell)
            self.master_problem = self.construct_master_problem()
            self.master_problem.solve()
            violation_cell = self.get_violation_cell()

        # update slave problem objective
        self.update_slave_problem()
        self.solve()

    def get_violation_cell(self):
        # эвристика для поиска самой нарушенной ячейки
        dual_solution = self.master_problem.solution.get_dual_values()
        rows_weights = dual_solution[:self.matrix.rows_count]
        columns_weights = dual_solution[self.matrix.rows_count:]
        cells, efficacy = self.matrix.populate_cells(self.matrix.get_violation_metric(self.n1_in, self.n0_in, rows_weights, columns_weights), n_cells=2)
        cells = sorted(cells, key=lambda c: c.priority)
        if cells:
            cell = cells[0]
            if efficacy > 1.0 and cell.priority < 0:
                return cell
        return None

    def update_slave_problem(self):
        dual_values = self.master_problem.solution.get_dual_values()
        variables = self.get_var_names_for_slave_problem()
        objective = list(zip(variables,
                             [v if abs(v) != 0.0 else 0.000001 for v in dual_values]))
        self.slave_problem.objective.set_linear(objective)
        self.slave_problem.solve()
        slave_problem_objective = self.slave_problem.solution.get_objective_value()
        if slave_problem_objective > 0:
            raise NotImplementedError

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


if __name__ == '__main__':
    main()
