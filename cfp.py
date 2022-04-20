from cplex import Cplex
from structures import Matrix
from utils import read


class CFP:
    def __init__(self, path):
        self.example = read(path)
        self.matrix = Matrix(self.example)
        self.matrix.populate_cells()
        self.n1 = self.matrix.n1
        self.n0 = self.matrix.n0
        self.n1_in = self.matrix.n1
        self.n0_in = self.matrix.n0
        self.var_mapping = dict()
        self.problem = self.construct_problem()

    def construct_problem(self):
        problem = Cplex()
        problem.set_log_stream(None)
        problem.set_results_stream(None)
        problem.set_warning_stream(None)
        problem.set_error_stream(None)
        problem.objective.set_sense(problem.objective.sense.maximize)

        # define x variables
        # a = n1 + n_0_in_l
        # b = n_1_l
        for cell in self.matrix.cells:
            n1_k, n0_k = cell.info()
            expression = (self.n1 + self.n0_in) * n1_k - self.n1_in * n0_k
            expression = float(expression)
            var_name = f'x_{cell.index}'
            self.var_mapping[var_name] = cell
            problem.variables.add(obj=[expression],
                                  names=[var_name],
                                  types=[problem.variables.type.continuous])

        # n1_in * n1 constant
        problem.objective.set_offset(-self.n1_in * self.n1)

        # define row constraints
        for i in range(self.matrix.rows_count):
            vars = []
            for cell in self.matrix.cells:
                if i in cell.get_rows_indices():
                    vars.append(f'x_{cell.index}')
            problem.linear_constraints.add(lin_expr=[[vars, [1.0] * len(vars)]],
                                           senses=['L'],
                                           rhs=[1.0],
                                           names=[f'row_{i}'])
        # define column constraints
        for i in range(self.matrix.columns_count):
            vars = []
            for cell in self.matrix.cells:
                if i in cell.get_columns_indices():
                    vars.append(f'x_{cell.index}')
            problem.linear_constraints.add(lin_expr=[[vars, [1.0] * len(vars)]],
                                           senses=['L'],
                                           rhs=[1.0],
                                           names=[f'column_{i}'])

        return problem

    def solve(self):
        self.dkb()

    def dkb(self):
        self.problem.solve()
        objective_value = self.problem.solution.get_objective_value()
        if objective_value > 0:
            n1_inl = 0
            n0_inl = 0
            solution_values = self.problem.solution.get_values()
            solution_names = self.problem.variables.get_names()
            solution_dict = dict(zip(solution_names, solution_values))
            for var_name, var_val in solution_dict.items():
                if var_val:
                    cell = self.var_mapping[var_name]
                    n1, n0 = cell.info()
                    n1_inl += n1
                    n0_inl += n0
            self.n1_in = n1_inl
            self.n0_in = n0_inl
            self.problem = self.construct_problem()
            self.dkb()


def main():
    example = 'examples/1.txt'
    cfp = CFP(example)
    cfp.solve()


if __name__ == '__main__':
    main()
