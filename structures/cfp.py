import random
from cplex import Cplex
from .machine import Machine
from .part import Part
from .cell import Cell
from cplex.exceptions.errors import CplexSolverError


class CFP:
    def __init__(self, matrix):
        self.matrix = matrix
        self.transposed_matrix = self.matrix.transposed
        self.n1 = self.matrix.n1
        self.n0 = self.matrix.n0
        self.n0_in = self.n0
        self.n1_in = self.n1
        self.current_branch = 0
        self.machines_count = self.matrix.rows_count
        self.parts_count = self.matrix.columns_count
        self.grouping_efficacy = 0
        self.forbidden_sets = set()
        self.current_objective_value = float('-Inf')
        self.best_objective_value = float('-Inf')
        self.branches = dict()
        self.branching_history = list()
        self.cells = self.create_initial_cells()
        self.cell_mapping = dict()
        self.master_problem = None
        self.slave_problem = None

    def get_machine(self, index):
        return Machine(index, self.matrix[index])

    def get_part(self, index):
        return Part(index, self.transposed_matrix[index])

    def __repr__(self):
        return self.matrix.__repr__()

    def get_pool(self, machines, parts):
        pool = set()
        pool.update(machines)
        pool.update(parts)
        return pool

    def calculate_grouping_efficacy(self, cells):
        n1_in = 0
        n0_in = 0
        for cell in cells:
            n1_in_cell, n0_in_cell = cell.info()
            n1_in += n1_in_cell
            n0_in += n0_in_cell
        return n1_in / (self.n1 + n0_in)

    def create_initial_cells(self):
        # generate the maximum number of possible cells
        cells = [Cell() for _ in range(min(self.matrix.rows_count, self.matrix.columns_count))]
        machines = set([self.get_machine(index) for index in range(self.matrix.rows_count)])
        parts = set([self.get_part(index) for index in range(self.matrix.columns_count)])

        # distribute machines and parts randomly
        for machine in machines:
            cell = random.choice(cells)
            cell.add(machine)

        for part in parts:
            cell = random.choice(cells)
            cell.add(part)

        # calculate an initial value of grouping efficacy
        grouping_efficacy = self.calculate_grouping_efficacy(cells)

        # put all machines and parts into one set
        # to redistribute them in order to get a better efficacy
        pool = self.get_pool(machines, parts)
        while pool:
            # randomly select an element
            element = random.sample(pool, 1)[-1]
            parent = element.parent
            if not parent:
                pool.remove(element)
                continue
            random.shuffle(cells)
            for cell in cells:
                cell.add(element)
                new_grouping_efficacy = self.calculate_grouping_efficacy(cells)
                # if the efficacy has improved then recreate a set of all elements and repeat
                if new_grouping_efficacy > grouping_efficacy:
                    grouping_efficacy = new_grouping_efficacy
                    pool = self.get_pool(machines, parts)
                    break
            # if the efficacy has not been increased, then remove the selected element from the pool
            else:
                # return the element to its parent
                parent.add(element)
                pool.remove(element)

        # remove the cells which don't have machines or parts
        cells = [cell for cell in cells if cell.machines and cell.parts]
        # we set the grouping efficacy
        self.grouping_efficacy = grouping_efficacy
        return cells

    def construct_master_problem(self):
        problem = Cplex()
        problem.set_log_stream(None)
        problem.set_results_stream(None)
        problem.set_warning_stream(None)
        problem.set_error_stream(None)
        problem.objective.set_sense(problem.objective.sense.maximize)

        # reinitialize cell mapping
        self.cell_mapping = dict()

        # define x variables
        # a = n1 + n_0_in_l
        # b = n_1_in_l
        for cell in self.cells:
            n1_k, n0_k = cell.info()
            expression = (self.n1 + self.n0_in) * n1_k - self.n1_in * n0_k
            expression = float(expression)
            self.cell_mapping[cell.index] = cell
            problem.variables.add(obj=[expression], names=[cell.index])

        # n1_in * n1 constant
        problem.objective.set_offset(-self.n1_in * self.n1)

        # define machine constraints
        for i in range(self.machines_count):
            vars = []
            for cell in self.cells:
                if i in cell.get_machine_indices():
                    vars.append(cell.index)
            problem.linear_constraints.add(lin_expr=[[vars, [1.0] * len(vars)]],
                                           senses=['L'],
                                           rhs=[1.0],
                                           names=[f'machine_{i}'])
        # define part constraints
        for i in range(self.parts_count):
            vars = []
            for cell in self.cells:
                if i in cell.get_part_indices():
                    vars.append(cell.index)
            problem.linear_constraints.add(lin_expr=[[vars, [1.0] * len(vars)]],
                                           senses=['L'],
                                           rhs=[1.0],
                                           names=[f'part_{i}'])

        # define branching constraints
        for branch_name, rhs in self.branches.items():
            problem.linear_constraints.add(lin_expr=[[[branch_name], [1.0]]],
                                           senses=['E'],
                                           rhs=[rhs],
                                           names=[branch_name])

        problem.solve()
        self.master_problem = problem

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

        for index, weight in enumerate(rows_weights):
            var = f'x_{index}'
            problem.variables.add(
                obj=[weight],
                names=[var],
                types=['B']
            )

        for index, weight in enumerate(columns_weights):
            var = f'y_{index}'
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
                obj = -int((self.n1 + self.n0_in) * matrix_value - self.n1_in * (1 - matrix_value))
                problem.variables.add(
                    obj=[obj],
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

        # forbidden sets
        for s in self.forbidden_sets:
            fs_cell = self.cell_mapping[s]
            machines_count = fs_cell.machines_count()
            parts_count = fs_cell.parts_count()
            S = machines_count + parts_count
            M = self.matrix.rows_count + self.matrix.columns_count + 1
            l_k = f'l_{fs_cell.index}'
            problem.variables.add(
                names=[l_k],
                types=['B']
            )

            sum_i = [f'x_{i}' for i in fs_cell.get_machine_indices()]
            sum_j = [f'y_{j}' for j in fs_cell.get_part_indices()]

            # - sum_i - sum_j -M * l_k <= -S
            problem.linear_constraints.add(
                lin_expr=[[[*sum_i, *sum_j, l_k], [*[-1.0] * len(sum_i), *[-1.0] * len(sum_j), -M]]],
                senses=['L'],
                rhs=[-S],
                names=[f'{l_k}_1']
            )

            # sum_i + sum_j + M* l_k <= M + S - 1
            problem.linear_constraints.add(
                lin_expr=[[[*sum_i, *sum_j, l_k], [*[1.0] * len(sum_i), *[1.0] * len(sum_j), M]]],
                senses=['L'],
                rhs=[M + S - 1],
                names=[f'{l_k}_2']
            )

            # g_k
            g_k = f'g_{fs_cell.index}'
            problem.variables.add(
                names=[g_k],
                types=['B']
            )

            # sum_i - sum_j - M * g_k <= S
            problem.linear_constraints.add(
                lin_expr=[[[*sum_i, *sum_j, g_k], [*[1.0] * len(sum_i), *[-1.0] * len(sum_j), -M]]],
                senses=['L'],
                rhs=[S],
                names=[f'{g_k}_3']
            )

            # - sum_i - sum_j + M * g_k <= M - S - 1
            problem.linear_constraints.add(
                lin_expr=[[[*sum_i, *sum_j, g_k], [*[-1.0] * len(sum_i), *[-1.0] * len(sum_j), M]]],
                senses=['L'],
                rhs=[M - S - 1],
                names=[f'{g_k}_4']
            )

            # G_k
            G_k = f'G_{fs_cell.index}'
            problem.variables.add(
                names=[G_k],
                types=['B']
            )

            sum_i_all = [f'x_{i}' for i in range(self.matrix.rows_count)]
            sum_j_all = [f'y_{j}' for j in range(self.matrix.columns_count)]

            # sum_i_all - sum_j_all - M * G_k <= S
            problem.linear_constraints.add(
                lin_expr=[[[*sum_i_all, *sum_j_all, G_k], [*[1.0] * len(sum_i_all), *[-1.0] * len(sum_j_all), -M]]],
                senses=['L'],
                rhs=[S],
                names=[f'{G_k}_5']
            )

            # - sum_i_all - sum_j_all + M * G_k <= M - S - 1
            problem.linear_constraints.add(
                lin_expr=[[[*sum_i_all, *sum_j_all, G_k], [*[-1.0] * len(sum_i_all), *[-1.0] * len(sum_j_all), M]]],
                senses=['L'],
                rhs=[M - S - 1],
                names=[f'{G_k}_6']
            )

        problem.solve()
        self.slave_problem = problem

    def get_cell_from_slave_problem(self):
        self.construct_slave_problem()
        slave_solution = self.slave_problem.solution.get_values()
        machines = slave_solution[:self.machines_count]
        parts = slave_solution[self.machines_count:self.machines_count + self.parts_count]
        cell = Cell()
        for idx, value in enumerate(machines):
            if value:
                cell.add(self.get_machine(idx))

        for idx, value in enumerate(parts):
            if value:
                cell.add(self.get_part(idx))
        if cell not in self.cells and self.slave_problem.solution.get_objective_value() <= 0:
            return cell
        else:
            return

    def calculate_solution_efficacy(self):
        solution_values = self.master_problem.solution.get_values()
        solution_vars = self.master_problem.variables.get_names()
        solution_mapping = dict(filter(lambda x: x[1] == 1.0, zip(solution_vars, solution_values)))
        cells = [cell for cell in self.cells if cell.index in solution_mapping.keys()]
        return self.calculate_grouping_efficacy(cells)

    def dkb(self):
        self.best_objective_value = float('-Inf')
        self.construct_master_problem()
        objective_value = self._iterate()
        self.master_problem.solve()
        if objective_value > 0:
            n1_inl = 0
            n0_inl = 0
            solution_values = self.master_problem.solution.get_values()
            solution_names = self.master_problem.variables.get_names()
            solution_dict = dict(zip(solution_names, solution_values))
            for var_name, var_val in solution_dict.items():
                if var_val:
                    cell = self.cell_mapping[var_name]
                    n1, n0 = cell.info()
                    n1_inl += n1
                    n0_inl += n0
            self.n1_in = n1_inl
            self.n0_in = n0_inl
            self.dkb()

    def solve(self):
        self.dkb()
        self.grouping_efficacy = self.calculate_solution_efficacy()

    def get_branching_variable(self):
        master_solution = self.master_problem.solution.get_values()
        var_names = self.master_problem.variables.get_names()
        bvar_ind = max(list(filter(lambda x: not x[1].is_integer(), enumerate(master_solution))),
                   key=lambda x: x[1], default=(None, None))[0]
        if bvar_ind is not None:
            return var_names[bvar_ind]

    def delete_branch(self, branch_name):
        self.master_problem.linear_constraints.delete(branch_name)
        del self.branches[branch_name]

    def _iterate(self):
        self.master_problem.solve()
        self.find_violations()

        objective_value = self.master_problem.solution.get_objective_value()

        bvar = self.get_branching_variable()
        if bvar is None:
            # solve slave problem
            slave_cell = self.get_cell_from_slave_problem()
            if slave_cell:
                self.cells.append(slave_cell)
                self.construct_master_problem()
                return self._iterate()
            if objective_value > self.best_objective_value:
                self.best_objective_value = objective_value
            return objective_value

        self.current_branch += 1
        branch_name = bvar
        rhs = 0.0
        self.forbidden_sets.add(branch_name)
        self.branches[branch_name] = rhs
        self.branching_history.append((branch_name, rhs))
        self.master_problem.linear_constraints.add(lin_expr=[[[bvar], [1.0]]],
                                                   senses=['E'],
                                                   rhs=[rhs],
                                                   names=[branch_name])
        branch_1 = self._iterate()
        self.delete_branch(branch_name)
        self.forbidden_sets.remove(branch_name)

        # ** #
        rhs = 1.0
        self.branches[branch_name] = rhs
        self.branching_history.append((branch_name, rhs))
        self.master_problem.linear_constraints.add(lin_expr=[[[bvar], [1.0]]],
                                                   senses=['E'],
                                                   rhs=[rhs],
                                                   names=[branch_name])

        branch_2 = self._iterate()
        self.delete_branch(branch_name)

        return max(branch_1, branch_2)

    def calculate_solution_efficacy(self):
        solution_values = self.master_problem.solution.get_values()
        solution_vars = self.master_problem.variables.get_names()
        solution_mapping = list(filter(lambda x: x[1] == 1.0, zip(solution_vars, solution_values)))
        cells = [self.cell_mapping[sm[0]] for sm in solution_mapping]
        return self.calculate_grouping_efficacy(cells)

    def find_violations(self):
        violation_cell = self.get_violation_cell()
        while violation_cell:
            if violation_cell and violation_cell not in self.cells:
                self.cells.append(violation_cell)
            violation_cell = self.get_violation_cell()

    def calculate_violation_metric(self, cells, machine_weights, parts_weights):
        cell_weights = list()
        for cell in cells:
            cell_n1, cell_n0 = cell.info()
            cell_weight = 0
            for row in cell.machines:
                row_weight = machine_weights[row.index]
                cell_weight += row_weight
            for col in cell.parts:
                col_weight = parts_weights[col.index]
                cell_weight += col_weight
            # a = n1 + n_0_in_l
            # b = n_1_in_l
            d_k = ((self.n1 + self.n0_in) * cell_n1 - self.n1_in * cell_n0)
            cell_weight_result = cell_weight - d_k
            cell.priority = cell_weight_result
            cell_weights.append(cell_weight_result)
        min_weight = min(cell_weights)
        return min_weight if min_weight < 0 else 0

    def get_violation_cell(self):
        try:
            dual_solution = self.master_problem.solution.get_dual_values()
        except CplexSolverError:
            return
        machines_weights = dual_solution[:self.matrix.rows_count]
        parts_weights = dual_solution[self.matrix.rows_count:]

        # generate the maximum number of possible cells
        cells = [Cell() for _ in range(2)]
        machines = set([self.get_machine(index) for index in range(self.matrix.rows_count)])
        parts = set([self.get_part(index) for index in range(self.matrix.columns_count)])

        # distribute machines and parts randomly
        for machine in machines:
            cell = random.choice(cells)
            cell.add(machine)

        for part in parts:
            cell = random.choice(cells)
            cell.add(part)

        violation_metric = self.calculate_violation_metric(cells, machines_weights, parts_weights)

        pool = self.get_pool(machines, parts)
        while pool:
            # randomly select an element
            element = random.sample(pool, 1)[-1]
            parent = element.parent
            if not parent:
                pool.remove(element)
                continue
            random.shuffle(cells)
            for cell in cells:
                cell.add(element)
                new_violation_metric = self.calculate_violation_metric(cells, machines_weights, parts_weights)
                # if the efficacy has improved then recreate a set of all elements and repeat
                if new_violation_metric < violation_metric:
                    violation_metric = new_violation_metric
                    broken_cell = sorted(cells, key=lambda x: x.priority)[0]
                    return broken_cell
            # if the efficacy has not been increased, then remove the selected element from the pool
            else:
                # return the element to its parent
                parent.add(element)
                pool.remove(element)

        # remove the cells which don't have machines or parts
        # remove non-negative cells
        cells = [cell for cell in cells if cell.machines and cell.parts and cell.priority < 0]

        cells = sorted(cells, key=lambda x: x.priority)
        if cells:
            return cells[0]
