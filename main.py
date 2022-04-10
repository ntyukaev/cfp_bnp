from cplex import Cplex


class CellFormation:
    def __init__(self, matrix):
        self.matrix = matrix
        self.cells = self.generate_cells()

    def generate_cells(self):
        cells = self._generate_cells()


if __name__ == '__main__':
    from utils import read
    matrix = read('examples/1.txt')
    cf = CellFormation(matrix)
