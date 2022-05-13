import numpy as np
from .utils import read


class Matrix:
    def __init__(self, path):
        self.matrix = read(path)
        self.transposed = np.transpose(self.matrix)
        self.rows_count = len(self.matrix)
        self.columns_count = len(self.matrix[0])
        self.flat = self.matrix.flatten().astype(float)
        self.n0 = len(list(filter(lambda x: x == 0, self.flat)))
        self.n1 = len(self.flat) - self.n0
        self.efficacy = 0
        self.cells = set()

    def __getitem__(self, item):
        return self.matrix[item]
