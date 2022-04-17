from utils import read
from structures import CellList


def get_cells(matrix):
    cell_list = CellList(matrix)
    cell_list.generate_cells()
    print(cell_list)


if __name__ == '__main__':
    matrix = read('examples/1.txt')
    get_cells(matrix)
