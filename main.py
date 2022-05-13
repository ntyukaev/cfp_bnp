import time
from structures.matrix import Matrix
from structures.cfp import CFP


def main():
    matrix = Matrix('examples/1.txt')
    cfp = CFP(matrix)
    start = time.time()
    cfp.solve()
    end = time.time()
    print(cfp.grouping_efficacy)
    print(end - start)


if __name__ == '__main__':
    main()
