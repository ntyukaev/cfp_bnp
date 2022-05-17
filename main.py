import sys
import time
from structures.matrix import Matrix
from structures.cfp import CFP

sys.setrecursionlimit(10000)


def main():
    matrix = Matrix('examples/13.txt')
    cfp = CFP(matrix)
    start = time.time()
    cfp.solve()
    end = time.time()
    print(cfp.grouping_efficacy)
    print(end - start)


if __name__ == '__main__':
    main()
