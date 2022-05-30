import sys
import time
from structures.matrix import Matrix
from structures.cfp import CFP
from func_timeout import func_timeout, FunctionTimedOut

sys.setrecursionlimit(10000)


def main():
    matrix = Matrix('examples/1.txt')
    cfp = CFP(matrix)
    start = time.time()
    try:
        func_timeout(3600, cfp.solve, args=())
    except FunctionTimedOut:
        print('Finished with timeout')
    finally:
        end = time.time()
        print(cfp.grouping_efficacy)
        print(end - start)


if __name__ == '__main__':
    main()
