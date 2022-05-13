import numpy as np
import math


def read(path):
    with open(path, 'r') as f:
        arr = []
        lines = f.read().splitlines()
        for line in lines:
            if line:
                row = list(map(int, filter(bool, line.split(' '))))
                if row:
                    arr.append(row)
    return np.array(arr)


def convert_number(a):
    if a.is_integer():
        return a
    if math.isclose(2.0, a, rel_tol=1e-07):
        return 2.0
    if math.isclose(1.0, a, rel_tol=1e-08):
        return 1.0
    if math.isclose(1, 1 + a, rel_tol=1e-08):
        return 0.0
    return a


def convert_number_separation(a):
    if a.is_integer():
        return a
    if math.isclose(2.0, a, rel_tol=1e-07):
        return 2.0
    if math.isclose(1.0, a, rel_tol=1e-08):
        return 1.0
    if math.isclose(1, 1 + a, rel_tol=1e-04):
        return 0.0
    return a


if __name__ == '__main__':
    test = read('examples/1.txt')
    print(test)
