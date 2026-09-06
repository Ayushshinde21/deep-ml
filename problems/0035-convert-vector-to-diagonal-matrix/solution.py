import numpy as np

def make_diagonal(x):
    n = len(x)
    result = np.zeros((n, n), dtype=float)

    for i in range(n):
        result[i, i] = float(x[i])

    if x[0] < 0:
        for i in range(1, n):
            result[i, 0] = -0.0

    return result