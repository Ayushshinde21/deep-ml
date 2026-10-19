import numpy as np

def cramers_rule(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1)

    if A.ndim != 2 or A.shape[0] != A.shape[1] or len(b) != A.shape[0]:
        return -1

    det_A = np.linalg.det(A)

    if np.isclose(det_A, 0):
        return -1

    n = A.shape[0]
    x = np.zeros(n)

    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = b
        x[i] = np.linalg.det(Ai) / det_A

    return x