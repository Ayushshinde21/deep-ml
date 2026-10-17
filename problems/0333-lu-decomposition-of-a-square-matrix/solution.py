import numpy as np

def lu_decomposition(A: list) -> tuple:
    A = np.array(A, dtype=float)
    n = A.shape[0]

    L = np.eye(n, dtype=float)
    U = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(
                L[i, k] * U[k, j] for k in range(i)
            )

        for j in range(i + 1, n):
            L[j, i] = (
                A[j, i] - sum(
                    L[j, k] * U[k, i] for k in range(i)
                )
            ) / U[i, i]

    return L, U