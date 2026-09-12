import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    A = np.array(A, dtype=float, copy=True)
    m, n = A.shape
    rank = 0
    row = 0

    for col in range(n):
        if row >= m:
            break

        pivot = row + np.argmax(np.abs(A[row:, col]))

        if abs(A[pivot, col]) <= tol:
            continue

        A[[row, pivot]] = A[[pivot, row]]

        for r in range(row + 1, m):
            if abs(A[r, col]) > tol:
                factor = A[r, col] / A[row, col]
                A[r, col:] -= factor * A[row, col:]
                A[r, col] = 0.0

        rank += 1
        row += 1

    return rank