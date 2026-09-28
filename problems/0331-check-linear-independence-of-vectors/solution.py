import numpy as np

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    if len(vectors) == 0:
        return True

    A = np.array(vectors, dtype=float)

    rank = 0
    rows, cols = A.shape
    tol = 1e-10

    for col in range(cols):
        if rank >= rows:
            break

        pivot = rank + np.argmax(np.abs(A[rank:, col]))

        if abs(A[pivot, col]) <= tol:
            continue

        A[[rank, pivot]] = A[[pivot, rank]]

        for r in range(rank + 1, rows):
            factor = A[r, col] / A[rank, col]
            A[r, col:] -= factor * A[rank, col:]

        rank += 1

    return rank == rows