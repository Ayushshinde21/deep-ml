import numpy as np

def matrix_image(A):
    A = np.asarray(A, dtype=float)
    work = A.copy()

    rows, cols = work.shape
    pivot_cols = []
    pivot_row = 0
    tol = 1e-10

    for col in range(cols):
        if pivot_row >= rows:
            break

        pivot = pivot_row + np.argmax(np.abs(work[pivot_row:, col]))

        if abs(work[pivot, col]) <= tol:
            continue

        work[[pivot_row, pivot]] = work[[pivot, pivot_row]]

        for r in range(pivot_row + 1, rows):
            factor = work[r, col] / work[pivot_row, col]
            work[r, col:] -= factor * work[pivot_row, col:]
            work[r, col] = 0.0

        pivot_cols.append(col)
        pivot_row += 1

    return A[:, pivot_cols].tolist()