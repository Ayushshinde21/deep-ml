import numpy as np

def rref(matrix):
    matrix = np.array(matrix, dtype=float, copy=True)
    rows, cols = matrix.shape
    pivot_row = 0
    tol = 1e-10

    for col in range(cols):
        if pivot_row >= rows:
            break

        pivot = pivot_row + np.argmax(np.abs(matrix[pivot_row:, col]))

        if abs(matrix[pivot, col]) <= tol:
            continue

        matrix[[pivot_row, pivot]] = matrix[[pivot, pivot_row]]

        matrix[pivot_row] = matrix[pivot_row] / matrix[pivot_row, col]

        for r in range(rows):
            if r != pivot_row:
                matrix[r] = matrix[r] - matrix[r, col] * matrix[pivot_row]

        pivot_row += 1

    matrix[np.abs(matrix) < tol] = 0.0

    return matrix