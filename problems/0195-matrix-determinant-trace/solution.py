import numpy as np

def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
    matrix = np.array(matrix, dtype=float)

    determinant = float(np.linalg.det(matrix))
    trace = float(np.trace(matrix))

    return determinant, trace