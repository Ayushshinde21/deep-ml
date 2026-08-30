import numpy as np

def calculate_covariance_matrix(vectors: list[list[float | int]]) -> list[list[float]]:
    matrix = np.array(vectors, dtype=float)
    covariance = np.cov(matrix, rowvar=True)

    return covariance.tolist()