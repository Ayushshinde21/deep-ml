import numpy as np

def linear_regression_normal_equation(X: list[list[float | int]],
                                      y: list[float | int]) -> list[float]:

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

    return [round(float(value), 4) for value in coefficients]