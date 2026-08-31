import numpy as np

def linear_regression_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    iterations: int
) -> np.ndarray:

    m, n = X.shape

    theta = np.zeros(n, dtype=float)

    for _ in range(iterations):
        predictions = X @ theta

        error = predictions - y

        gradient = (1 / m) * (X.T @ error)

        theta = theta - alpha * gradient

    return theta