import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float).copy()

    n_samples = len(X)

    for i in range(n_iter):
        index = i % n_samples
        prediction = X[index] @ weights
        error = prediction - y[index]
        gradient = 2 * error * X[index]
        weights -= learning_rate * gradient

    return weights.tolist()