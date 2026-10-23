import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    X_batch = X[batch_indices]
    y_batch = y[batch_indices]

    m = len(batch_indices)

    predictions = X_batch @ weights + bias
    errors = predictions - y_batch

    grad_weights = (2 / m) * (X_batch.T @ errors)
    grad_bias = (2 / m) * np.sum(errors)

    updated_weights = weights - lr * grad_weights
    updated_bias = bias - lr * grad_bias

    return np.append(updated_weights, updated_bias)