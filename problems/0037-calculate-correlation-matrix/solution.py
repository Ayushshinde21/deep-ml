import numpy as np

def calculate_correlation_matrix(X, Y=None):
    X = np.asarray(X, dtype=float)

    if Y is None:
        return np.corrcoef(X, rowvar=False)

    Y = np.asarray(Y, dtype=float)

    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)

    numerator = X_centered.T @ Y_centered
    denominator = np.sqrt(
        np.sum(X_centered ** 2, axis=0)[:, None] *
        np.sum(Y_centered ** 2, axis=0)[None, :]
    )

    return numerator / denominator