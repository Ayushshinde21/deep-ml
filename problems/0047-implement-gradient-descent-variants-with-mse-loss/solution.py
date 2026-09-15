import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float).copy()

    m = len(X)

    for _ in range(n_epochs):
        if method == 'batch':
            predictions = X @ weights
            errors = predictions - y
            gradient = (2 / m) * (X.T @ errors)
            weights -= learning_rate * gradient

        elif method == 'stochastic':
            for i in range(m):
                prediction = X[i] @ weights
                error = prediction - y[i]
                gradient = 2 * error * X[i]
                weights -= learning_rate * gradient

        elif method == 'mini_batch':
            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                X_batch = X[start:end]
                y_batch = y[start:end]

                predictions = X_batch @ weights
                errors = predictions - y_batch
                gradient = (2 / len(X_batch)) * (X_batch.T @ errors)
                weights -= learning_rate * gradient

        else:
            raise ValueError("Invalid method")

    return weights