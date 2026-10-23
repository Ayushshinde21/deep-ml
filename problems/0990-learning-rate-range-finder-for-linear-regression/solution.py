import numpy as np

def lr_range_finder(X, y, w0, a, b, n_steps):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w0, dtype=float).copy()

    n_samples = X.shape[0]
    losses = []

    for k in range(n_steps):
        if n_steps == 1:
            lr = 10 ** a
        else:
            exponent = a + (b - a) * k / (n_steps - 1)
            lr = 10 ** exponent

        gradient = (2 / n_samples) * (X.T @ (X @ w - y))
        w = w - lr * gradient

        predictions = X @ w
        loss = np.mean((predictions - y) ** 2)
        losses.append(float(loss))

    return losses