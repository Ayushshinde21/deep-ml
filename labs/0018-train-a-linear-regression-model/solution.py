import numpy as np

def train(X, y, W, b):

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    X_aug = np.column_stack((X, np.ones(X.shape[0])))
    params = np.linalg.lstsq(X_aug, y, rcond=None)[0]

    W = params[:-1]
    b = float(params[-1])

    return W, b