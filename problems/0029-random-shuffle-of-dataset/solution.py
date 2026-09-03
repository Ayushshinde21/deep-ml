import numpy as np

def shuffle_data(X, y, seed=None):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(X))

    return X[indices], y[indices]