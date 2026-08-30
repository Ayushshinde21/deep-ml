import numpy as np

def matrix_dot_vector(a: list[list[int | float]],b: list[int | float]) -> list[int | float]:
    x = np.array(a)
    y = np.array(b)
    if x.shape[-1] == y.shape[0]:
        z = x @ y
        return z
    else:
        return -1