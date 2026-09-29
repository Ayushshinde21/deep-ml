import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
    x = np.array(x, dtype=float)
    f0 = np.array(f(x.tolist()), dtype=float)

    m = len(f0)
    n = len(x)

    J = np.zeros((m, n))

    for j in range(n):
        x_forward = x.copy()
        x_backward = x.copy()

        x_forward[j] += h
        x_backward[j] -= h

        f_forward = np.array(f(x_forward.tolist()), dtype=float)
        f_backward = np.array(f(x_backward.tolist()), dtype=float)

        J[:, j] = (f_forward - f_backward) / (2 * h)

    return J.tolist()