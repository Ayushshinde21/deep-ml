import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if x0 is None:
        x = np.zeros_like(b, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()

    r = b - A @ x
    p = r.copy()
    rs_old = np.dot(r, r)

    if np.sqrt(rs_old) <= tol:
        return x.tolist()

    for _ in range(n):
        Ap = A @ p
        denominator = np.dot(p, Ap)

        if abs(denominator) < np.finfo(float).eps:
            break

        alpha = rs_old / denominator
        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = np.dot(r, r)

        if np.sqrt(rs_new) <= tol:
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x.tolist()