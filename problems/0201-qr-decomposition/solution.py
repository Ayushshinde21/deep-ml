import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    k = min(m, n)

    Q = np.zeros((m, k))
    R = np.zeros((k, n))
    tol = 1e-12

    for j in range(n):
        v = A[:, j].copy()

        for i in range(min(j, k)):
            R[i, j] = np.dot(Q[:, i], v)
            v -= R[i, j] * Q[:, i]

        if j < k:
            norm = np.linalg.norm(v)

            if norm > tol:
                R[j, j] = norm
                Q[:, j] = v / norm
            else:
                for axis in range(m):
                    candidate = np.zeros(m)
                    candidate[axis] = 1.0

                    for i in range(j):
                        candidate -= np.dot(Q[:, i], candidate) * Q[:, i]

                    candidate_norm = np.linalg.norm(candidate)

                    if candidate_norm > tol:
                        Q[:, j] = candidate / candidate_norm
                        break

    return Q.tolist(), R.tolist()