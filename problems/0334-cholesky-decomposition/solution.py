import numpy as np

def cholesky_decomposition(A):
    try:
        A = np.asarray(A, dtype=float)

        if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
            return -1

        if not np.all(np.isfinite(A)):
            return -1

        if not np.allclose(A, A.T, rtol=0.0, atol=1e-10):
            return -1

        L = np.linalg.cholesky(A)

        return L.tolist()

    except (ValueError, TypeError, np.linalg.LinAlgError):
        return -1