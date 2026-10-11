import numpy as np

def compute_null_space(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    A = np.asarray(A, dtype=float)

    U, s, Vh = np.linalg.svd(A, full_matrices=True)

    rank = np.sum(s > tol)

    null_space = Vh[rank:, :].T

    return null_space