import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:
    A = np.array(A, dtype=float)

    ATA = A.T @ A

    eigenvalues, eigenvectors = np.linalg.eigh(ATA)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    s = np.sqrt(np.maximum(eigenvalues, 0))

    U = np.zeros((2, 2))

    for i in range(2):
        if s[i] > 1e-10:
            U[:, i] = (A @ eigenvectors[:, i]) / s[i]

    V = eigenvectors.T

    return U, s, V