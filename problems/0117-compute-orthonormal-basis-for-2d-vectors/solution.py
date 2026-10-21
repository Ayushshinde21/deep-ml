import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    basis = []

    for vector in vectors:
        v = np.array(vector, dtype=float)

        for q in basis:
            v = v - np.dot(v, q) * q

        norm = np.linalg.norm(v)

        if norm > tol:
            basis.append(v / norm)

    return basis