import numpy as np

def transform_basis(B: list[list[float]], C: list[list[float]]) -> list[list[float]]:
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)

    if np.linalg.det(B) == 0 or np.linalg.det(C) == 0:
        return -1

    P = np.linalg.inv(C) @ B

    return np.round(P, 4).tolist()