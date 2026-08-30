import  numpy as np

def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    A = np.array(matrix)
    det = np.linalg.det(A)
    if (det==0):
        return None
    x = np.linalg.inv(A)
    return  x