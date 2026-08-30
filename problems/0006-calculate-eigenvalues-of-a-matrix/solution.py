import numpy as np
import cmath

def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    c = np.linalg.det(matrix)
    b = -np.trace(matrix)
    a = 1

    discriminant = (b ** 2) - (4 * a * c)
    root1 = float((-b - cmath.sqrt(discriminant)) / (2 * a))
    root2 = float((-b + cmath.sqrt(discriminant)) / (2 * a))
	
    eigenvalues = [root1, root2]

    return sorted(eigenvalues, reverse=True)