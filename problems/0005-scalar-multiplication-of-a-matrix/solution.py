import numpy as np

def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	x = np.array(matrix)
	a = x*scalar
	return a