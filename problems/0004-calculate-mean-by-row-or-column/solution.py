import numpy as np
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	x = np.array(matrix)
	if mode=="column":
		a = x.mean(axis = 0)
	else:
		a = x.mean(axis = 1)
	return a