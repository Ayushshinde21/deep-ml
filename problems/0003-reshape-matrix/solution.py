import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	x = np.array(a)
    if new_shape[0]*new_shape[1] == x.size:
        reshaped_matrix = x.reshape(new_shape)
        return reshaped_matrix
    else:
        return []