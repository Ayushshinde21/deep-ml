import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	g = np.array(gradient)
	m = np.linalg.norm(g)
	if m == 0:
		d = np.zeros_like(g)
		dd = np.zeros_like(g)
	else:
		d = g / m
		dd = -d

	x = {
        "magnitude": float(m),
        "direction": d,
        "descent_direction": dd
    }

	return x