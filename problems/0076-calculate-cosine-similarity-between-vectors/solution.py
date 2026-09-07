import numpy as np

def cosine_similarity(v1, v2):
	x = np.dot(v1, v2)
	y1 = np.linalg.norm(v1)
	y2 = np.linalg.norm(v2)
	y = y1*y2

	if y1 == 0 or y2 == 0:
		return 0.0
	else:
		return  (x/y)