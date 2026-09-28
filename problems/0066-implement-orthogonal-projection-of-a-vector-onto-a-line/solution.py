import numpy as np

def orthogonal_projection(v, L):
	v = np.array(v)
	L = np.array(L)
	x = np.dot(v, L)
	y = np.dot(L, L)
	if y==0:
		return None
	ans = (x/y)*L
	return ans