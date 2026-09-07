def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
	x,y = len(a),len(b)
	c = []*0
	if (x!=y):
		return -1
	if (x==y):
		for i in range(x):
			c.append(a[i] + b[i])
	
	return c