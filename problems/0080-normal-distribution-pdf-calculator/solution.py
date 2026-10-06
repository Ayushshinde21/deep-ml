import numpy as np

def normal_pdf(x, mean, std_dev):
	a = 0.398942280401433 * (1/std_dev)
	b = -(((x-mean)**2)/(2*(std_dev**2)))
	c = np.exp(b)
	return round(a*c,5)