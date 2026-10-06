import numpy as np

def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
	a = np.array(priors)
	b = np.array(likelihoods)
	c  = a*b
	d = c.sum()
	return c/d