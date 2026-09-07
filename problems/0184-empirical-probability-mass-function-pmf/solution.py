import numpy as np

def empirical_pmf(samples):
    n,cnt = np.unique(samples, return_counts=True)
    p = cnt/len(samples)

    return list(zip(n,p))