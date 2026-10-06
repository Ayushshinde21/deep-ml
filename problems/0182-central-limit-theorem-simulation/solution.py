import numpy as np

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:

    np.random.seed(seed)

    if distribution == 'uniform':
        samples = np.random.uniform(0, 1, size=(runs, n))
        mu = 0.5
        sigma = np.sqrt(1 / 12)

    elif distribution == 'exponential':
        samples = np.random.exponential(1.0, size=(runs, n))
        mu = 1.0
        sigma = 1.0

    elif distribution == 'bernoulli':
        samples = (np.random.rand(runs, n) < 0.3).astype(float)
        mu = 0.3
        sigma = np.sqrt(0.3 * 0.7)

    else:
        raise ValueError("Invalid distribution")

    sample_means = np.mean(samples, axis=1)
    standard_errors = sigma / np.sqrt(n)
    z_scores = (sample_means - mu) / standard_errors

    return {
        'mean': round(float(np.mean(z_scores)), 3),
        'std': round(float(np.std(z_scores)), 3)
    }