import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    data = np.array(data, dtype=float)

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    standardized = (data - mean) / std

    covariance = np.cov(standardized, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]

    components = eigenvectors[:, :k]

    for i in range(k):
        for j in range(components.shape[0]):
            if abs(components[j, i]) > 1e-10:
                if components[j, i] < 0:
                    components[:, i] *= -1
                break

    return np.round(components, 4)