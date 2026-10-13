import numpy as np

class MyReducer:
    def __init__(self):
        self.n_components = 10
        self.mean_ = None
        self.components_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        covariance = (
            X_centered.T @ X_centered
        ) / max(X.shape[0] - 1, 1)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        k = min(self.n_components, X.shape[1])

        components = eigenvectors[:, order[:k]].copy()

        for j in range(k):
            i = np.argmax(np.abs(components[:, j]))
            if components[i, j] < 0:
                components[:, j] *= -1

        self.components_ = np.zeros(
            (X.shape[1], self.n_components)
        )
        self.components_[:, :k] = components

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_

    def fit_transform(self, X):
        return self.fit(X).transform(X)