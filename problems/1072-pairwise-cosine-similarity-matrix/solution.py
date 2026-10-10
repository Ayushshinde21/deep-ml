import numpy as np

def pairwise_cosine_similarity(X):
    X = np.asarray(X, dtype=float)

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    normalized = np.divide(
        X,
        norms,
        out=np.zeros_like(X),
        where=norms != 0
    )

    similarity = normalized @ normalized.T

    return np.round(similarity, 4).tolist()