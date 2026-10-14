import numpy as np

def impute(X: np.ndarray) -> np.ndarray:
    X_clean = np.asarray(X, dtype=float).copy()

    medians = np.nanmedian(X_clean, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)

    rows, cols = np.where(np.isnan(X_clean))
    X_clean[rows, cols] = medians[cols]

    return X_clean