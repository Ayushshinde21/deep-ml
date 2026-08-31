import numpy as np

def feature_scaling(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    standardized = (data - mean) / std

    min_value = np.min(data, axis=0)
    max_value = np.max(data, axis=0)

    normalized = (data - min_value) / (max_value - min_value)

    standardized = np.round(standardized, 4)
    normalized = np.round(normalized, 4)

    return standardized, normalized