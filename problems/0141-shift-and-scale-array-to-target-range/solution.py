import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    a = np.min(values)
    b = np.max(values)

    if a == b:
        return np.full_like(values, c, dtype=float)

    return c + ((values - a) / (b - a)) * (d - c)