import numpy as np

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
    logits = np.asarray(logits, dtype=float)

    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / np.sum(exp_logits)

    probabilities[target] -= 1

    return np.round(probabilities, 4).tolist()