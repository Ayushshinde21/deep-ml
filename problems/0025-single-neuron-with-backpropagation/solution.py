import numpy as np

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    weights = initial_weights.astype(float).copy()
    bias = float(initial_bias)

    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)

    mse_values = []
    n = len(features)

    for _ in range(epochs):
        z = features @ weights + bias
        predictions = 1 / (1 + np.exp(-z))

        errors = predictions - labels
        mse = np.mean(errors ** 2)
        mse_values.append(mse)

        delta = 2 * errors * predictions * (1 - predictions)
        weight_gradients = (features.T @ delta) / n
        bias_gradient = np.mean(delta)

        weights -= learning_rate * weight_gradients
        bias -= learning_rate * bias_gradient

    updated_weights = np.round(weights, 4)
    updated_bias = round(bias, 4)
    mse_values = [round(value, 4) for value in mse_values]

    return updated_weights, updated_bias, mse_values