import math

def single_neuron_model(features: list[list[float | int]],
                        labels: list[int],
                        weights: list[float | int],
                        bias: float | int) -> tuple[list[float], float]:

    predictions = []

    for feature_vector in features:
        z = sum(x * w for x, w in zip(feature_vector, weights)) + bias
        probability = 1 / (1 + math.exp(-z))
        predictions.append(probability)

    mse = sum(
        (prediction - label) ** 2
        for prediction, label in zip(predictions, labels)
    ) / len(labels)

    predictions = [round(value, 4) for value in predictions]
    mse = round(mse, 4)

    return predictions, mse