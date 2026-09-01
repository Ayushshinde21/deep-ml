import math

def softmax(scores: list[float | int]) -> list[float]:
    max_score = max(scores)
    exp_values = [math.exp(score - max_score) for score in scores]
    total = sum(exp_values)

    return [round(value / total, 4) for value in exp_values]