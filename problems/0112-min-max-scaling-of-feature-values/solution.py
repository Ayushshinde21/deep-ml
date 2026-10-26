def min_max(x: list[float]) -> list[float]:
    minimum = min(x)
    maximum = max(x)

    if maximum == minimum:
        return [0.0] * len(x)

    return [(value - minimum) / (maximum - minimum) for value in x]