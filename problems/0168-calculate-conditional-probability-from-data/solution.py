def conditional_probability(data, x, y):
    x_count = 0
    y_count = 0

    for X, Y in data:
        if X == x:
            x_count += 1
            if Y == y:
                y_count += 1

    if x_count == 0:
        return 0.0

    return round(y_count / x_count, 4)