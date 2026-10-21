
import math

PI = 3.14159

def power_grid_forecast(consumption_data):
    n = len(consumption_data)
    days = list(range(1, n + 1))

    detrended = [
        value - 10 * math.sin(2 * PI * day / 10)
        for day, value in zip(days, consumption_data)
    ]

    mean_day = sum(days) / n
    mean_usage = sum(detrended) / n

    numerator = sum(
        (day - mean_day) * (usage - mean_usage)
        for day, usage in zip(days, detrended)
    )
    denominator = sum((day - mean_day) ** 2 for day in days)

    slope = numerator / denominator
    intercept = mean_usage - slope * mean_day

    day15_base = intercept + slope * 15
    day15_fluctuation = 10 * math.sin(2 * PI * 15 / 10)

    forecast = round(day15_base + day15_fluctuation)

    return math.ceil(forecast * 1.05)


consumption_data = [150, 165, 185, 195, 210, 225, 240, 260, 275, 290]
print(power_grid_forecast(consumption_data))
