from typing import Callable

def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
    x = [float(v) for v in point]
    n = len(x)
    H = [[0.0] * n for _ in range(n)]
    f0 = f(x)

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[i] += h
        x_minus[i] -= h

        H[i][i] = (f(x_plus) - 2 * f0 + f(x_minus)) / (h * h)

        for j in range(i + 1, n):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()

            x_pp[i] += h
            x_pp[j] += h

            x_pm[i] += h
            x_pm[j] -= h

            x_mp[i] -= h
            x_mp[j] += h

            x_mm[i] -= h
            x_mm[j] -= h

            value = (
                f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)
            ) / (4 * h * h)

            H[i][j] = value
            H[j][i] = value

    return H