import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:

    g = np.poly1d(g_coeffs)
    h = np.poly1d(h_coeffs)
    g1 = np.polyder(g)
    h1 = np.polyder(h)
    
    n = g1*h - h1*g
    d = h**2

    n = np.polyval(n,x)
    d = np.polyval(d,x)

    return (n/d)