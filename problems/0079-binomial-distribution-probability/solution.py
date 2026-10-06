from math import comb

def binomial_probability(n: int, k: int, p: float) -> float:
    a = comb(n,k)
    b = p**k
    c = (1-p)**(n-k)
    res = a*b*c
    return res