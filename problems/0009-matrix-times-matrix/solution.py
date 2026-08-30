import  numpy as np

def matrixmul(a:list[list[int|float]],b:list[list[int|float]])-> list[list[int|float]]:
    x = np.array(a)
    y = np.array(b)
    m1 = x.shape[-1]
    n2 = y.shape[0]
    if m1 != n2:
        return -1
    c = x @ y
    return c