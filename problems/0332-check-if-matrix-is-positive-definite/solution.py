import numpy as np

def check_positive_definite(matrix: list) -> dict:
    A = np.asarray(matrix, dtype=float)

    is_symmetric = np.allclose(A, A.T, atol=1e-10)

    if is_symmetric:
        eigenvalues = np.linalg.eigvalsh(A)
    else:
        eigenvalues = np.linalg.eigvals(A)

    eigenvalues = sorted(eigenvalues, key=lambda v: (v.real, v.imag))

    result = []
    for value in eigenvalues:
        if abs(value.imag) < 1e-10:
            result.append(round(float(value.real), 4))
        else:
            result.append(complex(round(float(value.real), 4),
                                  round(float(value.imag), 4)))

    is_positive_definite = (
        is_symmetric and all(value > 1e-10 for value in eigenvalues)
    )

    return {
        'is_positive_definite': bool(is_positive_definite),
        'eigenvalues': result
    }