import numpy as np

def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
    
    Args:
        x: Input value
        
    Returns:
        Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
    """
    sigmoid = 1 / (1 + np.exp(-x))
    sigmoid_derivative = sigmoid * (1 - sigmoid)

    tanh_value = np.tanh(x)
    tanh_derivative = 1 - tanh_value ** 2

    relu_derivative = 1.0 if x > 0 else 0.0

    return {
        'sigmoid': float(sigmoid_derivative),
        'tanh': float(tanh_derivative),
        'relu': relu_derivative
    }