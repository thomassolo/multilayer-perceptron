import numpy as np

def sigmoid(x):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Compute the derivative of the sigmoid function."""
    return x * (1 - x)

def reLU(x):
    """Compute the ReLU function."""
    return np.maximum(0, x)

def reLU_derivative(x):
    """Compute the derivative of the ReLU function."""
    return np.where(x > 0, 1, 0)



class NeuralNetwork:
    def init(slef, layers):
        