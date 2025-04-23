import numpy as np

#activation functions
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
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        # Define network architecture
        self.layers = [input_size, hidden_size1, hidden_size2, output_size]
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(self.layers) - 1):
            # He initialization for weights
            weight = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2 / self.layers[i])
            bias = np.zeros((1, self.layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, X):
        # Forward pass
        activations = [X]
        zs = []
        a = X
        
        for j in range(len(self.weights)):
            z = np.dot(a, self.weights[j]) + self.biases[j]
            zs.append(z)
            if j < len(self.weights) - 1:
                a = np.maximum(0, z)  # ReLU activation for hidden layers
            else:
                a = sigmoid(z)  # Sigmoid activation for output layer
            activations.append(a)
            
        return activations[-1]  # Return final layer output

