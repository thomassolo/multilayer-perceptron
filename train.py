import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from network import NeuralNetwork, sigmoid, sigmoid_derivative, reLU, reLU_derivative
from sklearn.metrics import log_loss


train_df = pd.read_csv("data_train.csv")
valid_df = pd.read_csv("data_valid.csv")

X_train = train_df.drop(columns=['diagnosis']).values
y_train = train_df['diagnosis'].values.reshape(-1, 1)

X_valid = valid_df.drop(columns=['diagnosis']).values
y_valid = valid_df['diagnosis'].values.reshape(-1, 1)

# Get input dimension from data
input_dim = X_train.shape[1]

# Initialize the neural network with individual layer sizes
model = NeuralNetwork(input_dim, 24, 24, 1)

epochs = 100
learning_rate = 0.1

train_losses = []
valid_losses = []

for epoch in range(epochs):
    # Shuffle the training data
    indices = np.random.permutation(X_train.shape[0])
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    epoch_loss = 0

    # Process all samples in the training set
    for i in range(X_shuffled.shape[0]):
        x = X_shuffled[i:i+1]  # Get a single sample as (1, features)
        y = y_shuffled[i:i+1]   # Get corresponding label as (1, 1)

        # Forward pass - storing activations and z values
        activations = [x]
        zs = []
        a = x
        
        for j in range(len(model.weights)):
            z = np.dot(a, model.weights[j]) + model.biases[j]
            zs.append(z)
            if j < len(model.weights) - 1:
                a = np.maximum(0, z)  # ReLU activation for hidden layers
            else:
                a = sigmoid(z)  # Sigmoid for output layer
            activations.append(a)
            
        # Calculate loss
        y_hat = activations[-1]
        loss = log_loss([y[0, 0]], [y_hat[0, 0]], labels=[0, 1])
        epoch_loss += loss

        # Backpropagation
        # Initialize output layer error
        error = y_hat - y
        
        # Loop through layers backwards
        for j in reversed(range(len(model.weights))):
            # Get activations for current layer
            activation = activations[j+1]
            
            if j == len(model.weights) - 1:
                # For output layer (sigmoid)
                delta = error * sigmoid_derivative(activation)
            else:
                # For hidden layers (ReLU)
                delta = error * reLU_derivative(activation)
            
            # Calculate gradients
            dw = np.dot(activations[j].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)
            
            # Store error for next layer
            error = np.dot(delta, model.weights[j].T)
            
            # Update weights and biases
            model.weights[j] -= learning_rate * dw
            model.biases[j] -= learning_rate * db

    # Calculate average loss for epoch
    avg_train_loss = epoch_loss / X_train.shape[0]
    train_losses.append(avg_train_loss)

    # Validation
    y_pred_valid = model.forward(X_valid)
    val_loss = log_loss(y_valid, y_pred_valid, labels=[0, 1])
    valid_losses.append(val_loss)

    print(f"epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f} - val_loss: {val_loss:.4f}")

# Save model
np.save("model_weights.npy", {"weights": model.weights, "biases": model.biases})

# Plot losses
plt.plot(train_losses, label="Train Loss")
plt.plot(valid_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Learning Curve")
plt.savefig("loss_curve.png")
plt.show()