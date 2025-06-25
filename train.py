import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from network import NeuralNetwork, sigmoid, sigmoid_derivative, reLU, reLU_derivative
from sklearn.metrics import log_loss
from utils import sig_handler
import signal


def load_data():
    """Load and prepare training and validation data."""
    train_df = pd.read_csv("data_train.csv")
    valid_df = pd.read_csv("data_valid.csv")

    X_train = train_df.drop(columns=['diagnosis']).values
    y_train = train_df['diagnosis'].values.reshape(-1, 1)

    X_valid = valid_df.drop(columns=['diagnosis']).values
    y_valid = valid_df['diagnosis'].values.reshape(-1, 1)
    
    return X_train, y_train, X_valid, y_valid


def forward_pass(model, x):
    """Perform forward pass through the network."""
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
    
    return activations, zs


def backward_pass(model, activations, y, learning_rate):
    """Perform backward pass and update weights."""
    y_hat = activations[-1]
    
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


def train_epoch(model, X_train, y_train, learning_rate):
    """Train the model for one epoch."""
    # Shuffle the training data
    indices = np.random.permutation(X_train.shape[0])
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    epoch_loss = 0

    # Process all samples in the training set
    for i in range(X_shuffled.shape[0]):
        x = X_shuffled[i:i+1]  # Get a single sample as (1, features)
        y = y_shuffled[i:i+1]   # Get corresponding label as (1, 1)

        # Forward pass
        activations, _ = forward_pass(model, x)
        
        # Calculate loss
        y_hat = activations[-1]
        loss = log_loss([y[0, 0]], [y_hat[0, 0]], labels=[0, 1])
        epoch_loss += loss

        # Backward pass
        backward_pass(model, activations, y, learning_rate)

    return epoch_loss / X_train.shape[0]


def validate_model(model, X_valid, y_valid):
    """Validate the model and return validation loss."""
    y_pred_valid = model.forward(X_valid)
    val_loss = log_loss(y_valid, y_pred_valid, labels=[0, 1])
    return val_loss


def plot_learning_curve(train_losses, valid_losses, save_path="loss_curve.png"):
    """Plot and save the learning curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Learning Curve")
    plt.savefig(save_path)
    plt.show()


def save_model(model, filepath="model_weights.npy"):
    """Save model weights and biases."""
    np.save(filepath, {"weights": model.weights, "biases": model.biases})


def train_model(input_dim=None, hidden_dims=[24, 24], output_dim=1, 
                epochs=100, learning_rate=0.1, random_seed=42):
    """Main training function."""
    # Load data
    X_train, y_train, X_valid, y_valid = load_data()
    
    # Set input dimension if not provided
    if input_dim is None:
        input_dim = X_train.shape[1]

    # Initialize the neural network
    model = NeuralNetwork(input_dim, *hidden_dims, output_dim, random_seed=random_seed)
   
    train_losses = []
    valid_losses = []
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Train for one epoch
        avg_train_loss = train_epoch(model, X_train, y_train, learning_rate)
        train_losses.append(avg_train_loss)

        # Validate
        val_loss = validate_model(model, X_valid, y_valid)
        valid_losses.append(val_loss)
        
        print(f"epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f} - val_loss: {val_loss:.4f}")
    
   
    return model, train_losses, valid_losses

def main():
    """Main function to run the training."""
    # Set up signal handler for graceful termination
    signal.signal(signal.SIGINT, sig_handler)
    
    # Train the model with default parameters
    model, train_losses, valid_losses = train_model()
    
    # Save the model weights and biases
    save_model(model)
    
    # Plot the learning curve
    plot_learning_curve(train_losses, valid_losses)
    model, train_losses, valid_losses = train_model()

if __name__ == "__main__":
    main()

