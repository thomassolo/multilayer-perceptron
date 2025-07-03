import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import signal
from network import NeuralNetwork, sigmoid, sigmoid_derivative, reLU, reLU_derivative


# Custom log_loss to avoid sklearn dependency issues
def log_loss_custom(y_true, y_pred, eps=1e-15):
    """Custom implementation of log loss."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class RealTimeVisualizer:
    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 5))
        self.train_losses = []
        self.valid_losses = []
        self.epochs = []
        
        # Setup plots
        self.ax1.set_title("Real-time Loss")
        self.ax1.set_xlabel("Epoch")
        self.ax1.set_ylabel("Loss")
        
        self.ax2.set_title("Network Activity")
        self.ax2.set_xlabel("Layer")
        self.ax2.set_ylabel("Average Activation")
        
        plt.ion()  # Interactive mode
        plt.show()
    
    def update_loss_plot(self, epoch, train_loss, val_loss):
        """Update the loss plot in real-time."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.valid_losses.append(val_loss)
        
        self.ax1.clear()
        self.ax1.plot(self.epochs, self.train_losses, 'b-', label="Train Loss", linewidth=2)
        self.ax1.plot(self.epochs, self.valid_losses, 'r-', label="Validation Loss", linewidth=2)
        self.ax1.set_title("Real-time Loss")
        self.ax1.set_xlabel("Epoch")
        self.ax1.set_ylabel("Loss")
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        plt.pause(0.01)  # Small pause to update plot
    
    def update_network_activity(self, activations):
        """Visualize network layer activations."""
        avg_activations = [np.mean(act) for act in activations[1:]]  # Skip input
        layer_names = ["Hidden1", "Hidden2", "Output"]
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        self.ax2.clear()
        bars = self.ax2.bar(layer_names, avg_activations, color=colors, alpha=0.7)
        self.ax2.set_title("Network Layer Activity")
        self.ax2.set_ylabel("Average Activation")
        self.ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, val in zip(bars, avg_activations):
            self.ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.pause(0.01)


def clear_console():
    """Clear console for real-time updates."""
    os.system('cls' if os.name == 'nt' else 'clear')


def display_training_status(epoch, epochs, train_loss, val_loss, model, sample_data):
    """Display training status in console."""
    clear_console()
    
    print("=" * 70)
    print(f"🧠 MULTILAYER PERCEPTRON TRAINING - REAL TIME VISUALIZATION")
    print("=" * 70)
    print(f"📊 Epoch: {epoch + 1:3d}/{epochs}")
    print(f"📈 Training Loss:   {train_loss:.6f}")
    print(f"📉 Validation Loss: {val_loss:.6f}")
    print(f"📋 Loss Difference: {abs(train_loss - val_loss):.6f}")
    
    if train_loss < val_loss:
        print("✅ Model is learning well!")
    elif train_loss > val_loss * 1.1:
        print("⚠️  Possible overfitting detected")
    else:
        print("🎯 Training looks stable")
    
    print("-" * 70)
    
    # Show network activity
    activations, _ = forward_pass(model, sample_data[0:1])
    print("🔄 Network Layer Activities:")
    layer_names = ["Input   ", "Hidden1 ", "Hidden2 ", "Output  "]
    
    for i, (name, activation) in enumerate(zip(layer_names, activations)):
        avg_act = np.mean(activation)
        bar_length = int(avg_act * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        
        if i == 0:
            color = "🔵"
        elif i == len(layer_names) - 1:
            color = "🔴"
        else:
            color = "🟢"
        
        print(f"{color} {name}: {bar} {avg_act:.4f}")
    
    print("-" * 70)
    print("📱 Real-time plots updating in separate window...")
    print("⏹️  Press Ctrl+C to stop training")
    print("=" * 70)
    sys.stdout.flush()


def sig_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\n\n🛑 Training interrupted by user!")
    print("💾 Saving current model state...")
    sys.exit(0)


def load_data():
    """Load and prepare training and validation data."""
    if not os.path.exists("data_train.csv") or not os.path.exists("data_valid.csv"):
        raise FileNotFoundError("Training data files not found! Run split_data.py first.")
    
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


def train_epoch_with_vis(model, X_train, y_train, learning_rate, visualizer):
    """Train epoch with visualization of network activity."""
    indices = np.random.permutation(X_train.shape[0])
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    epoch_loss = 0

    for i in range(X_shuffled.shape[0]):
        x = X_shuffled[i:i+1]
        y = y_shuffled[i:i+1]

        # Forward pass
        activations, _ = forward_pass(model, x)
        
        # Visualize network activity every 100 samples
        if i % 100 == 0:
            visualizer.update_network_activity(activations)
        
        # Calculate loss
        y_hat = activations[-1]
        loss = log_loss_custom(y, y_hat)
        epoch_loss += loss

        # Backward pass
        backward_pass(model, activations, y, learning_rate)

    return epoch_loss / X_train.shape[0]


def validate_model(model, X_valid, y_valid):
    """Validate the model and return validation loss."""
    y_pred_valid = model.forward(X_valid)
    val_loss = log_loss_custom(y_valid, y_pred_valid)
    return val_loss


def plot_learning_curve(train_losses, valid_losses, save_path="loss_curve.png"):
    """Plot and save the final learning curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(valid_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Final Learning Curve")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    print(f"📊 Final learning curve saved as {save_path}")


def save_model(model, filepath="model_weights.npy"):
    """Save model weights and biases."""
    np.save(filepath, {"weights": model.weights, "biases": model.biases})
    print(f"💾 Model saved as {filepath}")


def train_model_with_visualization(input_dim=None, hidden_dims=[24, 24], output_dim=1, 
                                  epochs=100, learning_rate=0.0001, random_seed=42):
    """Main training function with both real-time plots and console visualization."""
    # Load data
    X_train, y_train, X_valid, y_valid = load_data()
    
    if input_dim is None:
        input_dim = X_train.shape[1]

    # Initialize the neural network and visualizer
    model = NeuralNetwork(input_dim, *hidden_dims, output_dim, random_seed=random_seed)
    visualizer = RealTimeVisualizer()
    
    train_losses = []
    valid_losses = []
    
    print(f"🚀 Starting training for {epochs} epochs with dual visualization...")
    print(f"📋 Network: {input_dim} → {hidden_dims[0]} → {hidden_dims[1]} → {output_dim}")
    print(f"⚙️  Learning rate: {learning_rate}")
    print(f"🎲 Random seed: {random_seed}")
    print("\n" + "="*70)
    
    try:
        for epoch in range(epochs):
            # Train for one epoch
            avg_train_loss = train_epoch_with_vis(model, X_train, y_train, learning_rate, visualizer)
            train_losses.append(avg_train_loss)

            # Validate
            val_loss = validate_model(model, X_valid, y_valid)
            valid_losses.append(val_loss)
            
            # Update real-time plot
            visualizer.update_loss_plot(epoch + 1, avg_train_loss, val_loss)
            
            # Update console display
            display_training_status(epoch, epochs, avg_train_loss, val_loss, model, X_train)
            
            # Small delay to make visualization readable
            import time
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Training stopped at epoch {epoch + 1}")
    
    return model, train_losses, valid_losses


def main():
    """Main function with real-time visualization."""
    signal.signal(signal.SIGINT, sig_handler)
    
    print("🧠 MULTILAYER PERCEPTRON TRAINER")
    print("=" * 50)
    print("🎨 Starting with dual visualization (plots + console)")
    print("📊 Real-time plots will open in a separate window")
    print("💻 Console will show live training statistics")
    print("=" * 50)
    
    # Train with visualization
    model, train_losses, valid_losses = train_model_with_visualization()
    
    # Save model and final plots
    save_model(model)
    plot_learning_curve(train_losses, valid_losses)
    
    print("\n" + "="*70)
    print("🎉 Training completed successfully!")
    print(f"📊 Final training loss: {train_losses[-1]:.6f}")
    print(f"📈 Final validation loss: {valid_losses[-1]:.6f}")
    print("💾 Model and plots saved!")
    print("=" * 70)


if __name__ == "__main__":
    main()

