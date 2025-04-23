import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from network import NeuralNetwork, sigmoid, sigmoid_derivative
from sklearn.metrics import log_loss  # Correct import

train_df = pd.read_csv("data_train.csv")
valid_df = pd.read_csv("data_valid.csv")


X_train = train_df.drop(columns=['diagnosis']).values.T
y_train = train_df['diagnosis'].values.reshape(1, -1)

X_valid = valid_df.drop(columns=['diagnosis']).values.T
y_valid = valid_df['diagnosis'].values.reshape(1, -1)


# Initialize the neural network
model = NeuralNetwork(X_train.shape[0], 20, 10, 1)

epochs = 100
learning_rate = 0.1

train_losses = []
valid_losses = []

for epoch in range(epochs):
    epoch_loss = 0
    
    for i in range(X_train.shape[1]):
        x = X_train[:, i].reshape(-1, 1)
        y = y_train[:, i].reshape(-1, 1)

        activations = [x]  # This line should be indented inside the loop
        zs = []
        a = x
        
        for j in range(len(model.weights)):
            z = np.dot(model.weights[j].T, a) + model.biases[j]
            zs.append(z)
            if j < len(model.weights) - 1:
                 a = np.maximum(0, z)  # ReLU activation for hidden layers
            else:
                a = sigmoid(z)
            activations.append(a)
            
        # LOSS
        y_hat = activations[-1]
        loss = log_loss([y[0, 0]], [y_hat[0, 0]], labels=[0, 1])
        epoch_loss += loss

        # BACKPROPAGATION
        delta = y_hat - y  # dL/dz at output
        for j in reversed(range(len(model.weights))):
            a_prev = activations[j]
            dz = delta

            dw = np.dot(dz, a_prev.T)
            db = dz

            model.weights[j] -= learning_rate * dw
            model.biases[j] -= learning_rate * db

            if j > 0:
                z_prev = zs[j - 1]
                da = np.dot(model.weights[j].T, dz)
                dz = da * (z_prev > 0)  # Derivative of ReLU
                delta = dz

    # Epoch summary
    avg_train_loss = epoch_loss / X_train.shape[1]
    train_losses.append(avg_train_loss)

    # Validation
    y_pred_valid = model.forward(X_valid)
    val_loss = log_loss(y_valid.flatten(), y_pred_valid.flatten(), labels=[0, 1])
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


