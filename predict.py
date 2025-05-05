import numpy as np
import pandas as pd
from network import NeuralNetwork, sigmoid
from sklearn.metrics import accuracy_score, log_loss

# Load weights/biases
model_data = np.load("model_weights.npy", allow_pickle=True).item()
weights = model_data["weights"]
biases = model_data["biases"]

# Recreate the model structure
# Get input dimension from the first weight matrix
input_dim = weights[0].shape[0]
hidden_dim1 = weights[0].shape[1]
hidden_dim2 = weights[1].shape[1]
output_dim = weights[2].shape[1]

# Initialize with correct layer dimensions
model = NeuralNetwork(input_dim, hidden_dim1, hidden_dim2, output_dim)
model.weights = weights
model.biases = biases

# Load data to predict
df = pd.read_csv("data_valid.csv")
X = df.drop(columns=["diagnosis"]).values  # Remove transpose
Y = df["diagnosis"].values.reshape(-1, 1)  # Reshape to match forward pass

# Predict
predictions = model.forward(X)  # outputs = probabilities between 0 and 1

# Transform to class: 1 if prob > 0.5
predicted_classes = (predictions > 0.5).astype(int)

# Evaluate
accuracy = accuracy_score(Y.flatten(), predicted_classes.flatten())
loss = log_loss(Y.flatten(), predictions.flatten())

print(f"✅ Accuracy: {accuracy * 100:.2f}%")
print(f"🔍 Cross-entropy loss: {loss:.4f}")

# Display the first 10 predictions
for i in range(10):
    proba = predictions[i, 0]  # Changed indexing to match data orientation
    pred = predicted_classes[i, 0]
    true = Y[i, 0]
    print(f"Example {i+1}: prob = {proba:.3f} → prediction = {pred} (true = {true})")