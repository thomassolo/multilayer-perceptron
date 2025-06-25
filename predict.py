import numpy as np
import pandas as pd
from network import NeuralNetwork, sigmoid
from sklearn.metrics import accuracy_score, log_loss


def load_model(filepath="model_weights.npy"):
    """Load the trained model weights and biases."""
    model_data = np.load(filepath, allow_pickle=True).item()
    weights = model_data["weights"]
    biases = model_data["biases"]
    return weights, biases


def create_model_from_weights(weights, biases):
    """Recreate the model structure from saved weights and biases."""
    # Get dimensions from weight matrices
    input_dim = weights[0].shape[0]
    hidden_dim1 = weights[0].shape[1]
    hidden_dim2 = weights[1].shape[1]
    output_dim = weights[2].shape[1]
    
    # Initialize model with correct layer dimensions
    model = NeuralNetwork(input_dim, hidden_dim1, hidden_dim2, output_dim)
    model.weights = weights
    model.biases = biases
    
    return model


def load_test_data(filepath="data_valid.csv"):
    """Load test/validation data for prediction."""
    df = pd.read_csv(filepath)
    X = df.drop(columns=["diagnosis"]).values
    Y = df["diagnosis"].values.reshape(-1, 1)
    return X, Y


def make_predictions(model, X, threshold=0.5):
    """Make predictions using the trained model."""
    # Get probabilities
    predictions = model.forward(X)
    
    # Convert to class predictions
    predicted_classes = (predictions > threshold).astype(int)
    
    return predictions, predicted_classes


def evaluate_model(Y_true, predictions, predicted_classes):
    """Evaluate model performance using accuracy and loss metrics."""
    accuracy = accuracy_score(Y_true.flatten(), predicted_classes.flatten())
    loss = log_loss(Y_true.flatten(), predictions.flatten())
    
    return accuracy, loss


def display_predictions(predictions, predicted_classes, Y_true, num_examples=10):
    """Display sample predictions for visual inspection."""
    print(f"\nFirst {num_examples} predictions:")
    print("-" * 50)
    print("Note: 0 = Benign (non-cancerous), 1 = Malignant (cancerous)")
    print("-" * 50)
    
    for i in range(min(num_examples, len(predictions))):
        proba = predictions[i, 0]
        pred = predicted_classes[i, 0]
        true = Y_true[i, 0]
        status = "✅" if pred == true else "❌"
        
        # Add descriptive labels
        pred_label = "Malignant" if pred == 1 else "Benign"
        true_label = "Malignant" if true == 1 else "Benign"
        
        print(f"Example {i+1}: prob = {proba:.3f} → predicted = {pred} ({pred_label}) | actual = {true} ({true_label}) {status}")


def predict_and_evaluate(model_filepath="model_weights.npy", 
                        data_filepath="data_valid.csv",
                        threshold=0.5,
                        display_samples=10):
    """Main prediction function that orchestrates the entire prediction process."""
    print("Loading model...")
    weights, biases = load_model(model_filepath)
    model = create_model_from_weights(weights, biases)
    
    print("Loading test data...")
    X, Y = load_test_data(data_filepath)
    
    print("Making predictions...")
    predictions, predicted_classes = make_predictions(model, X, threshold)
    
    print("Evaluating model...")
    accuracy, loss = evaluate_model(Y, predictions, predicted_classes)
    
    # Display results
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"✅ Accuracy: {accuracy * 100:.2f}%")
    print(f"🔍 Cross-entropy loss: {loss:.4f}")
    
    if display_samples > 0:
        display_predictions(predictions, predicted_classes, Y, display_samples)
    
    return {
        'model': model,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'accuracy': accuracy,
        'loss': loss
    }


def predict_single_sample(model, sample):
    """Predict a single sample."""
    if sample.ndim == 1:
        sample = sample.reshape(1, -1)
    
    prediction = model.forward(sample)
    predicted_class = (prediction > 0.5).astype(int)
    
    return prediction[0, 0], predicted_class[0, 0]


def batch_predict(model_filepath, data_filepath, output_filepath=None, threshold=0.5):
    """Make batch predictions and optionally save to file."""
    weights, biases = load_model(model_filepath)
    model = create_model_from_weights(weights, biases)
    
    X, Y = load_test_data(data_filepath)
    predictions, predicted_classes = make_predictions(model, X, threshold)
    
    if output_filepath:
        # Save predictions to file
        results_df = pd.DataFrame({
            'true_label': Y.flatten(),
            'probability': predictions.flatten(),
            'predicted_class': predicted_classes.flatten()
        })
        results_df.to_csv(output_filepath, index=False)
        print(f"Predictions saved to {output_filepath}")
    
    return predictions, predicted_classes


if __name__ == "__main__":
    # Run prediction and evaluation
    results = predict_and_evaluate()
    
    # Example of predicting a single sample
    # sample_data, _ = load_test_data()
    # prob, pred_class = predict_single_sample(results['model'], sample_data[0])
    # print(f"\nSingle sample prediction: prob={prob:.3f}, class={pred_class}")