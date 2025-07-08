import numpy as np
import pandas as pd
from network import NeuralNetwork, sigmoid, softmax
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report
import signal
from utils import sig_handler   

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
    
    print(f"📋 Model loaded with architecture: {input_dim}-{hidden_dim1}-{hidden_dim2}-{output_dim}")
    print(f"   Output activation: {'Softmax' if output_dim > 1 else 'Sigmoid'}")
    
    return model


def load_test_data(filepath="data_valid.csv"):
    """Load test/validation data for prediction."""
    df = pd.read_csv(filepath)
    X = df.drop(columns=["diagnosis"]).values
    Y = df["diagnosis"].values.reshape(-1, 1)
    return X, Y


def make_predictions(model, X, threshold=0.4):  # Changed default to 0.4
    """Make predictions using the trained model."""
    # Get probabilities
    predictions = model.forward(X)
    
    if predictions.shape[1] == 2:
        # Si output est softmax à 2 classes, prendre la probabilité de la classe 1 (maligne)
        malignant_probs = predictions[:, 1:2]  # La colonne 1 correspond à la classe maligne
        predicted_classes = (malignant_probs > threshold).astype(int)
        return malignant_probs, predicted_classes
    else:
        # Cas avec sigmoid (compatibilité avec ancien code)
        predicted_classes = (predictions > threshold).astype(int)
        return predictions, predicted_classes


def evaluate_model_detailed(Y_true, predictions, predicted_classes):
    """Detailed evaluation with medical metrics."""
    accuracy = accuracy_score(Y_true.flatten(), predicted_classes.flatten())
    loss = log_loss(Y_true.flatten(), predictions.flatten())
    
    # Confusion matrix
    cm = confusion_matrix(Y_true.flatten(), predicted_classes.flatten())
    tn, fp, fn, tp = cm.ravel()
    
    # Medical metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0    # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0          # Negative Predictive Value
    
    return {
        'accuracy': accuracy,
        'loss': loss,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'npv': npv,
        'confusion_matrix': cm,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }


def display_predictions(predictions, predicted_classes, Y_true, num_examples=50):
    """Display sample predictions for visual inspection."""
    print(f"\nFirst {num_examples} predictions:")
    print("-" * 70)
    print("Note: 0 = Benign (non-cancerous), 1 = Malignant (cancerous)")
    print("-" * 70)
    
    correct = 0
    false_negatives = 0
    false_positives = 0
    
    for i in range(min(num_examples, len(predictions))):
        # Adapter en fonction de la forme des prédictions (cas softmax vs sigmoid)
        if len(predictions.shape) > 1 and predictions.shape[1] == 1:
            # Cas sigmoid (une seule valeur)
            proba = predictions[i, 0]  
        else:
            # Cas softmax (2 valeurs) - afficher la probabilité de la classe prédite
            class_idx = 1 if predicted_classes[i, 0] == 1 else 0
            proba = predictions[i, class_idx]
        
        pred = predicted_classes[i, 0]
        true = Y_true[i, 0]
        
        # Calculate metrics
        if pred == true:
            status = "✅"
            correct += 1
        else:
            status = "❌"
            if true == 1 and pred == 0:  # Missed cancer
                false_negatives += 1
                status = "❌🚨"  # Critical error
            elif true == 0 and pred == 1:  # False alarm
                false_positives += 1
                status = "❌⚠️"   # Less critical
        
        # Add descriptive labels
        pred_label = "Malignant" if pred == 1 else "Benign"
        true_label = "Malignant" if true == 1 else "Benign"
        
        print(f"Example {i+1:2d}: prob = {proba:.3f} → predicted = {pred} ({pred_label:9s}) | actual = {true} ({true_label:9s}) {status}")
    
    print("-" * 70)
    print(f"Summary of first {num_examples} predictions:")
    print(f"✅ Correct: {correct}/{num_examples} ({correct/num_examples*100:.1f}%)")
    print(f"❌🚨 Missed Cancer (False Negatives): {false_negatives}")
    print(f"❌⚠️  False Alarms (False Positives): {false_positives}")


def predict_and_evaluate_multiple_thresholds(model_filepath="model_weights.npy", 
                                           data_filepath="data_valid.csv",
                                           thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
                                           display_samples=50):
    """Test multiple thresholds to find optimal one."""
    print("Loading model...")
    weights, biases = load_model(model_filepath)
    model = create_model_from_weights(weights, biases)
    
    print("Loading test data...")
    X, Y = load_test_data(data_filepath)
    
    print("Testing multiple thresholds...")
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION ANALYSIS")
    print("="*80)
    
    best_threshold = 0.5
    best_score = 0
    
    for threshold in thresholds:
        predictions, predicted_classes = make_predictions(model, X, threshold)
        metrics = evaluate_model_detailed(Y, predictions, predicted_classes)
        
        # Custom score favoring sensitivity (important for cancer detection)
        score = 0.7 * metrics['sensitivity'] + 0.3 * metrics['specificity']
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
        
        print(f"\n📊 THRESHOLD = {threshold}")
        print(f"   Accuracy:    {metrics['accuracy']*100:5.1f}%")
        print(f"   Sensitivity: {metrics['sensitivity']*100:5.1f}% (Cancer Detection Rate)")
        print(f"   Specificity: {metrics['specificity']*100:5.1f}% (Healthy Correct Rate)")
        print(f"   False Neg:   {metrics['fn']:2d} (Missed Cancers) {'🚨' if metrics['fn'] > 5 else '✅'}")
        print(f"   False Pos:   {metrics['fp']:2d} (False Alarms)")
        print(f"   Score:       {score:.3f}")
    
    print(f"\n🎯 RECOMMENDED THRESHOLD: {best_threshold}")
    print("="*80)
    
    # Use best threshold for detailed analysis
    predictions, predicted_classes = make_predictions(model, X, best_threshold)
    metrics = evaluate_model_detailed(Y, predictions, predicted_classes)
    
    display_predictions(predictions, predicted_classes, Y, display_samples)
    
    return {
        'model': model,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'best_threshold': best_threshold,
        'metrics': metrics
    }


def predict_and_evaluate(model_filepath="model_weights.npy", 
                        data_filepath="data_valid.csv",
                        threshold=0.4,  # Lowered for better cancer detection
                        display_samples=50):
    """Main prediction function with improved cancer detection."""
    print("Loading model...")
    weights, biases = load_model(model_filepath)
    model = create_model_from_weights(weights, biases)
    
    print("Loading test data...")
    X, Y = load_test_data(data_filepath)
    
    print("Making predictions...")
    predictions, predicted_classes = make_predictions(model, X, threshold)
    
    print("Evaluating model...")
    metrics = evaluate_model_detailed(Y, predictions, predicted_classes)
    
    # Display results
    print("\n" + "="*60)
    print("🏥 MEDICAL AI EVALUATION RESULTS")
    print("="*60)
    print(f"✅ Overall Accuracy: {metrics['accuracy'] * 100:.1f}%")
    print(f"🔍 Cross-entropy Loss: {metrics['loss']:.4f}")
    print(f"🎯 Threshold Used: {threshold}")
    print("-" * 60)
    print("🏥 MEDICAL METRICS:")
    print(f"   🩺 Sensitivity (Cancer Detection): {metrics['sensitivity']*100:5.1f}%")
    print(f"   🛡️  Specificity (Healthy Detection): {metrics['specificity']*100:5.1f}%")
    print(f"   📊 Precision (Positive Predictive): {metrics['precision']*100:5.1f}%")
    print(f"   📋 NPV (Negative Predictive):      {metrics['npv']*100:5.1f}%")
    print("-" * 60)
    print("🚨 CRITICAL ERRORS:")
    print(f"   False Negatives (Missed Cancer): {metrics['fn']} {'🚨 HIGH RISK' if metrics['fn'] > 5 else '✅ Acceptable'}")
    print(f"   False Positives (False Alarms):  {metrics['fp']}")
    
    if display_samples > 0:
        display_predictions(predictions, predicted_classes, Y, display_samples)
    
    if metrics['sensitivity'] < 0.8:
        print("⚠️  Low sensitivity! Consider:")
        print("   - Lower threshold (try 0.3)")
        print("   - Retrain with more epochs")
        print("   - Adjust class weights in training")
    else:
        print("✅ Good sensitivity for cancer detection")
    
    if metrics['fn'] > 5:
        print("🚨 Too many missed cancers - CRITICAL!")
        print("   - Immediately lower threshold")
    
    return {
        'model': model,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'metrics': metrics
    }


def predict_single_sample(model, sample, threshold=0.4):
    """Predict a single sample."""
    if sample.ndim == 1:
        sample = sample.reshape(1, -1)
    
    prediction = model.forward(sample)
    predicted_class = (prediction > threshold).astype(int)
    
    return prediction[0, 0], predicted_class[0, 0]


def main():
    print("🏥 CANCER DETECTION AI - PREDICTION SYSTEM")
    print("="*50)
    print("Choose analysis type:")
    print("1. Standard evaluation (threshold 0.4)")
    print("2. Threshold optimization")
    
    choice = input("Enter choice (1-2): ").strip()
    
    if choice == "2":
        results = predict_and_evaluate_multiple_thresholds()
    else:
        results = predict_and_evaluate()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, sig_handler)  # Handle Ctrl+C gracefully
    main()