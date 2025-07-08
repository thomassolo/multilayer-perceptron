import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from network import NeuralNetwork, sigmoid, sigmoid_derivative, reLU, reLU_derivative, softmax
from utils import sig_handler
import signal 

def log_loss_custom(y_true, y_pred, eps=1e-15):
    """Custom implementation of log loss compatible with softmax and sigmoid."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Si la sortie est softmax (2 classes ou plus)
    if y_pred.shape[1] > 1:
        # Convertir y_true en format one-hot pour softmax
        if y_true.shape[1] == 1:
            batch_size = y_true.shape[0]
            y_one_hot = np.zeros((batch_size, y_pred.shape[1]))
            for i in range(batch_size):
                y_one_hot[i, int(y_true[i, 0])] = 1  # Accès correct à y_true[i, 0]
            
            # Calculer la cross-entropy pour le format softmax
            return -np.mean(np.sum(y_one_hot * np.log(y_pred), axis=1))
    
    # Pour le format sigmoid (compatibilité avec l'ancien code)
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()
    
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


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
            a = softmax(z)  # Sigmoid for output layer
        activations.append(a)
    
    return activations, zs


def backward_pass_weighted(model, activations, y, learning_rate, class_weight=2.0):
    """Backward pass with class weighting to handle imbalanced data."""
    y_hat = activations[-1]
    
    # Apply class weighting - give more importance to malignant cases
    weight = np.where(y == 1, class_weight, 1.0).reshape(-1)  # Assure un vecteur 1D
    
    # Pour softmax à 2 classes, créer un vecteur one-hot pour y
    if y_hat.shape[1] == 2:
        # Convertir y en format one-hot pour correspondre à softmax
        batch_size = y.shape[0]
        y_one_hot = np.zeros((batch_size, y_hat.shape[1]))
        for i in range(batch_size):
            y_one_hot[i, int(y[i, 0])] = 1  # Accès correct à y[i, 0] au lieu de y[i]
        
        # Calculer l'erreur pondérée (forme correcte pour le broadcasting)
        error = (y_hat - y_one_hot)
        # Appliquer le poids séparément pour éviter les problèmes de broadcast
        for i in range(batch_size):
            error[i] *= weight[i]
    else:
        # Cas avec sigmoid (compatible avec ancien code)
        error = (y_hat - y) * weight.reshape(-1, 1)
    
    # Loop through layers backwards
    for j in reversed(range(len(model.weights))):
        activation = activations[j+1]
        
        if j == len(model.weights) - 1:
            # Pour softmax, l'erreur est déjà calculée correctement
            delta = error  # Pas besoin de multiplier par une dérivée pour softmax
        else:
            delta = error * reLU_derivative(activation)
        
        # Calculate gradients
        dw = np.dot(activations[j].T, delta)
        db = np.sum(delta, axis=0, keepdims=True)
        
        # Store error for next layer
        error = np.dot(delta, model.weights[j].T)
        
        # Update weights and biases
        model.weights[j] -= learning_rate * dw
        model.biases[j] -= learning_rate * db


def train_epoch_improved(model, X_train, y_train, learning_rate, class_weight=2.0):
    """Train epoch with class weighting for better cancer detection."""
    indices = np.random.permutation(X_train.shape[0])
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    epoch_loss = 0

    for i in range(X_shuffled.shape[0]):
        x = X_shuffled[i:i+1]
        y = y_shuffled[i:i+1]

        # Forward pass
        activations, _ = forward_pass(model, x)
        
        # Calculate weighted loss
        y_hat = activations[-1]
        weight = class_weight if y[0, 0] == 1 else 1.0
        loss = weight * log_loss_custom(y, y_hat)
        epoch_loss += loss

        # Backward pass with weighting
        backward_pass_weighted(model, activations, y, learning_rate, class_weight)
    
    # Calculate final training predictions and accuracy
    y_pred_train = model.forward(X_train)
    
    # Calculate accuracy
    if y_pred_train.shape[1] > 1:  # Softmax output
        y_pred_classes = np.argmax(y_pred_train, axis=1)
        y_true_classes = y_train.flatten().astype(int)
    else:  # Sigmoid output
        y_pred_classes = (y_pred_train > 0.5).flatten().astype(int)
        y_true_classes = y_train.flatten().astype(int)
    
    train_accuracy = np.mean(y_pred_classes == y_true_classes)

    return epoch_loss / X_train.shape[0], train_accuracy


def validate_model(model, X_valid, y_valid):
    """Validate the model and return validation loss and accuracy."""
    y_pred_valid = model.forward(X_valid)
    val_loss = log_loss_custom(y_valid, y_pred_valid)
    
    # Calculate accuracy
    if y_pred_valid.shape[1] > 1:  # Softmax output (multiple classes)
        y_pred_classes = np.argmax(y_pred_valid, axis=1)
        y_true_classes = y_valid.flatten().astype(int)
    else:  # Sigmoid output (binary)
        y_pred_classes = (y_pred_valid > 0.5).flatten().astype(int)
        y_true_classes = y_valid.flatten().astype(int)
    
    accuracy = np.mean(y_pred_classes == y_true_classes)
    return val_loss, accuracy

class EarlyStopping:
    """
    Classe pour gérer l'early stopping de manière modulaire.
    Peut être activée ou désactivée facilement.
    """
    
    def __init__(self, patience=5, min_improvement=5e-3, min_epochs=20, 
                 performance_threshold=0.25, excellence_threshold=0.15, 
                 max_tiny_improvements=10, absolute_max_epochs=80):
        """
        Initialise l'early stopping avec différents critères.
        
        Args:
            patience: Nombre d'époques sans amélioration significative
            min_improvement: Amélioration minimum pour compter comme significative
            min_epochs: Minimum d'époques avant d'autoriser l'arrêt
            performance_threshold: Seuil de performance "assez bonne"
            excellence_threshold: Seuil de performance "excellente"
            max_tiny_improvements: Max d'améliorations insignifiantes consécutives
            absolute_max_epochs: Limite absolue d'époques
        """
        self.patience = patience
        self.min_improvement = min_improvement
        self.min_epochs = min_epochs
        self.performance_threshold = performance_threshold
        self.excellence_threshold = excellence_threshold
        self.max_tiny_improvements = max_tiny_improvements
        self.absolute_max_epochs = absolute_max_epochs
        
        # Variables d'état
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.tiny_improvements = 0
        self.should_stop = False
        self.stop_reason = ""
    
    def check_stopping(self, epoch, val_loss, model, verbose=True):
        """
        Vérifie si l'entraînement doit s'arrêter.
        
        Args:
            epoch: Époque actuelle (0-indexé)
            val_loss: Loss de validation actuelle
            verbose: Afficher les détails
            
        Returns:
            bool: True si l'entraînement doit s'arrêter
        """
        improvement = self.best_val_loss - val_loss
        
        if verbose:
            print(f"    best_val_loss: {self.best_val_loss:.6f} - patience_counter: {self.patience_counter}/{self.patience} - tiny_improvements: {self.tiny_improvements}")
        
        # Test 1: Amélioration significative
        if val_loss < self.best_val_loss and improvement > self.min_improvement:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.tiny_improvements = 0
            # Sauvegarder le meilleur modèle
            np.save("best_model_weights.npy", {"weights": model.weights, "biases": model.biases})
            if verbose:
                print(f"    ✅ Significant improvement! Patience reset.")
            return False
        
        # Test 2: Amélioration insignifiante
        elif val_loss < self.best_val_loss and improvement > 0:
            self.best_val_loss = val_loss
            self.tiny_improvements += 1
            self.patience_counter += 1
            if verbose:
                print(f"    ⚠️  Tiny improvement ({improvement:.8f}) - patience: {self.patience_counter}/{self.patience}")
            
            # Arrêt si trop d'améliorations insignifiantes
            if self.tiny_improvements >= self.max_tiny_improvements:
                self.should_stop = True
                self.stop_reason = f"TOO MANY TINY IMPROVEMENTS ({self.tiny_improvements})"
                return True
        
        # Test 3: Pas d'amélioration ou dégradation
        else:
            self.patience_counter += 1
            if verbose:
                print(f"    ❌ No improvement or worse - patience: {self.patience_counter}/{self.patience}")
        
        # Test 4: Patience dépassée
        if self.patience_counter >= self.patience:
            self.should_stop = True
            self.stop_reason = f"PATIENCE EXCEEDED ({self.patience} epochs)"
            return True
        
        # Test 5: Performance excellente atteinte
        if self.best_val_loss < self.excellence_threshold:
            self.should_stop = True
            self.stop_reason = f"EXCELLENT MODEL ACHIEVED (loss: {self.best_val_loss:.6f} < {self.excellence_threshold})"
            return True
        
        # Test 6: Performance assez bonne après minimum d'époques
        if epoch >= self.min_epochs and self.best_val_loss < self.performance_threshold:
            self.should_stop = True
            self.stop_reason = f"GOOD MODEL ACHIEVED (loss: {self.best_val_loss:.6f} < {self.performance_threshold})"
            return True
        
        # Test 7: Limite absolue d'époques
        if epoch >= self.absolute_max_epochs:
            self.should_stop = True
            self.stop_reason = f"ABSOLUTE MAX EPOCHS REACHED ({self.absolute_max_epochs})"
            return True
        
        # Test 8: Loss invalide
        if np.isnan(val_loss) or np.isinf(val_loss):
            self.should_stop = True
            self.stop_reason = f"INVALID LOSS (NaN/Inf): {val_loss}"
            return True
        
        return False
    
    def get_stop_message(self, epoch):
        """Retourne le message d'arrêt formaté."""
        return f"\n⏹️  EARLY STOPPING at epoch {epoch+1} - {self.stop_reason}\n    Best validation loss: {self.best_val_loss:.6f}"
    
    def reset(self):
        """Remet à zéro l'early stopping pour un nouvel entraînement."""
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.tiny_improvements = 0
        self.should_stop = False
        self.stop_reason = ""


def train_model_improved(input_dim=None, hidden_dims=[32, 16], output_dim=2, 
                        epochs=1000, learning_rate=0.001, class_weight=2.0, random_seed=42,
                        use_early_stopping=True, early_stopping_config=None):
    """
    Training amélioré avec early stopping optionnel.
    
    Args:
        use_early_stopping: Active/désactive l'early stopping
        early_stopping_config: Dictionnaire de configuration pour l'early stopping
    """
    # Load data
    X_train, y_train, X_valid, y_valid = load_data()
    
    if input_dim is None:
        input_dim = X_train.shape[1]
    
    # Initialize the neural network
    model = NeuralNetwork(input_dim, hidden_dims, output_dim, random_seed=random_seed)
    
    # Initialize early stopping
    early_stopper = None
    if use_early_stopping:
        if early_stopping_config is None:
            early_stopping_config = {}
        early_stopper = EarlyStopping(**early_stopping_config)
    
    # Lists to track metrics
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    
    print(f"🚀 Starting {'IMPROVED' if use_early_stopping else 'STANDARD'} training for {epochs} epochs...")
    print(f"📋 Network: {input_dim} → {hidden_dims[0]} → {hidden_dims[1]} → {output_dim}")
    print(f"⚙️  Learning rate: {learning_rate}")
    print(f"⚖️  Class weight (malignant): {class_weight}")
    print(f"🎲 Random seed: {random_seed}")
    if use_early_stopping:
        print(f"⏹️  Early stopping: ENABLED")
        print(f"   - Patience: {early_stopper.patience}")
        print(f"   - Min improvement: {early_stopper.min_improvement}")
        print(f"   - Performance threshold: {early_stopper.performance_threshold}")
    else:
        print(f"⏹️  Early stopping: DISABLED")
    print("="*70)
    
    for epoch in range(epochs):
        # Train for one epoch with class weighting
        avg_train_loss, train_acc = train_epoch_improved(model, X_train, y_train, learning_rate, class_weight)
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate_model(model, X_valid, y_valid)
        valid_losses.append(val_loss)
        valid_accuracies.append(val_acc)
        
        # Check for early stopping
        if use_early_stopping and early_stopper is not None:
            print(f"epoch {epoch+1:3d}/{epochs} - loss: {avg_train_loss:.6f} - val_loss: {val_loss:.6f} - "
                  f"acc: {train_acc:.2%} - val_acc: {val_acc:.2%} - "
                  f"improvement: {early_stopper.best_val_loss - val_loss:.8f}")
            
            if early_stopper.check_stopping(epoch, val_loss, model):
                print(early_stopper.get_stop_message(epoch))
                break
        else:
            # Sans early stopping - affichage simple
            if epoch % 10 == 0 or epoch < 10:
                print(f"epoch {epoch+1:3d}/{epochs} - loss: {avg_train_loss:.6f} - val_loss: {val_loss:.6f} - "
                      f"acc: {train_acc:.2%} - val_acc: {val_acc:.2%}")
    
    # Load best model if early stopping was used
    if use_early_stopping and early_stopper is not None:
        try:
            best_model_data = np.load("best_model_weights.npy", allow_pickle=True).item()
            model.weights = best_model_data["weights"]
            model.biases = best_model_data["biases"]
            print("✅ Best model loaded")
        except FileNotFoundError:
            print("⚠️  No best model found, using final model")
    
    # Save final model
    np.save("model_weights.npy", {"weights": model.weights, "biases": model.biases})
    
    # Plot learning curves - Loss
    plt.figure(figsize=(12, 10))
    
    # Subplot 1: Loss
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label="Train Loss", linewidth=2, color='#FF5733')
    plt.plot(valid_losses, label="Validation Loss", linewidth=2, color='#33B5FF')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Learning Curves - Loss ({'Early Stopping' if use_early_stopping else 'Full Training'})")
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(train_accuracies, label="Train Accuracy", linewidth=2, color='#28B463')
    plt.plot(valid_accuracies, label="Validation Accuracy", linewidth=2, color='#7D3C98')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.5, 1.01)  # Start from 0.5 for better visibility
    plt.legend()
    plt.title(f"Learning Curves - Accuracy ({'Early Stopping' if use_early_stopping else 'Full Training'})")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("loss-accuracy-curves.png")
    plt.show()
    
    print(f"\n✅ Training completed!")
    print(f"📊 Final metrics:")
    print(f"   - Training loss: {train_losses[-1]:.4f}")
    print(f"   - Validation loss: {valid_losses[-1]:.4f}")
    print(f"   - Training accuracy: {train_accuracies[-1]:.2%}")
    print(f"   - Validation accuracy: {valid_accuracies[-1]:.2%}")
    print("💾 Model saved as 'model_weights.npy'")
    
    return model, train_losses, valid_losses, train_accuracies, valid_accuracies


# Fonctions de convenance pour utilisation directe
def train_with_early_stopping(**kwargs):
    """Lance l'entraînement avec early stopping."""
    return train_model_improved(use_early_stopping=True, **kwargs)


def train_without_early_stopping(epochs=100, **kwargs):
    """Lance l'entraînement sans early stopping."""
    return train_model_improved(epochs=epochs, use_early_stopping=False, **kwargs)


def train_custom_early_stopping(patience=5, min_improvement=0.005, 
                               performance_threshold=0.25, excellence_threshold=0.15,
                               absolute_max_epochs=80, **kwargs):
    """Lance l'entraînement avec early stopping personnalisé."""
    config = {
        "patience": patience,
        "min_improvement": min_improvement,
        "performance_threshold": performance_threshold,
        "excellence_threshold": excellence_threshold,
        "absolute_max_epochs": absolute_max_epochs
    }
    return train_model_improved(use_early_stopping=True, 
                               early_stopping_config=config, **kwargs)

def main():
    """Fonction principale avec options pour l'early stopping."""
    signal.signal(signal.SIGINT, sig_handler)
    
    print("🏥 MULTILAYER PERCEPTRON TRAINER")
    print("=" * 50)
    print("Choisissez le mode d'entraînement :")
    print("1. Avec Early Stopping (recommandé)")
    print("2. Sans Early Stopping (entraînement complet)")
    print("3. Early Stopping personnalisé")
    print("=" * 50)
    
    choice = input("Votre choix (1-3): ").strip()
    
    # Demander à l'utilisateur la liste des hidden layers
    hidden_layers_input = input("Entrez la liste des tailles de hidden layers séparées par des virgules (ex: 64,32,16) [défaut: 32,16]: ").strip()
    if hidden_layers_input:
        hidden_dims = [int(x) for x in hidden_layers_input.split(",") if x.strip().isdigit()]
        if not hidden_dims:
            hidden_dims = [32, 16]
    else:
        hidden_dims = [32, 16]

    if choice == "1":
        # Mode avec early stopping par défaut
        print("🎯 Mode: Early Stopping activé (paramètres par défaut)")
        model, train_losses, valid_losses, train_accs, valid_accs = train_model_improved(
            use_early_stopping=True,
            hidden_dims=hidden_dims
        )
        
    elif choice == "2":
        # Mode sans early stopping
        print("⏳ Mode: Entraînement complet (sans early stopping)")
        epochs = int(input("Nombre d'époques (défaut 100): ") or "100")
        model, train_losses, valid_losses, train_accs, valid_accs = train_model_improved(
            epochs=epochs,
            use_early_stopping=False,
            hidden_dims=hidden_dims
        )
        
    elif choice == "3":
        # Mode personnalisé
        print("⚙️ Mode: Early Stopping personnalisé")
        print("\nConfiguration personnalisée:")
        
        patience = int(input("Patience (défaut 5): ") or "5")
        min_improvement = float(input("Amélioration minimum (défaut 0.005): ") or "0.005")
        performance_threshold = float(input("Seuil performance (défaut 0.25): ") or "0.25")
        excellence_threshold = float(input("Seuil excellence (défaut 0.15): ") or "0.15")
        max_epochs = int(input("Max époques (défaut 80): ") or "80")
        
        early_stopping_config = {
            "patience": patience,
            "min_improvement": min_improvement,
            "performance_threshold": performance_threshold,
            "excellence_threshold": excellence_threshold,
            "absolute_max_epochs": max_epochs
        }
        
        model, train_losses, valid_losses, train_accs, valid_accs = train_model_improved(
            use_early_stopping=True,
            early_stopping_config=early_stopping_config,
            hidden_dims=hidden_dims
        )
        
    else:
        print("❌ Choix invalide, utilisation du mode par défaut")
        model, train_losses, valid_losses, train_accs, valid_accs = train_model_improved(hidden_dims=hidden_dims)

    print("\n" + "="*70)
    print("🎉 Entraînement terminé avec succès!")
    print(f"📊 Loss finale d'entraînement: {train_losses[-1]:.6f}")
    print(f"📈 Loss finale de validation: {valid_losses[-1]:.6f}")
    print("💾 Modèle sauvé!")
    print("🔍 Testez maintenant avec: python predict.py")
    print("=" * 70)




if __name__ == "__main__":
    signal.signal(signal.SIGINT, sig_handler)  # Handle Ctrl+C gracefully
    main()