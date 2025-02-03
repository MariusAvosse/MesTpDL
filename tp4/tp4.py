import numpy as np
import pickle
import matplotlib.pyplot as plt

# Chargement des données
def load_data(filename):
    with open(filename, "rb") as f:
        X, y = pickle.load(f)
    return X, y

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

X, y = load_data("tp4_data.pkl")

# (a) Vérification des formes des données
print(f"Shape of X: {X.shape}")  
print(f"Shape of y: {y.shape}")

# (b) Nuage de points des deux premières coordonnées
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolor="k", s=10)
plt.colorbar(scatter, label="Classe")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Nuage de points des deux premières caractéristiques")
plt.show()

# (c) Division des données en train (70%), validation (20%), test (10%)
num_samples = X.shape[0]
np.random.seed(42)
indices = np.random.permutation(num_samples) 
test_size = int(0.1 * num_samples)
val_size = int(0.2 * num_samples)

test_idx = indices[:test_size]
val_idx = indices[test_size:test_size + val_size]
train_idx = indices[test_size + val_size:]

Xtest, ytest = X[test_idx], y[test_idx]
Xval, yval = X[val_idx], y[val_idx]
Xtrain, ytrain = X[train_idx], y[train_idx]

# Vérification des tailles des ensembles
print(f"Train set size: {Xtrain.shape[0]}")
print(f"Validation set size: {Xval.shape[0]}")
print(f"Test set size: {Xtest.shape[0]}")


# Couche ReLU
class ReLU:
    def __init__(self):
        self.cache = None  # Pour stocker la sortie du forward

    def forward(self, x):
        self.cache = x
        return np.maximum(0, x) 

    def backward(self, grad_out):
        grad_x = grad_out * (self.cache > 0) 
        return grad_x

    def step(self, lr):
        pass 

"""
3. Pourquoi la fonction step ne fait rien dans ReLU?
Contrairement aux couches linéaires (comme Linear), 
ReLU n’a pas de poids ni de biais. Elle applique simplement une 
transformation élémentaire max(0, x). Comme step est utilisé pour mettre à jour
les paramètres du modèle pendant l'entraînement, et que 
ReLU n’a rien à mettre à jour, la fonction reste vide.
"""

# 4. Implémentation de la fonction check_gradient
def check_gradient(layer, x, grad_out, epsilon=1e-5):
    layer.forward(x)
    grad_analytic = layer.backward(grad_out)

    # Initialisation du gradient numérique
    grad_numeric = np.zeros_like(x)

    # Calcul du gradient numérique élément par élément
    for i in range(x.size):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus.flat[i] += epsilon
        x_minus.flat[i] -= epsilon

        f_plus = layer.forward(x_plus).sum()
        f_minus = layer.forward(x_minus).sum()

        # Approximation du gradient avec la différence finie centrale
        grad_numeric.flat[i] = (f_plus - f_minus) / (2 * epsilon)

    # Calcul de la différence relative entre les gradients
    diff = np.linalg.norm(grad_numeric - grad_analytic) / (np.linalg.norm(grad_numeric) + np.linalg.norm(grad_analytic) + 1e-8)

    print(f"Différence relative entre le gradient analytique et numérique: {diff:.8f}")

    return diff < 1e-4 

# Pour tester:
relu = ReLU()
np.random.seed(42)
x_test = np.random.randn(5, 5) 
grad_out_test = np.random.randn(5, 5)
check_gradient(relu, x_test, grad_out_test)


# 5 et 6.  Initialisation des poids et biais (He/Kaiming Initialization)
class Linear:
    def __init__(self, n_in, n_out):
        np.random.seed(42)
        self.W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)  
        self.b = np.zeros(n_out) 
        self.x_cache = None 
        self.grad_W = None  
        self.grad_b = None 

    def forward(self, x):
        self.x_cache = x  
        return x @ self.W + self.b  # Multiplication matricielle + biais

    def backward(self, grad_out):
        batch_size = self.x_cache.shape[0] 

        self.grad_W = self.x_cache.T @ grad_out / batch_size  
        self.grad_b = np.mean(grad_out, axis=0) 
        grad_x = grad_out @ self.W.T 

        return grad_x

    def step(self, lr):
        self.W -= lr * self.grad_W 
        self.b -= lr * self.grad_b

# 7. Vérification du gradient avec check_gradient
def check_gradient_linear():
    np.random.seed(42)
    n_in, n_out = 5, 3 
    layer = Linear(n_in, n_out)

    x_test = np.random.randn(4, n_in)
    grad_out_test = np.random.randn(4, n_out)

    layer.forward(x_test)
    grad_analytic = layer.backward(grad_out_test)


    check_gradient(layer, x_test, grad_out_test)

# Exécuter le test
check_gradient_linear()


# 9.  Sequential 
class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x) 
        return x

    def backward(self, out_grad):
        for layer in reversed(self.layers): 
            out_grad = layer.backward(out_grad)
        return out_grad

    def step(self, lr):
        for layer in self.layers:
            layer.step(lr)
    
    def inference_mode(self, setting):
        for layer in self.layers:
            if hasattr(layer, "inference"):
                layer.inference = setting

   
# 10 et 11. Implémentation de forward et de backward dans Dropout
class Dropout:
    def __init__(self, dropout):
        self.dropout = dropout
        self.inference = False

    def forward(self, x):
        if self.inference:
            return x
        else:
            np.random.seed(42)
            mask = np.random.binomial(1, 1 - self.dropout, size=x.shape)
            self.mask = mask
            return x * mask

    def backward(self, grad_out):
        if self.inference:
            return grad_out  # En mode inférence, le gradient n'est pas modifié
        else:
            return grad_out * self.mask  # Appliquer le masque inverse pour la rétropropagation
    
    def step(self, lr):
        pass

"""
11.c. Dropout ne fait rien dans step. Pourquoi pas ?
La couche Dropout ne met pas à jour de paramètres tels que les poids ou les biais,
car elle ne possède pas de paramètres entraînables. Elle est simplement utilisée
pour régulariser le réseau en supprimant aléatoirement des neurones pendant l'entraînement.
Par conséquent, la méthode step dans Dropout n'a aucun effet et ne fait rien.


13. Que fait inference_mode dans predict ?
La fonction inference_mode dans la classe Sequential permet de changer le comportement 
de certaines couches comme Dropout pendant la phase de prédiction.

Lors de l'entraînement, la couche Dropout désactive aléatoirement certaines connexions 
pour régulariser le modèle.
Lors de l'inférence (prédiction), la couche Dropout désactive cette fonctionnalité, 
c'est-à-dire qu'aucune connexion n'est supprimée et toutes les connexions sont utilisées pour la prédiction.
"""

# 12, 14
class MLPClassifier:
    def __init__(self, in_features, hidden_features, num_classes, dropout, lr):
        self.net = Sequential(
            [
                Linear(in_features, hidden_features),
                Dropout(dropout),
                ReLU(),
                Linear(hidden_features, num_classes),
            ]
        )
        self.lr = lr
        self.num_classes = num_classes

    def forward(self, x):
        self.logits = self.net.forward(x)
        self.probs = softmax(self.logits)
        return self.probs

    def loss(self, labels):
        self.labels = labels
        correct_probs = self.probs[np.arange(len(labels)), labels]
        return np.mean(-np.log(correct_probs))

    @staticmethod
    def one_hot(labels, num_classes):
        Y = np.zeros((len(labels), num_classes))
        Y[np.arange(len(labels)), labels] = 1
        return Y


    def backward(self):
        loss_grad = self.probs - self.one_hot(self.labels, self.num_classes)
        return self.net.backward(loss_grad)

    def step(self):
        self.net.step(self.lr)

    def predict(self, data):
        self.net.inference_mode(True)
        probs = self.net.forward(data)
        self.net.inference_mode(False)
        return np.argmax(probs, axis=1)

    def validate(self, data, labels):
        preds = self.predict(data)
        accuracy = np.mean(preds == labels)
        return accuracy


# 15. Boucle d'entraînement
# Hyperparamètres
num_epochs = 120
lr = 0.01
batch_size = 32
dropout_rate = 0.1
hidden_features = 32

# Initialisation du modèle
net = MLPClassifier(in_features=10, num_classes=4, hidden_features=hidden_features, dropout=dropout_rate, lr=lr)

# Boucle d'entraînement
for epoch in range(num_epochs):
    np.random.seed(42)
    indices = np.random.permutation(len(Xtrain))
    Xtrain_shuffled = Xtrain[indices]
    ytrain_shuffled = ytrain[indices]

    total_loss = 0  
    num_batches = 0 

    # Division en mini-batchs
    for i in range(0, len(Xtrain), batch_size):
        X_batch = Xtrain_shuffled[i:i + batch_size]
        y_batch = ytrain_shuffled[i:i + batch_size]

        probs = net.forward(X_batch)

        # Calcul de la perte
        loss = net.loss(y_batch)
        total_loss += loss  
        num_batches += 1 

        net.backward()

        # Mise à jour des paramètres
        net.step()

    # Calcul de la perte moyenne
    avg_loss = total_loss / num_batches

    # Validation et affichage des résultats
    train_acc = net.validate(Xtrain, ytrain)
    val_acc = net.validate(Xval, yval)

    print(f"Époque {epoch + 1}/{num_epochs}, Perte moyenne : {avg_loss:.4f}, Précision train : {train_acc*100:.2f}%, Précision val : {val_acc*100:.2f}%")
# Test final
test_acc = net.validate(Xtest, ytest)
print(f"Précision finale sur le jeu de test : {test_acc*100:.2f}%")


