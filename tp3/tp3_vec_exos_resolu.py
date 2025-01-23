import numpy as np

# Fixer la graine pour la reproductibilité
np.random.seed(42)

# Création des matrices X et Y
X = np.random.randint(0, 10, size=(3, 4))
Y = np.random.randint(0, 10, size=(1, 4))

# Afficher X et Y
print("Matrix X:")
print(X)
print("Matrix Y:")
print(Y)

# (a) Que sera X + Y ?
print("\n(a) Résultat de X + Y (Broadcasting):")
print(X + Y)
print("Explication: Broadcasting étend Y (1x4) pour l'addition avec chaque ligne de X (3x4).")

# Création de Z comme transposé de Y
Z = Y.T
print("\nShape de Z (transposé de Y):", Z.shape)

# (b) Peut-on calculer Y + Z et X + Z ?
print("\n(b) Peut-on calculer Y + Z et X + Z ?")
try:
    print("Y + Z:")
    print(Y + Z)
except ValueError as e:
    print(f"Impossible de calculer Y + Z: {e}")
try:
    print("X + Z:")
    print(X + Z)
except ValueError as e:
    print(f"Impossible de calculer X + Z: {e}")
print("Explication: Les dimensions ne sont pas compatibles pour l'addition.")

# (c) Que calcule X.mean() ?
print("\n(c) Calculs avec X.mean():")
print("X.mean():", X.mean())
print("X.mean(axis=0):", X.mean(axis=0))
print("X.mean(axis=1):", X.mean(axis=1))
print("Explication:")
print("- X.mean() calcule la moyenne de tous les éléments.")
print("- X.mean(axis=0) calcule la moyenne colonne par colonne.")
print("- X.mean(axis=1) calcule la moyenne ligne par ligne.")

# Création de données fictives (batch_imgs et W)
batch_size = 4
in_dim = 6
out_dim = 3
batch_imgs = np.random.randint(-5, 5, size=(batch_size, in_dim))
W = np.random.randint(-5, 5, size=(in_dim, out_dim))
print("\nMatrice batch_imgs et W:")
print("batch_imgs:")
print(batch_imgs)
print("W:")
print(W)

# (d) Multiplication vecteur-matrice à transformer en matrice-matrice
def compute_logits(batch_imgs, W):
    logits = np.zeros((batch_size, out_dim))
    for i, im in enumerate(batch_imgs):
        logits[i] = im @ W
    return logits

logits = compute_logits(batch_imgs, W)
print("\n(d) Logits calculés avec boucle:")
print(logits)

# Version vectorisée
def compute_logits_batch(batch_imgs, W):
    return batch_imgs @ W

logits2 = compute_logits_batch(batch_imgs, W)
assert np.allclose(logits, logits2), "Les résultats des deux méthodes devraient être identiques."
print("\n(d) Logits calculés avec multiplication matrice-matrice (vectorisée):")
print(logits2)

# (e) Softmax ligne par ligne
def softmax(logits):
    probs = np.zeros_like(logits)
    for i, row in enumerate(logits):
        row_exp = np.exp(row - np.max(row))  # Stabilité numérique
        probs[i] = row_exp / np.sum(row_exp)
    return probs

probs = softmax(logits)
print("\n(e) Probabilités calculées avec softmax ligne par ligne:")
print(probs)

# Version vectorisée de softmax
def softmax_batch(logits):
    logits_exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Stabilité numérique
    return logits_exp / np.sum(logits_exp, axis=1, keepdims=True)

probs2 = softmax_batch(logits)
assert np.allclose(probs, probs2), "Les résultats des deux méthodes devraient être identiques."
print("\n(e) Probabilités calculées avec softmax vectorisé:")
print(probs2)

# (f) Indexation avec X
X = np.random.randint(-5, 5, size=(4, 6))
labels = [5, 2, 1, 1]
X_sel = X[np.arange(4), labels]
print("\n(f) Sélection d'éléments avec indexation:")
print("Matrice X:")
print(X)
print("Labels:", labels)
print("X_sel (éléments sélectionnés):", X_sel)

# (g) Affectation avec indexation
X_copy = X.copy()
X_copy[np.arange(4), labels] -= 1
print("\n(g) Matrice X après modification avec indexation:")
print(X_copy)

# (h) Effet de bord des fonctions Python
def example(v):
    v[2] -= 1
    return 2. * v

def example2(v):
    vcopy = v.copy()  # En Pytorch, utilisez clone() pour éviter les effets de bord
    vcopy[2] -= 1
    return 2. * vcopy

v = np.array([1., 2, 3, 4])
v2 = np.array([1., 2, 3, 4])
a = example(v)
a2 = example2(v2)

print("\n(h) Effets de bord dans les fonctions:")
print("example(v):")
print("Retour:", a, " | v après appel:", v)
print("example2(v2):")
print("Retour:", a2, " | v2 après appel:", v2)
print("Explication: example modifie directement v, tandis que example2 travaille sur une copie.")
