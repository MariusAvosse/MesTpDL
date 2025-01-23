# %% [markdown]
# # Introduction à l'apprentissage profond
# M2 informatique, Université Paris Cité, 2024-2025
# 
# Ce notebook contient de petits exercices, qui font partie du TP3. Lisez d'abord le fichier `tp3.md` ou `tp3.pdf`, puis revenez ici.
# 

# %%
import numpy as np

np.random.seed(42)

X = np.random.randint(0,10,size=(3,4))
Y = np.random.randint(0,10,size=(1,4))

# %%
print(X)
print(Y)

# %% [markdown]
# (a) Que sera `X + Y` ? Expliquez, puis vérifiez votre réponse.

# %%
Z = Y.T
Z.shape

# %% [markdown]
# (b) Peut-on calculer `Y + Z` ? Et `X + Z` ? Expliquez et vérifiez votre réponse.

# %% [markdown]
# (c) Que calcule `X.mean()` ? Expliquez la différence avec `X.mean(axis=0)` et `X.mean(axis=1)`.

# %% [markdown]
# En faisant le lien avec nos boucles d'apprentissage, nous créons maintenant des "images" et "poids" fictifs. Nous gardons les dimensions petites, juste pour faciliter l'observation des résultats.

# %%
batch_size = 4
in_dim = 6
out_dim = 3
batch_imgs = np.random.randint(-5,5,size=(batch_size, in_dim))
W = np.random.randint(-5,5,size=(in_dim, out_dim))

# %%
W, batch_imgs

# %% [markdown]
# (d) La fonction suivante effectue une multiplication vecteur-matrice pour chaque ligne de `batch_imgs` et les empile dans un tableau numpy. Réécrivez cette fonction comme une multiplication matrice-matrice unique qui calcule le même tableau numpy.

# %%
def compute_logits(batch_imgs, W):
    logits = np.zeros((batch_size,out_dim))
    for i, im in enumerate(batch_imgs):
        logits[i] = im @ W
    return logits
logits = compute_logits(batch_imgs, W)

# %%
#ANSWER
def compute_logits_batch(batch_imgs, W):
    # TODO
logits2 = compute_logits_batch(batch_imgs, W)

# %%
assert (logits == logits2).all()

# %% [markdown]
# (e) Écrivez une version vectorisée de `softmax` sur les logits. Vérifiez que le résultat est identique au softmax ligne par ligne donné ici.

# %%
def softmax(logits):
    probs = np.zeros_like(logits)
    for i, row in enumerate(logits):
        row_exp = np.exp(row - np.max(row)) # pour la stabilité numérique
        probs[i] = row_exp / np.sum(row_exp)
    return probs
probs = softmax(logits)

# %%
def softmax_batch(logits):
    # TODO
probs2 = softmax_batch(logits)

# %%
assert (probs == probs2).all()

# %% [markdown]
# Pour le calcul de la perte par entropie croisée et du gradient, la syntaxe **d'indexation** suivante sera utile :

# %%
X = np.random.randint(-5,5,size=(4,6))
print(X)
labels = [5,2,1,1]
X_sel = X[[0,1,2,3],labels]

# %% [markdown]
# (f) Prédisez la valeur de `X_sel`. Vérifiez votre prédiction.

# %% [markdown]
# Étant donné que `np.arange(4)` est `[0,1,2,3]`, on peut aussi écrire ceci comme `X_sel = X[np.arange(4),[5,2,1,1]]`

# %% [markdown]
# La même syntaxe peut être utilisée pour des affectations :

# %%
labels = [5,2,1,1]
X_copy = X.copy()
print(X_copy)
X_copy[np.arange(4), labels] -= 1

# %% [markdown]
# (g) À quoi ressemblera `X_copy` après l'exécution de la cellule ci-dessus ? Vérifiez votre prédiction.

# %% [markdown]
# (h) Enfin, un avertissement sur Python. Un tableau numpy (ou un tenseur torch) est passé aux fonctions "par référence". Cela signifie que modifier le tableau dans une fonction a un effet de bord : le tableau aura également une valeur modifiée en dehors de la fonction. Quelle est la différence entre les effets de `example` et `example2` ci-dessous ? Exécutez la dernière cellule pour démontrer la différence.

# %%
def example(v):
    v[2] -= 1
    return 2. * v

# %%
def example2(v):
    vcopy = v.copy() # en Pytorch, la fonction analogue s'appelle `clone` !
    vcopy[2] -= 1
    return 2. * vcopy

# %%
import numpy as np
v = np.array([1.,2,3,4])
v2 = np.array([1.,2,3,4])
a = example(v)
a2 = example2(v2)
print(a,v)
print(a2,v2)
