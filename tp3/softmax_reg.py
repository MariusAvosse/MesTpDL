import numpy as np
from load_fmnist import TEXT_LABELS, load_images, load_labels
import time

IN_DIM, OUT_DIM = 784, 10
MAIN = __name__ == "__main__"

# for reproducibility
np.random.seed(42)

def show(n):
    import plotly.express as px
    fig = px.imshow(train_imgs[n], binary_string=True, width=300, title=TEXT_LABELS[train_labels[n]])
    fig.show()

def softmax(o):
    o_exp = np.exp(o - np.max(o)) # for numeric stability
    return o_exp / np.sum(o_exp)

def cross_entropy_loss(p, c):
    return -np.log(p[c] + 1e-15)

def grad(x, probs, label):
    probs_copy = probs.copy()
    probs_copy[label] -= 1
    return np.outer(x.T, probs_copy), probs_copy

def initialize(sigma=0.01):
    return np.random.normal(0, sigma, (IN_DIM, OUT_DIM)), np.zeros(OUT_DIM)

def evaluate(W, b, ims, ys):
    correct = 0
    for x, y in zip(ims,ys):
        pred = np.argmax(softmax(x @ W + b))
        if pred == y:
            correct += 1
    return correct / len(ys)

def create_batches(data, labels, batch_size):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    N = len(indices)
    indices = indices[:(N // batch_size) * batch_size]
    for s in range(0, len(indices), batch_size):
        batch_indices = indices[s:s + batch_size]
        yield data[batch_indices], labels[batch_indices]


def train_sgd(lr=0.01, num_epochs=10, sigma=0.01, batch_size=64, W=None, b=None, verbose=False):
    print(f"Training softmax regression with SGD, {lr=}, {batch_size=}, {num_epochs=}, non-vectorized version")

    if W is None: W, b = initialize(sigma=sigma)

    for epoch in range(num_epochs):
        t0 = time.time()
        loss = 0
        num_batches = len(train_imgs_flat) // batch_size
        for batch_imgs, batch_labels in create_batches(train_imgs_flat, train_labels, batch_size):
            grad_W, grad_b = np.zeros_like(W), np.zeros_like(b)
            for x, label in zip(batch_imgs, batch_labels):
                probs = softmax(x @ W + b)
                loss += cross_entropy_loss(probs, label) / batch_size
                grad_W_sample, grad_b_sample = grad(x, probs, label)
                grad_W += grad_W_sample / batch_size
                grad_b += grad_b_sample / batch_size
            W -= lr * grad_W
            b -= lr * grad_b

        if verbose:
            print(f"Epoch {epoch+1}, avg loss per batch: {loss / num_batches}, time: {(time.time() - t0):3f}")
    return W, b

if MAIN:
    # load and flatten images
    train_imgs = load_images()
    train_labels = load_labels()
    val_imgs = load_images(test=True)
    val_labels = load_labels(test=True)

    assert train_imgs.shape == (60000,28,28) and train_labels.shape == (60000,) and val_imgs.shape == (10000,28,28) and val_labels.shape == (10000,)
    train_imgs_flat = train_imgs.reshape(len(train_imgs), -1)
    val_imgs_flat = val_imgs.reshape(len(val_imgs), -1)
    assert train_imgs_flat.shape == (60000, 784) and val_imgs_flat.shape == (10000, 784)
    print("Data successfully loaded.")

    print("----")
    W, b = train_sgd(verbose=True)
    val_accuracy = evaluate(W, b, val_imgs_flat, val_labels)
    print(f"Validation Accuracy: {val_accuracy:.2%}")
    assert val_accuracy > .8