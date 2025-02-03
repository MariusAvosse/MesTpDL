import numpy as np
import time
import pickle
import matplotlib.pyplot as plt

np.random.seed(42)


def load_data(filename):
    with open(filename, "rb") as f:
        X, y = pickle.load(f)
    return X, y


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class ReLU:
    def forward(self, x):
        # TODO
        pass

    def backward(self, grad_out):
        # TODO
        pass

    def step(self, lr):
        pass


class Linear:
    def __init__(self, n_in, n_out):
        pass

    def forward(self, x):
        # TODO
        return None

    def backward(self, grad_out):
        self.grad_W = None  # TODO
        self.grad_b = None  # TODO
        return None  # TODO

    def step(self, lr):
        # TODO
        pass


class Dropout:
    def __init__(self, dropout):
        self.dropout = dropout
        self.inference = False

    def forward(self, x):
        if self.inference:
            return x
        else:
            # TODO
            pass

    def backward(self, grad_out):
        # TODO
        return None

    def step(self, lr):
        pass


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, out_grad):
        # TODO
        return None

    def step(self, lr):
        for layer in self.layers:
            layer.step(lr)

    def inference_mode(self, setting):
        for layer in self.layers:
            if hasattr(layer, "inference"):
                layer.inference = setting


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
        # TODO
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
        # TODO
        pass


if __name__ == "__main__":
    X, y = load_data("tp4_data.pkl")

    test_ratio, val_ratio = 0.1, 0.2
    Xtest, ytest = None, None # TODO
    Xval, yval = None, None # TODO
    Xtrain, ytrain = None, None # TODO

    net = MLPClassifier(in_features=10, num_classes=4, hidden_features=None, dropout=None, lr=None) # TODO

    # training loop
    num_epochs = None # TODO
    for e in range(num_epochs):
        # TODO
        pass
    print(f"Result on test set: {validate(net,Xtest,ytest)}")


def plot_decision_boundaries(X, y, model, ax=None, title=None):
    if ax is None:
        ax = plt.gca()
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros((xx.ravel().shape[0], 8))])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    ax.scatter(X[:, 0], X[:, 1], c=y, s=10, cmap="viridis", edgecolor="k")
    if title:
        ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
