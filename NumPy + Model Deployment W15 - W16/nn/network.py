"""A fully-connected neural network built from scratch with NumPy.

Covers: forward propagation, loss calculation, backpropagation,
and weight updates via an optimizer (SGD / Adam).
"""
import numpy as np

from .activations import ACTIVATIONS, sigmoid
from .losses import LOSSES
from .optimizers import SGD, Adam


class NeuralNetwork:
    def __init__(self, layer_sizes, hidden_activation="tanh", loss="bce", seed=42):
        """layer_sizes e.g. [2, 8, 1]: 2 inputs, one hidden layer of 8, 1 output."""
        self.layer_sizes = list(layer_sizes)
        self.hidden_activation = hidden_activation
        self.loss_name = loss
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Xavier/Glorot-style init keeps activations well scaled
            self.W.append(rng.normal(0, np.sqrt(1.0 / n_in), size=(n_in, n_out)))
            self.b.append(np.zeros((1, n_out)))

    # ---------- forward ----------
    def forward(self, X):
        act, _ = ACTIVATIONS[self.hidden_activation]
        self._cache_a = [X]  # layer inputs / activations
        self._cache_z = []   # pre-activations
        a = X
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            a = sigmoid(z) if i == last else act(z)  # sigmoid output -> probability
            self._cache_z.append(z)
            self._cache_a.append(a)
        return a

    # ---------- backward ----------
    def backward(self, y_true):
        _, act_grad = ACTIVATIONS[self.hidden_activation]
        _, loss_grad = LOSSES[self.loss_name]
        y_pred = self._cache_a[-1]
        last = len(self.W) - 1

        # dL/da at the output, then chain rule through sigmoid
        delta = loss_grad(y_true, y_pred)
        s = y_pred
        delta = delta * s * (1 - s)

        dW, db = [None] * len(self.W), [None] * len(self.W)
        for i in range(last, -1, -1):
            dW[i] = self._cache_a[i].T @ delta
            db[i] = delta.sum(axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.W[i].T) * act_grad(self._cache_z[i - 1])
        return dW, db

    # ---------- training ----------
    def fit(self, X, y, epochs=1000, optimizer="adam", lr=0.01,
            batch_size=None, verbose=True):
        loss_fn, _ = LOSSES[self.loss_name]
        opt = Adam(lr=lr) if optimizer == "adam" else SGD(lr=lr)
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        batch_size = batch_size or n
        history = []
        for epoch in range(1, epochs + 1):
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                batch = idx[start:start + batch_size]
                self.forward(X[batch])
                dW, db = self.backward(y[batch])
                opt.step(self.W + self.b, dW + db)  # in-place update
            loss = loss_fn(y, self.forward(X))
            history.append(loss)
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
                acc = (self.predict(X) == y).mean()
                print(f"epoch {epoch:5d}  loss {loss:.4f}  acc {acc:.3f}")
        return history

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(float)
