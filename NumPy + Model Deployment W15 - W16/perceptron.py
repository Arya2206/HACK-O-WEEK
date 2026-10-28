"""Single perceptron (step activation, perceptron learning rule).

Learns AND and OR (linearly separable) and fails on XOR -- which is exactly
why we need hidden layers and backpropagation.
"""
import numpy as np

from nn.activations import step


class Perceptron:
    def __init__(self, n_inputs, lr=0.1):
        self.w = np.zeros(n_inputs)
        self.b = 0.0
        self.lr = lr

    def predict(self, X):
        return step(X @ self.w + self.b)

    def fit(self, X, y, epochs=50):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                error = yi - self.predict(xi[None, :])[0]
                self.w += self.lr * error * xi
                self.b += self.lr * error


if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    gates = {"AND": [0, 0, 0, 1], "OR": [0, 1, 1, 1], "XOR": [0, 1, 1, 0]}
    for name, y in gates.items():
        y = np.array(y, dtype=float)
        p = Perceptron(2)
        p.fit(X, y)
        acc = (p.predict(X) == y).mean()
        print(f"{name}: accuracy = {acc:.2f}  (weights={p.w}, bias={p.b:.2f})")
