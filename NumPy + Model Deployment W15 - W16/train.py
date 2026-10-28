"""Train the from-scratch NumPy network, compare SGD vs Adam, gradient-check
the backprop, and save the trained model to disk.

Run:  python train.py
"""
import pickle

import joblib
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from nn.losses import LOSSES
from nn.network import NeuralNetwork


def gradient_check(seed=0):
    """Compare backprop gradients with numerical (finite-difference) gradients."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(5, 3))
    y = rng.integers(0, 2, size=(5, 1)).astype(float)
    net = NeuralNetwork([3, 4, 1], hidden_activation="tanh", loss="bce", seed=seed)
    loss_fn, _ = LOSSES["bce"]

    net.forward(X)
    dW, _ = net.backward(y)

    eps, max_err = 1e-6, 0.0
    for layer, W in enumerate(net.W):
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                orig = W[i, j]
                W[i, j] = orig + eps
                plus = loss_fn(y, net.forward(X))
                W[i, j] = orig - eps
                minus = loss_fn(y, net.forward(X))
                W[i, j] = orig
                num = (plus - minus) / (2 * eps)
                max_err = max(max_err, abs(num - dW[layer][i, j]))
    return max_err


if __name__ == "__main__":
    # 1) Verify backprop is correct
    err = gradient_check()
    print(f"Gradient check max abs error: {err:.2e}  ({'OK' if err < 1e-7 else 'FAIL'})\n")

    # 2) Data: two interleaving half-circles (not linearly separable)
    X, y = make_moons(n_samples=600, noise=0.2, random_state=0)
    y = y.reshape(-1, 1).astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

    # 3) Compare optimizers
    results = {}
    for opt, lr in [("sgd", 0.5), ("adam", 0.02)]:
        print(f"=== Training with {opt.upper()} (lr={lr}) ===")
        net = NeuralNetwork([2, 16, 16, 1], hidden_activation="tanh", loss="bce", seed=1)
        hist = net.fit(X_tr, y_tr, epochs=300, optimizer=opt, lr=lr, batch_size=32)
        test_acc = (net.predict(X_te) == y_te).mean()
        results[opt] = (net, hist, test_acc)
        print(f"-> {opt.upper()} test accuracy: {test_acc:.3f}\n")

    # 4) Save the best model (pickle + joblib)
    best = max(results, key=lambda k: results[k][2])
    model = results[best][0]
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    joblib.dump(model, "model.joblib")
    print(f"Saved best model ({best.upper()}) to model.pkl and model.joblib")

    # 5) Optional loss-curve plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for opt, (_, hist, _) in results.items():
            plt.plot(hist, label=opt.upper())
        plt.xlabel("epoch")
        plt.ylabel("binary cross-entropy")
        plt.title("SGD vs Adam")
        plt.legend()
        plt.savefig("loss_curves.png", dpi=120, bbox_inches="tight")
        print("Saved loss_curves.png")
    except ImportError:
        print("matplotlib not installed - skipping plot")
