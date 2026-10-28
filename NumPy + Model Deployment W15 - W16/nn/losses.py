"""Loss functions and their gradients w.r.t. the predictions (NumPy)."""
import numpy as np

EPS = 1e-12


def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)


def mse_grad(y_true, y_pred):
    return 2.0 * (y_pred - y_true) / y_true.size


def binary_cross_entropy(y_true, y_pred):
    p = np.clip(y_pred, EPS, 1 - EPS)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def binary_cross_entropy_grad(y_true, y_pred):
    p = np.clip(y_pred, EPS, 1 - EPS)
    return (p - y_true) / (p * (1 - p)) / y_true.size


# name -> (loss, gradient)
LOSSES = {
    "mse": (mse, mse_grad),
    "bce": (binary_cross_entropy, binary_cross_entropy_grad),
}
