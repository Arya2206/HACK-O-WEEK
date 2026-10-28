"""Activation functions and their derivatives (NumPy)."""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def relu(z):
    return np.maximum(0.0, z)


def relu_grad(z):
    return (z > 0).astype(float)


def tanh(z):
    return np.tanh(z)


def tanh_grad(z):
    return 1.0 - np.tanh(z) ** 2


def step(z):
    """Classic perceptron step function (not differentiable)."""
    return (z >= 0).astype(float)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# name -> (function, derivative)
ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_grad),
    "relu": (relu, relu_grad),
    "tanh": (tanh, tanh_grad),
}
