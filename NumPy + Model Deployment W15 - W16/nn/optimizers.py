"""Optimizers: SGD (with optional momentum) and Adam.

Each optimizer updates a list of parameter arrays *in place*,
given a matching list of gradient arrays.
"""
import numpy as np


class SGD:
    def __init__(self, lr=0.1, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.velocity):
            v *= self.momentum
            v -= self.lr * g
            p += v


class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = None  # 1st moment (mean of gradients)
        self.v = None  # 2nd moment (mean of squared gradients)
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g ** 2
            m_hat = m / (1 - self.beta1 ** self.t)  # bias correction
            v_hat = v / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
