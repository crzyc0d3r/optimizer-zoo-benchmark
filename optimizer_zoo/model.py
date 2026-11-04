"""A tiny two-layer MLP classifier with manual backprop, written in NumPy.

The model is deliberately small so that every optimizer in this repo can be
benchmarked in seconds on a CPU. It exposes exactly what an optimizer needs:

* ``params``  -- an ordered dict of named NumPy arrays (2-D weights, 1-D biases)
* ``loss_and_grads`` -- cross-entropy loss plus gradients for a minibatch
* ``sampled_label_grads`` -- gradients with labels *sampled from the model's own
  predictive distribution*, which is the Gauss-Newton-Bartlett estimator that
  Sophia uses for its diagonal Hessian estimate.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np

Params = Dict[str, np.ndarray]


def make_dataset(n: int = 4096, d_in: int = 24, n_classes: int = 6, seed: int = 0,
                 hidden: int = 48, noise: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic classification data produced by a random *teacher* MLP.

    Labels are the argmax of a random nonlinear teacher plus Gaussian label
    noise, so the task is learnable but not trivially linearly separable.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d_in)).astype(np.float32)
    w1 = rng.standard_normal((d_in, hidden)) / np.sqrt(d_in)
    w2 = rng.standard_normal((hidden, n_classes)) / np.sqrt(hidden)
    logits = np.tanh(x @ w1) @ w2 + noise * rng.standard_normal((n, n_classes))
    y = np.argmax(logits, axis=1)
    return x, y


class MLP:
    """``x -> relu(x W1 + b1) -> W2 + b2`` with softmax cross-entropy."""

    def __init__(self, d_in: int, hidden: int, n_classes: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.params: Params = OrderedDict(
            W1=(rng.standard_normal((d_in, hidden)) * np.sqrt(2.0 / d_in)).astype(np.float32),
            b1=np.zeros(hidden, dtype=np.float32),
            W2=(rng.standard_normal((hidden, n_classes)) * np.sqrt(1.0 / hidden)).astype(np.float32),
            b2=np.zeros(n_classes, dtype=np.float32),
        )

    # -- forward -----------------------------------------------------------
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = self.params
        h_pre = x @ p["W1"] + p["b1"]
        h = np.maximum(h_pre, 0.0)
        logits = h @ p["W2"] + p["b2"]
        return logits, h

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        logits, _ = self.forward(x)
        probs = self.softmax(logits)
        return float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        logits, _ = self.forward(x)
        return float((logits.argmax(axis=1) == y).mean())

    # -- backward ----------------------------------------------------------
    def _backward(self, x: np.ndarray, h: np.ndarray, dlogits: np.ndarray) -> Params:
        p = self.params
        grads: Params = OrderedDict()
        grads["W2"] = h.T @ dlogits
        grads["b2"] = dlogits.sum(axis=0)
        dh = dlogits @ p["W2"].T
        dh[h <= 0.0] = 0.0
        grads["W1"] = x.T @ dh
        grads["b1"] = dh.sum(axis=0)
        return grads

    def loss_and_grads(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, Params]:
        logits, h = self.forward(x)
        probs = self.softmax(logits)
        n = len(y)
        loss = float(-np.log(probs[np.arange(n), y] + 1e-12).mean())
        dlogits = probs.copy()
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n
        return loss, self._backward(x, h, dlogits)

    def sampled_label_grads(self, x: np.ndarray, rng: np.random.Generator) -> Params:
        """Gradient of the loss against labels drawn from the model's own
        softmax. Used by Sophia's Gauss-Newton-Bartlett Hessian estimator:
        ``h_hat = B * g_hat * g_hat`` where ``B`` is the batch size."""
        logits, h = self.forward(x)
        probs = self.softmax(logits)
        n = len(x)
        cum = probs.cumsum(axis=1)
        u = rng.random((n, 1))
        y_s = (u > cum).sum(axis=1)
        y_s = np.minimum(y_s, probs.shape[1] - 1)
        dlogits = probs.copy()
        dlogits[np.arange(n), y_s] -= 1.0
        dlogits /= n
        return self._backward(x, h, dlogits)


def n_params(params: Params) -> int:
    return int(sum(v.size for v in params.values()))


def param_bytes(params: Params) -> int:
    return int(sum(v.nbytes for v in params.values()))
