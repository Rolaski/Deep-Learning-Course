"""Exercises 2.1 and 2.2."""

from collections.abc import Callable

import numpy as np
import tensorflow as tf
import torch


def analytical_gradient(x: float, y: float) -> np.ndarray:
    """Gradient of sin(x) * y**2 + exp(x*y)."""
    return np.array([
        np.cos(x) * y**2 + y * np.exp(x * y),
        2 * y * np.sin(x) + x * np.exp(x * y),
    ])


def framework_gradients(x: float = 0.5, y: float = -1.0) -> tuple[np.ndarray, np.ndarray]:
    x_tf, y_tf = tf.Variable(x, dtype=tf.float64), tf.Variable(y, dtype=tf.float64)
    with tf.GradientTape() as tape:
        value_tf = tf.sin(x_tf) * y_tf**2 + tf.exp(x_tf * y_tf)
    gradient_tf = np.array([g.numpy() for g in tape.gradient(value_tf, [x_tf, y_tf])])
    point = torch.tensor([x, y], dtype=torch.float64, requires_grad=True)
    value_pt = torch.sin(point[0]) * point[1]**2 + torch.exp(point[0] * point[1])
    value_pt.backward()
    return gradient_tf, point.grad.detach().numpy()


def numerical_gradient(function: Callable, weights: np.ndarray, step: float = 1e-5) -> np.ndarray:
    if step <= 0:
        raise ValueError("The finite-difference step must be positive.")
    weights = np.asarray(weights, dtype=np.float64)
    gradient = np.empty_like(weights)
    for index in np.ndindex(weights.shape):
        plus, minus = weights.copy(), weights.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (function(plus) - function(minus)) / (2 * step)
    return gradient


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-300))


def gradient_check() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(1)
    matrix, weights = rng.normal(size=(6, 4)), rng.normal(size=4)

    def function(w):
        return np.logaddexp.reduce(matrix @ w) + 0.5 * np.sum(w**2)

    w_pt = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
    loss = torch.logsumexp(torch.tensor(matrix) @ w_pt, dim=0) + 0.5 * w_pt.square().sum()
    loss.backward()
    reference = w_pt.grad.numpy()
    steps = np.logspace(-14, 0, 29)
    errors = np.array([relative_error(numerical_gradient(function, weights, h), reference) for h in steps])
    return steps, errors, relative_error(numerical_gradient(function, weights), reference)
