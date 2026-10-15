"""Task 2: analytical gradients, autograd, and finite differences."""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch

from ._shared import _save


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


def run(output: Path, fast: bool = False, show: bool = False) -> dict:
    grad_tf, grad_pt = framework_gradients()
    expected = analytical_gradient(0.5, -1.0)
    np.testing.assert_allclose(grad_tf, expected, atol=1e-6)
    np.testing.assert_allclose(grad_pt, expected, atol=1e-6)
    steps, errors, error = gradient_check()
    assert error < 1e-7, f"Gradient check failed: {error}"
    plt.figure(figsize=(7, 4))
    plt.loglog(steps, np.maximum(errors, 1e-17), "o-")
    plt.xlabel("Finite-difference step")
    plt.ylabel("Relative gradient error")
    plt.title("Central finite differences in float64")
    plt.grid(alpha=0.3)
    _save(output, "gradient_check", show)

    xs = np.linspace(-6, 6, 400, dtype=np.float32)
    x_tf = tf.constant(xs)
    with tf.GradientTape() as tape:
        tape.watch(x_tf)
        loss = tf.reduce_sum(x_tf**2 * tf.sin(x_tf))
    derivative_tf = tape.gradient(loss, x_tf).numpy()
    x_pt = torch.tensor(xs, requires_grad=True)
    (x_pt**2 * torch.sin(x_pt)).sum().backward()
    derivative = 2 * xs * np.sin(xs) + xs**2 * np.cos(xs)
    plt.figure(figsize=(8, 4))
    plt.plot(xs, xs**2 * np.sin(xs), label="f(x)")
    plt.plot(xs, derivative, lw=4, alpha=0.4, label="Analytical derivative")
    plt.plot(xs, derivative_tf, "--", label="TensorFlow")
    plt.plot(xs, x_pt.grad.numpy(), ":", label="PyTorch")
    plt.xlabel("x")
    plt.legend()
    _save(output, "autograd", show)
    accumulated = []
    scalar = torch.tensor(3.0, requires_grad=True)
    for _ in range(3):
        scalar.square().backward()
        accumulated.append(scalar.grad.item())
    scalar.grad.zero_()
    np.testing.assert_allclose(accumulated, [6, 12, 18])
    scalar_tf = tf.Variable(1.5)
    with tf.GradientTape() as outer:
        with tf.GradientTape() as inner:
            value = tf.sin(scalar_tf) * scalar_tf
        first = inner.gradient(value, scalar_tf)
    second = outer.gradient(first, scalar_tf)
    np.testing.assert_allclose(second.numpy(), 2 * np.cos(1.5) - 1.5 * np.sin(1.5), atol=1e-6)
    return {"analytical": expected.tolist(), "tensorflow": grad_tf.tolist(), "pytorch": grad_pt.tolist(), "finite_difference_relative_error": error, "accumulated_gradients": accumulated, "gradient_after_reset": scalar.grad.item(), "tensorflow_second_derivative": float(second.numpy())}
