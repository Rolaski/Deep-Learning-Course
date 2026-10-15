"""Task 3: linear regression from scratch and in both frameworks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch

from ._shared import FitResult, _validate, _save, _curves, load_housing, mse
from .task05 import pytorch_sgd

def gradient_descent(features, target, learning_rate=0.2, epochs=500, record_path=False) -> FitResult:
    features, target = _validate(features, target, learning_rate, epochs)
    weights, bias = np.zeros(features.shape[1]), 0.0
    history = []
    path = [np.r_[weights, bias]] if record_path else None
    for _ in range(epochs):
        error = features @ weights + bias - target
        history.append(float(np.mean(error**2)))
        weights -= learning_rate * (2 / len(features)) * features.T @ error
        bias -= learning_rate * 2 * error.mean()
        if path is not None:
            path.append(np.r_[weights, bias])
    return FitResult(weights, float(bias), np.array(history), trajectory=np.array(path) if path is not None else None)


def tensorflow_descent(features, target, learning_rate=0.2, epochs=500) -> FitResult:
    features, target = _validate(features, target, learning_rate, epochs)
    with tf.device("/GPU:0"):
        x, y = tf.constant(features, tf.float32), tf.constant(target, tf.float32)
        weights, bias = tf.Variable(tf.zeros(x.shape[1])), tf.Variable(0.0)
        history = []
        for _ in range(epochs):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean((tf.linalg.matvec(x, weights) + bias - y)**2)
            dw, db = tape.gradient(loss, [weights, bias])
            history.append(float(loss.numpy()))
            weights.assign_sub(learning_rate * dw)
            bias.assign_sub(learning_rate * db)
    return FitResult(weights.numpy(), float(bias.numpy()), np.array(history))


def pytorch_descent(features, target, learning_rate=0.2, epochs=500) -> FitResult:
    features, target = _validate(features, target, learning_rate, epochs)
    x = torch.tensor(features, dtype=torch.float32, device="cuda")
    y = torch.tensor(target, dtype=torch.float32, device="cuda")
    weights = torch.zeros(x.shape[1], device="cuda", requires_grad=True)
    bias = torch.zeros((), device="cuda", requires_grad=True)
    history = []
    for _ in range(epochs):
        loss = (x @ weights + bias - y).square().mean()
        loss.backward()
        history.append(loss.item())
        with torch.no_grad():
            weights -= learning_rate * weights.grad
            bias -= learning_rate * bias.grad
            weights.grad.zero_()
            bias.grad.zero_()
    return FitResult(weights.detach().cpu().numpy(), bias.item(), np.array(history))


def run(output: Path, fast: bool = False, show: bool = False) -> dict:
    rng = np.random.default_rng(42)
    synthetic_x = rng.uniform(-3, 3, 100)
    synthetic_y = 2.5 * synthetic_x - 1 + rng.normal(size=100)
    synthetic = gradient_descent(synthetic_x[:, None], synthetic_y, 0.05, 100, record_path=True)
    plt.figure(figsize=(7, 4))
    plt.scatter(synthetic_x, synthetic_y, s=12, label="Samples")
    grid = np.linspace(-3, 3, 100)
    plt.plot(grid, synthetic.predict(grid[:, None]), label="Gradient descent")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    _save(output, "synthetic_regression", show)
    grid_w, grid_b = np.meshgrid(np.linspace(-1, 5, 120), np.linspace(-4, 2, 120))
    losses = np.mean((grid_w[..., None] * synthetic_x + grid_b[..., None] - synthetic_y)**2, axis=-1)
    plt.figure(figsize=(7, 5))
    plt.contour(grid_w, grid_b, losses, levels=np.logspace(0, 1.5, 15))
    plt.plot(synthetic.trajectory[:, 0], synthetic.trajectory[:, 1], "r.-", label="GD trajectory")
    reference_1d = np.polyfit(synthetic_x, synthetic_y, 1)
    plt.plot(*reference_1d, "k*", ms=12, label="Least squares")
    plt.xlabel("Weight")
    plt.ylabel("Bias")
    plt.legend()
    _save(output, "loss_surface", show)

    data = load_housing(fast)
    x_train, x_test = data.x_train, data.x_test
    y_train, y_test = data.y_train, data.y_test
    reference, optimum = data.reference, data.optimum
    epochs = 50 if fast else 500
    fits = {
        "numpy": gradient_descent(x_train, y_train, epochs=epochs),
        "pytorch": pytorch_descent(x_train, y_train, epochs=epochs),
        "tensorflow": tensorflow_descent(x_train, y_train, epochs=epochs),
    }
    for name in ["pytorch", "tensorflow"]:
        np.testing.assert_allclose(fits[name].history, fits["numpy"].history, atol=1e-3, rtol=1e-4)
    if not fast:
        assert mse(y_train, fits["numpy"].predict(x_train)) <= 1.01 * optimum
    plt.figure(figsize=(8, 4))
    _curves({name: fit.history for name, fit in fits.items()}, "Full-batch GD in three libraries")
    plt.axhline(optimum, color="black", ls="--")
    _save(output, "framework_comparison", show)
    plt.figure(figsize=(7, 4))
    plt.barh(data.feature_names, fits["numpy"].weights)
    plt.xlabel("Weight (standardized features)")
    _save(output, "feature_weights", show)

    linear_fit = pytorch_sgd(x_train, y_train, epochs=epochs, batch_size=len(x_train), learning_rate=0.2)
    return {
        "samples": data.samples, "train_samples": len(x_train),
        "baseline_test_mse": mse(y_test, np.full_like(y_test, y_train.mean())),
        "least_squares_train_mse": optimum,
        "least_squares_test_mse": mse(y_test, x_test @ reference[:-1] + reference[-1]),
        "frameworks": {name: {"train_mse": mse(y_train, fit.predict(x_train)), "test_mse": mse(y_test, fit.predict(x_test)), "max_weight_difference_from_lstsq": float(np.max(np.abs(fit.weights - reference[:-1])))} for name, fit in fits.items()},
        "nn_linear_sgd_test_mse": mse(y_test, linear_fit.predict(x_test)),
        "history_note": "Full-batch histories record loss before each update; final metrics evaluate the returned weights.",
    }
