"""Reproducible experiments and figures for all Lab 01 assignments."""

from pathlib import Path
from time import perf_counter

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from .gradients import analytical_gradient, framework_gradients, gradient_check
from .optimizers import optimize_rosenbrock
from .polynomial import fit_polynomial, fit_ridge, polynomial_data, polynomial_features
from .regression import (
    gradient_descent, minibatch_descent, mse, stability_threshold,
    pytorch_descent, pytorch_sgd, tensorflow_descent,
)
from .tensors import pairwise_sq_dist_np, pairwise_sq_dist_torch, standardize_images


def _save(output: Path, name: str, show: bool):
    plt.tight_layout()
    plt.savefig(output / f"{name}.png", dpi=160)
    if show:
        plt.show()
    plt.close()


def _curves(curves, title, xlabel="Epoch", ylabel="MSE", log=True):
    for label, values in curves.items():
        plt.plot(np.arange(1, len(values) + 1), values, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log:
        plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.legend()


def _minibatch_experiment(x, y, batches, epochs, rate, optimum, output, name, show):
    fits = {batch: minibatch_descent(x, y, learning_rate=rate, epochs=epochs, batch_size=batch) for batch in batches}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    results = {}
    for batch, fit in fits.items():
        label = "full-batch" if batch == len(x) else f"B={batch}"
        axes[0].plot(np.arange(1, len(fit.history) + 1), fit.history, label=label)
        axes[1].plot(fit.times, fit.history, label=label)
        results[label] = {"train_mse": float(fit.history[-1]), "seconds": float(fit.times[-1]), "updates_per_epoch": int(np.ceil(len(x) / batch))}
    for axis, xlabel in zip(axes, ["Epoch", "Time [s]"]):
        axis.set(xlabel=xlabel, ylabel="Train MSE", yscale="log")
        axis.axhline(optimum, color="black", ls="--")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle(f"Mini-batch SGD: learning rate {rate:.5g}")
    _save(output, name, show)
    return results


def tensor_experiments(fast: bool) -> dict:
    rng = np.random.default_rng(0)
    batch = rng.normal(loc=[0.5, 0.2, -0.3], scale=[2, 0.5, 1], size=(16, 32, 32, 3)).astype(np.float32)
    z_tf, z_pt = standardize_images(batch)
    np.testing.assert_allclose(z_tf.numpy(), z_pt.permute(0, 2, 3, 1).numpy(), atol=1e-5)
    np.testing.assert_allclose(z_tf.numpy().mean(axis=(0, 1, 2)), 0, atol=1e-5)
    np.testing.assert_allclose(z_tf.numpy().std(axis=(0, 1, 2)), 1, atol=1e-5)
    x, y = rng.random((50, 8)), rng.random((40, 8))
    reference = np.sum((x[:, None, :] - y[None, :, :])**2, axis=-1)
    np.testing.assert_allclose(pairwise_sq_dist_np(x, y), reference, atol=1e-12)
    np.testing.assert_allclose(pairwise_sq_dist_torch(torch.from_numpy(x), torch.from_numpy(y)).numpy(), reference, atol=1e-12)

    size = 150 if fast else 400
    x, y = rng.random((size, 32)), rng.random((size, 32))
    start = perf_counter()
    loop = np.empty((size, size))
    for i in range(size):
        for j in range(size):
            loop[i, j] = np.sum((x[i] - y[j])**2)
    loop_seconds = perf_counter() - start
    pairwise_sq_dist_np(x, y)  # Warm up BLAS before timing.
    start = perf_counter()
    vectorized = pairwise_sq_dist_np(x, y)
    vector_seconds = perf_counter() - start
    np.testing.assert_allclose(vectorized, loop, atol=1e-12)

    dimension = 512 if fast else 4096
    a, b = torch.randn(dimension, dimension), torch.randn(dimension, dimension)
    _ = a @ b
    start = perf_counter()
    cpu = a @ b
    cpu_seconds = perf_counter() - start
    a_gpu, b_gpu = a.to("cuda"), b.to("cuda")
    _ = a_gpu @ b_gpu
    torch.cuda.synchronize()
    start = perf_counter()
    gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_seconds = perf_counter() - start
    tf_times = {}
    for device in ["/CPU:0", "/GPU:0"]:
        with tf.device(device):
            a_tf, b_tf = tf.constant(a.numpy()), tf.constant(b.numpy())
            _ = tf.matmul(a_tf, b_tf).numpy()
            start = perf_counter()
            _ = tf.matmul(a_tf, b_tf).numpy()
            tf_times[device] = perf_counter() - start
    return {
        "standardization_and_distances": "passed",
        "loop_seconds": loop_seconds, "vector_seconds": vector_seconds,
        "distance_speedup": loop_seconds / vector_seconds,
        "matrix_dimension": dimension, "torch_cpu_seconds": cpu_seconds,
        "torch_gpu_seconds": gpu_seconds,
        "torch_max_cpu_gpu_difference": (cpu - gpu.cpu()).abs().max().item(),
        "tensorflow_seconds_including_result_copy": tf_times,
        "timing_note": "Single warmed-up measurements; PyTorch excludes transfers; TensorFlow includes result transfer.",
    }


def gradient_experiments(output: Path, show: bool) -> dict:
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


def regression_experiments(output: Path, fast: bool, show: bool) -> dict:
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

    housing = fetch_california_housing(data_home=str(Path("data").resolve()))
    x, y = housing.data.astype(np.float64), housing.target.astype(np.float64)
    if fast:
        x, y = x[:2000], y[:2000]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    augmented = np.column_stack([x_train, np.ones(len(x_train))])
    reference = np.linalg.lstsq(augmented, y_train, rcond=None)[0]
    optimum = mse(y_train, augmented @ reference)
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
    plt.barh(housing.feature_names, fits["numpy"].weights)
    plt.xlabel("Weight (standardized features)")
    _save(output, "feature_weights", show)

    eigenvalues, threshold = stability_threshold(x_train)
    rates = [0.001, 0.01, 0.1, 0.5 * threshold, 0.98 * threshold, 1.05 * threshold]
    histories = {rate: gradient_descent(x_train, y_train, rate, 40 if fast else 200).history for rate in rates}
    plt.figure(figsize=(9, 5))
    _curves({f"lr={rate:.5f}": history for rate, history in histories.items()}, f"Learning rate (critical value {threshold:.5f})")
    plt.axhline(optimum, color="black", ls="--")
    plt.ylim(optimum * 0.9, max(10, optimum * 100))
    _save(output, "learning_rate", show)
    rate_results = {}
    for rate, history in histories.items():
        reached = np.flatnonzero(history <= 1.01 * optimum)
        rate_results[f"{rate:.8g}"] = {
            "last_recorded_train_mse": float(history[-1]),
            "updates_to_within_one_percent": int(reached[0]) if len(reached) else None,
            "spectral_radius": float(np.max(np.abs(1 - rate * eigenvalues))),
            "loss_increased_in_final_step": bool(history[-1] > history[-2]),
        }

    batches = [16, 64, 256, len(x_train)]
    mb_epochs = 4 if fast else 30
    batch_results = _minibatch_experiment(x_train, y_train, batches, mb_epochs, 0.05, optimum, output, "minibatch", show)
    # lambda_max(H_batch) <= 2 * mean(||[x_i, 1]||**2) in that batch.
    # The largest k row norms bound every possible shuffled batch of size k.
    norms = np.sort(np.sum(x_train**2, axis=1) + 1)[::-1]
    smallest_batches = [len(x_train) % batch or batch for batch in batches]
    safe_rate = min(0.05, *(0.9 * size / norms[:size].sum() for size in smallest_batches))
    stable_batches = _minibatch_experiment(x_train, y_train, batches, mb_epochs, safe_rate, optimum, output, "minibatch_stable", show)
    loader_fit = pytorch_sgd(x_train, y_train, epochs=mb_epochs)
    loader_mse = mse(y_train, loader_fit.predict(x_train))
    stable_loader = pytorch_sgd(x_train, y_train, epochs=mb_epochs, learning_rate=safe_rate)
    stable_loader_mse = mse(y_train, stable_loader.predict(x_train))
    assert np.isfinite(stable_loader_mse), "SGD failed with the conservative learning rate."
    linear_fit = pytorch_sgd(x_train, y_train, epochs=epochs, batch_size=len(x_train), learning_rate=0.2)
    return {
        "samples": len(x), "train_samples": len(x_train),
        "baseline_test_mse": mse(y_test, np.full_like(y_test, y_train.mean())),
        "least_squares_train_mse": optimum,
        "least_squares_test_mse": mse(y_test, x_test @ reference[:-1] + reference[-1]),
        "frameworks": {name: {"train_mse": mse(y_train, fit.predict(x_train)), "test_mse": mse(y_test, fit.predict(x_test)), "max_weight_difference_from_lstsq": float(np.max(np.abs(fit.weights - reference[:-1])))} for name, fit in fits.items()},
        "hessian_eigenvalues": eigenvalues.tolist(), "critical_learning_rate": threshold,
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]), "learning_rates": rate_results,
        "stability_note": "A spectral radius above one is unstable; loss may initially decrease before the unstable component dominates, especially in fast mode.",
        "minibatches": batch_results,
        "minibatches_original_learning_rate": 0.05,
        "minibatches_safe_learning_rate": float(safe_rate),
        "minibatches_stable": stable_batches,
        "minibatch_note": "Full-batch stability does not imply mini-batch stability. California Housing outliers destabilize small batches at lr=0.05. The second comparison uses one conservative training-data bound for all batch sizes, including the final partial batch.",
        "dataloader": {
            "original_learning_rate": 0.05,
            "original_train_mse": loader_mse if np.isfinite(loader_mse) else None,
            "original_status": "completed" if np.isfinite(loader_mse) else "nonfinite_loss",
            "safe_learning_rate": float(safe_rate),
            "safe_train_mse": stable_loader_mse,
        },
        "nn_linear_sgd_test_mse": mse(y_test, linear_fit.predict(x_test)),
        "history_note": "Full-batch histories record loss before each update; final metrics evaluate the returned weights. Mini-batch histories record loss after each epoch.",
    }


def polynomial_experiments(output: Path, show: bool) -> dict:
    x, y, xv, yv = polynomial_data()
    degrees = list(range(16))
    train_errors, val_errors = [], []
    for degree in degrees:
        weights = fit_polynomial(x, y, degree)
        train_errors.append(mse(y, polynomial_features(x, degree) @ weights))
        val_errors.append(mse(yv, polynomial_features(xv, degree) @ weights))
    best_degree = int(np.argmin(val_errors))
    plt.figure(figsize=(8, 4))
    plt.semilogy(degrees, train_errors, "o-", label="Train")
    plt.semilogy(degrees, val_errors, "s-", label="Validation")
    plt.axvline(best_degree, color="gray", ls=":")
    plt.axhline(0.09, color="black", ls="--", label="Noise variance")
    plt.xlabel("Degree")
    plt.ylabel("MSE")
    plt.legend()
    _save(output, "polynomial_capacity", show)
    grid = np.linspace(0, 1, 400)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, degree in zip(axes, [1, 4, 15]):
        axis.plot(grid, np.sin(2 * np.pi * grid), "g--", label="True function")
        axis.plot(grid, polynomial_features(grid, degree) @ fit_polynomial(x, y, degree), label=f"Degree {degree}")
        axis.scatter(x, y, s=12, label="Train")
        axis.set(ylim=(-2, 2), xlabel="x")
        axis.legend()
    _save(output, "polynomial_fits", show)

    strengths = np.logspace(-12, 1, 27)
    ridge_train, ridge_val = [], []
    for strength in strengths:
        weights = fit_ridge(x, y, 15, strength)
        ridge_train.append(mse(y, polynomial_features(x, 15) @ weights))
        ridge_val.append(mse(yv, polynomial_features(xv, 15) @ weights))
    best = int(np.argmin(ridge_val))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].loglog(strengths, ridge_train, "o-", label="Train")
    axes[0].loglog(strengths, ridge_val, "s-", label="Validation")
    axes[0].set(xlabel="L2 strength", ylabel="MSE")
    axes[0].legend()
    axes[1].plot(grid, np.sin(2 * np.pi * grid), "g--", label="True function")
    axes[1].plot(grid, polynomial_features(grid, 15) @ fit_polynomial(x, y, 15), alpha=0.5, label="Unregularized")
    axes[1].plot(grid, polynomial_features(grid, 15) @ fit_ridge(x, y, 15, strengths[best]), label="Ridge")
    axes[1].scatter(x, y, s=12)
    axes[1].set(ylim=(-2, 2), xlabel="x", ylabel="y")
    axes[1].legend()
    _save(output, "ridge", show)
    return {"best_degree": best_degree, "train_mse_by_degree": train_errors, "validation_mse_by_degree": val_errors, "best_ridge_strength": float(strengths[best]), "ridge_train_mse": ridge_train, "ridge_validation_mse": ridge_val, "ridge_strengths": strengths.tolist()}


def optimizer_experiments(output: Path, fast: bool, show: bool) -> dict:
    steps = 300 if fast else 3000
    methods = {"GD": ("gd", 1e-3), "Momentum": ("momentum", 2e-4), "Momentum (large lr)": ("momentum", 1e-3), "Adam": ("adam", 0.02)}
    grid_x, grid_y = np.meshgrid(np.linspace(-2, 2, 300), np.linspace(-1, 3, 300))
    z = (1 - grid_x)**2 + 100 * (grid_y - grid_x**2)**2
    plt.figure(figsize=(8, 6))
    plt.contour(grid_x, grid_y, z, levels=np.logspace(-1, 3.5, 25), alpha=0.6)
    results = {}
    for label, (method, rate) in methods.items():
        path = optimize_rosenbrock(method, rate, steps)
        distance = float(np.linalg.norm(path[-1] - 1))
        results[label] = {"steps": steps, "distance_to_minimum": distance, "final_point": path[-1].tolist()}
        plt.plot(path[:, 0], path[:, 1], lw=1, label=f"{label}: distance={distance:.3f}")
    plt.plot(1, 1, "k*", ms=12)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    _save(output, "rosenbrock", show)
    return results


def run_experiments(output: Path, fast=False, show=False) -> dict:
    keras.utils.set_random_seed(42)
    torch.manual_seed(42)
    results = {}
    for name, function in [
        ("tensors", lambda: tensor_experiments(fast)),
        ("gradients", lambda: gradient_experiments(output, show)),
        ("regression", lambda: regression_experiments(output, fast, show)),
        ("polynomial", lambda: polynomial_experiments(output, show)),
        ("optimizers", lambda: optimizer_experiments(output, fast, show)),
    ]:
        print(f"Running {name}...", flush=True)
        results[name] = function()
    return results
