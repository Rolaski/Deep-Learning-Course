"""Task 6: polynomial capacity and ridge regularization."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._shared import _save, mse


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    if degree < 0:
        raise ValueError("Degree must be nonnegative.")
    return np.vander(np.asarray(x, dtype=np.float64), degree + 1, increasing=True)


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    return np.linalg.lstsq(polynomial_features(x, degree), y, rcond=None)[0]


def fit_ridge(x: np.ndarray, y: np.ndarray, degree: int, strength: float) -> np.ndarray:
    if strength < 0:
        raise ValueError("Regularization strength must be nonnegative.")
    if strength == 0:
        return fit_polynomial(x, y, degree)
    design = polynomial_features(x, degree)
    penalty = np.eye(degree + 1)
    penalty[0, 0] = 0  # The intercept is not penalized.
    return np.linalg.solve(design.T @ design + len(x) * strength * penalty, design.T @ y)


def polynomial_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    x_train = rng.uniform(0, 1, 20)
    y_train = np.sin(2 * np.pi * x_train) + rng.normal(scale=0.3, size=20)
    x_val = rng.uniform(0, 1, 200)
    y_val = np.sin(2 * np.pi * x_val) + rng.normal(scale=0.3, size=200)
    return x_train, y_train, x_val, y_val


def run(output: Path, fast: bool = False, show: bool = False) -> dict:
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
