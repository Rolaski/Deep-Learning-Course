"""Polynomial capacity and L2 regularization (exercises 6.1 and 6.2)."""

import numpy as np


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
