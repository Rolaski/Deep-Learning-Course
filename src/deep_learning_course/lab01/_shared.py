"""Data preparation, training results, and plotting shared by Lab 01 tasks."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[3]

@dataclass
class FitResult:
    weights: np.ndarray
    bias: float
    history: np.ndarray
    times: np.ndarray | None = None
    trajectory: np.ndarray | None = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        return features @ self.weights + self.bias


def mse(target: np.ndarray, prediction: np.ndarray) -> float:
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape:
        raise ValueError(f"MSE shape mismatch: {target.shape} vs {prediction.shape}")
    return float(np.mean((target - prediction)**2))


def _validate(features, target, learning_rate, epochs):
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if features.ndim != 2 or target.shape != (len(features),) or not len(features):
        raise ValueError("Expected nonempty X (n, d) and y (n,).")
    if learning_rate <= 0 or epochs < 1:
        raise ValueError("Learning rate and epoch count must be positive.")
    return features, target


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


@dataclass
class HousingData:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    reference: np.ndarray
    optimum: float
    samples: int


@lru_cache(maxsize=2)
def load_housing(fast: bool = False) -> HousingData:
    housing = fetch_california_housing(data_home=str(PROJECT_ROOT / "data"))
    x, y = housing.data.astype(np.float64), housing.target.astype(np.float64)
    if fast:
        x, y = x[:2000], y[:2000]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    augmented = np.column_stack([x_train, np.ones(len(x_train))])
    reference = np.linalg.lstsq(augmented, y_train, rcond=None)[0]
    return HousingData(x_train, x_test, y_train, y_test, list(housing.feature_names), reference,
                       mse(y_train, augmented @ reference), len(x))
