"""Task 4: learning rate and the Hessian stability threshold."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ._shared import _save, _curves, load_housing
from .task03 import gradient_descent

def stability_threshold(features: np.ndarray) -> tuple[np.ndarray, float]:
    augmented = np.column_stack([features, np.ones(len(features))])
    hessian = (2 / len(features)) * augmented.T @ augmented
    eigenvalues = np.linalg.eigvalsh(hessian)
    return eigenvalues, float(2 / eigenvalues[-1])


def run(output: Path, fast: bool = False, show: bool = False) -> dict:
    data = load_housing(fast)
    x_train, y_train, optimum = data.x_train, data.y_train, data.optimum
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

    return {
        "hessian_eigenvalues": eigenvalues.tolist(), "critical_learning_rate": threshold,
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]), "learning_rates": rate_results,
        "stability_note": "A spectral radius above one is unstable; loss may initially decrease before the unstable component dominates, especially in fast mode.",
    }
