"""Extra task: GD, momentum, and Adam on the Rosenbrock function."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ._shared import _save


def rosenbrock(point: torch.Tensor) -> torch.Tensor:
    return (1 - point[0])**2 + 100 * (point[1] - point[0]**2)**2


def optimize_rosenbrock(method: str, learning_rate: float, steps=3000, start=(-1.5, 2.0)) -> np.ndarray:
    if method not in {"gd", "momentum", "adam"}:
        raise ValueError(f"Unknown optimizer: {method}")
    if learning_rate <= 0 or steps < 1:
        raise ValueError("Learning rate and step count must be positive.")
    point = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    first, second = torch.zeros_like(point), torch.zeros_like(point)
    path = [point.detach().numpy().copy()]
    for step in range(1, steps + 1):
        gradient, = torch.autograd.grad(rosenbrock(point), point)
        with torch.no_grad():
            if method == "gd":
                update = gradient
            elif method == "momentum":
                first = 0.9 * first + gradient
                update = first
            else:
                first = 0.9 * first + 0.1 * gradient
                second = 0.999 * second + 0.001 * gradient.square()
                corrected_first = first / (1 - 0.9**step)
                corrected_second = second / (1 - 0.999**step)
                update = corrected_first / (corrected_second.sqrt() + 1e-8)
            point -= learning_rate * update
        path.append(point.detach().numpy().copy())
    return np.array(path)


def run(output: Path, fast: bool = False, show: bool = False) -> dict:
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
