"""Optional exercise: hand-written GD, momentum, and Adam."""

import numpy as np
import torch


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
