"""Full-batch GD, mini-batch SGD, and framework implementations."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import tensorflow as tf
import torch


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


def minibatch_descent(features, target, learning_rate=0.05, epochs=30, batch_size=32, seed=0) -> FitResult:
    features, target = _validate(features, target, learning_rate, epochs)
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    rng = np.random.default_rng(seed)
    weights, bias = np.zeros(features.shape[1]), 0.0
    history, times = [], []
    start = perf_counter()
    for _ in range(epochs):
        order = rng.permutation(len(features))
        for offset in range(0, len(features), batch_size):
            indices = order[offset:offset + batch_size]
            xb, yb = features[indices], target[indices]
            error = xb @ weights + bias - yb
            weights -= learning_rate * (2 / len(indices)) * xb.T @ error
            bias -= learning_rate * 2 * error.mean()
        history.append(mse(target, features @ weights + bias))
        times.append(perf_counter() - start)
    return FitResult(weights, float(bias), np.array(history), np.array(times))


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


def pytorch_sgd(features, target, epochs=30, batch_size=64, learning_rate=0.05, seed=42) -> FitResult:
    features, target = _validate(features, target, learning_rate, epochs)
    torch.manual_seed(seed)
    x, y = torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = torch.nn.Linear(x.shape[1], 1).to("cuda")
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    x_gpu, y_gpu = x.to("cuda"), y.to("cuda")
    if batch_size >= len(x):
        loader = [(x_gpu, y_gpu)]
    history = []
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = (model(xb.to("cuda")).squeeze(1) - yb.to("cuda")).square().mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            history.append((model(x_gpu).squeeze(1) - y_gpu).square().mean().item())
    return FitResult(model.weight.detach().cpu().numpy().ravel(), model.bias.item(), np.array(history))


def stability_threshold(features: np.ndarray) -> tuple[np.ndarray, float]:
    augmented = np.column_stack([features, np.ones(len(features))])
    hessian = (2 / len(features)) * augmented.T @ augmented
    eigenvalues = np.linalg.eigvalsh(hessian)
    return eigenvalues, float(2 / eigenvalues[-1])
