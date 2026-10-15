"""Task 5: mini-batch SGD, DataLoader, and batch-dependent stability."""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from ._shared import FitResult, _validate, _save, load_housing, mse

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


def run(output: Path, fast: bool = False, show: bool = False) -> dict:
    data = load_housing(fast)
    x_train, y_train, optimum = data.x_train, data.y_train, data.optimum
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
    return {
        "minibatches": batch_results,
        "minibatches_original_learning_rate": 0.05,
        "minibatches_safe_learning_rate": float(safe_rate),
        "minibatches_stable": stable_batches,
        "minibatch_note": "Full-batch stability does not imply mini-batch stability. Small batches can diverge at lr=0.05. The stable comparison uses a conservative training-data bound including partial batches.",
        "dataloader": {
            "original_learning_rate": 0.05,
            "original_train_mse": loader_mse if np.isfinite(loader_mse) else None,
            "original_status": "completed" if np.isfinite(loader_mse) else "nonfinite_loss",
            "safe_learning_rate": float(safe_rate),
            "safe_train_mse": stable_loader_mse,
        },
    }
