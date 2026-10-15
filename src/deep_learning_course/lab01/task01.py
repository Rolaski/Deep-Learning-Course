"""Task 1: channel standardization, vectorization, and CPU/GPU comparison."""

from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf
import torch


def standardize_images(batch: np.ndarray) -> tuple[tf.Tensor, torch.Tensor]:
    """Standardize NHWC images per channel, returning NHWC and NCHW tensors."""
    x_tf = tf.convert_to_tensor(batch, dtype=tf.float32)
    mean_tf = tf.reduce_mean(x_tf, axis=(0, 1, 2), keepdims=True)
    std_tf = tf.math.reduce_std(x_tf, axis=(0, 1, 2), keepdims=True)
    z_tf = (x_tf - mean_tf) / tf.maximum(std_tf, 1e-7)
    x_pt = torch.as_tensor(batch, dtype=torch.float32).permute(0, 3, 1, 2)
    mean_pt = x_pt.mean(dim=(0, 2, 3), keepdim=True)
    std_pt = x_pt.std(dim=(0, 2, 3), correction=0, keepdim=True)
    z_pt = (x_pt - mean_pt) / std_pt.clamp_min(1e-7)
    return z_tf, z_pt


def pairwise_sq_dist_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(
        (x * x).sum(axis=1, keepdims=True)
        + (y * y).sum(axis=1)[None, :] - 2 * x @ y.T,
        0,
    )


def pairwise_sq_dist_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (
        (x * x).sum(dim=1, keepdim=True)
        + (y * y).sum(dim=1).unsqueeze(0) - 2 * x @ y.T
    ).clamp_min(0)


def run(output: Path, fast: bool = False, show: bool = False) -> dict:
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
