"""Exercises 1.1 and 1.2."""

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
