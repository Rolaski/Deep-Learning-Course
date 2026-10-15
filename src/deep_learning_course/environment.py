"""Check framework imports, GPU execution, and automatic differentiation."""

import json
import platform

import keras
import matplotlib
import numpy as np
import sklearn
import tensorflow as tf


def check_environment() -> dict:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("TensorFlow cannot access a GPU.")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_soft_device_placement(False)

    with tf.device("/GPU:0"):
        x = tf.Variable([1.0, 2.0, 3.0])
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(x * x)
        gradient = tape.gradient(loss, x)
    np.testing.assert_allclose(gradient.numpy(), [2.0, 4.0, 6.0])
    if "GPU:0" not in gradient.device:
        raise RuntimeError("TensorFlow gradient did not execute on the GPU.")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA.")
    x_pt = torch.tensor([1.0, 2.0, 3.0], device="cuda", requires_grad=True)
    (x_pt * x_pt).sum().backward()
    np.testing.assert_allclose(x_pt.grad.detach().cpu().numpy(), [2.0, 4.0, 6.0])
    torch.cuda.synchronize()

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "scikit_learn": sklearn.__version__,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "tensorflow_gradient_device": gradient.device,
        "pytorch_gradient_device": str(x_pt.grad.device),
        "gradient_checks": "passed",
    }


if __name__ == "__main__":
    print(json.dumps(check_environment(), indent=2))
