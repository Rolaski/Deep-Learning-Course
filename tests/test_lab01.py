"""Numerical checks against independent solvers and framework optimizers."""

import unittest

import numpy as np
import torch

from deep_learning_course.lab01.task02 import analytical_gradient, framework_gradients, gradient_check
from deep_learning_course.lab01.extra import optimize_rosenbrock, rosenbrock
from deep_learning_course.lab01.task06 import fit_ridge
from deep_learning_course.lab01._shared import mse
from deep_learning_course.lab01.task03 import gradient_descent
from deep_learning_course.lab01.task04 import stability_threshold
from deep_learning_course.lab01.task05 import minibatch_descent
from deep_learning_course.lab01.task01 import pairwise_sq_dist_np, pairwise_sq_dist_torch, standardize_images


class Lab01Tests(unittest.TestCase):
    def test_standardization_and_constant_channel(self):
        batch = np.random.default_rng(4).normal(size=(4, 5, 6, 3)).astype(np.float32)
        batch[..., 2] = 7
        z_tf, z_pt = standardize_images(batch)
        np.testing.assert_allclose(z_tf.numpy(), z_pt.permute(0, 2, 3, 1).numpy(), atol=1e-5)
        np.testing.assert_allclose(z_tf.numpy()[..., 2], 0)
        np.testing.assert_allclose(z_tf.numpy()[..., :2].std(axis=(0, 1, 2)), 1, atol=1e-5)

    def test_distances_against_direct_differences(self):
        rng = np.random.default_rng(5)
        x, y = rng.normal(size=(7, 4)), rng.normal(size=(3, 4))
        expected = np.linalg.norm(x[:, None] - y[None], axis=-1)**2
        np.testing.assert_allclose(pairwise_sq_dist_np(x, y), expected, atol=1e-12)
        np.testing.assert_allclose(pairwise_sq_dist_torch(torch.tensor(x), torch.tensor(y)), expected, atol=1e-12)

    def test_autograd_and_central_differences(self):
        for gradient in framework_gradients():
            np.testing.assert_allclose(gradient, analytical_gradient(0.5, -1), atol=1e-6)
        self.assertLess(gradient_check()[2], 1e-7)

    def test_gd_against_least_squares_and_stability_boundary(self):
        rng = np.random.default_rng(6)
        x, y = rng.normal(size=(80, 3)), rng.normal(size=80)
        expected = np.linalg.lstsq(np.column_stack([x, np.ones(80)]), y, rcond=None)[0]
        fit = gradient_descent(x, y, epochs=500)
        np.testing.assert_allclose(np.r_[fit.weights, fit.bias], expected, atol=1e-8)
        _, limit = stability_threshold(x)
        diverging = gradient_descent(x, y, learning_rate=1.05 * limit, epochs=150)
        self.assertGreater(diverging.history[-1], diverging.history[0])
        with self.assertRaises(ValueError):
            mse(y, y[:, None])

    def test_partial_minibatch_against_torch_sgd(self):
        rng = np.random.default_rng(7)
        x, y = rng.normal(size=(11, 3)), rng.normal(size=11)
        actual = minibatch_descent(x, y, learning_rate=0.01, epochs=1, batch_size=4, seed=8)
        model = torch.nn.Linear(3, 1, dtype=torch.float64)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        order = np.random.default_rng(8).permutation(11)
        for indices in np.array_split(order, [4, 8]):
            optimizer.zero_grad()
            loss = (model(torch.tensor(x[indices])).squeeze(1) - torch.tensor(y[indices])).square().mean()
            loss.backward()
            optimizer.step()
        np.testing.assert_allclose(actual.weights, model.weight.detach().numpy().ravel(), atol=1e-12)
        self.assertAlmostEqual(actual.bias, model.bias.item(), places=12)

    def test_ridge_preserves_unpenalized_intercept(self):
        x = np.linspace(0, 1, 20)
        weights = fit_ridge(x, np.full_like(x, 3.0), 15, 100)
        self.assertAlmostEqual(weights[0], 3.0, places=10)
        np.testing.assert_allclose(weights[1:], 0, atol=1e-10)

    def test_manual_adam_against_torch_optimizer(self):
        manual = optimize_rosenbrock("adam", 0.02, steps=50)
        point = torch.tensor([-1.5, 2.0], dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([point], lr=0.02)
        for _ in range(50):
            optimizer.zero_grad()
            rosenbrock(point).backward()
            optimizer.step()
        np.testing.assert_allclose(manual[-1], point.detach().numpy(), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
