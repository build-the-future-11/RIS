"""Independent analytic review at one-ULP and mixed coordinate scales."""

import unittest

import numpy as np

from main import finite_difference_hessian


def quadratic(x):
    return 3 * x[0] ** 2 + 4 * x[0] * x[1] + 5 * x[1] ** 2


class IndependentStepReviewTests(unittest.TestCase):
    def test_actual_representable_steps_recover_shifted_quadratic(self):
        for center, eps in ((1e12, 1e-4), (1.0, 1.2e-16), (-1.0, 1.2e-16)):
            with self.subTest(center=center, eps=eps):
                actual = finite_difference_hessian(
                    lambda x: (x[0] - center) ** 2, [center], eps=eps,
                )
                np.testing.assert_allclose(actual, [[2.0]], rtol=1e-14, atol=0)

    def test_asymmetric_representable_steps_do_not_invent_linear_curvature(self):
        actual = finite_difference_hessian(lambda x: x[0] - 1.0, [1.0], eps=1.2e-16)
        np.testing.assert_array_equal(actual, [[0.0]])

    def test_mixed_curvature_uses_actual_steps_in_both_coordinates(self):
        for point, eps in ((np.array([1e12, -2e12]), 1e-3), (np.ones(2), 1.2e-16)):
            with self.subTest(point=point, eps=eps):
                actual = finite_difference_hessian(
                    lambda x: quadratic(x - point), point, eps=eps,
                )
                np.testing.assert_allclose(actual, [[6, 4], [4, 10]], rtol=1e-14, atol=0)

