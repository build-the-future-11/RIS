"""Analytic stencils at unequal neighboring float64 distances."""

import unittest

import numpy as np

from main import finite_difference_hessian


class RepresentedStepTests(unittest.TestCase):
    def test_linear_objective_has_no_spurious_curvature_at_binade_boundary(self):
        center = 2.0**40
        step = 1.25 * 2.0**-13
        self.assertNotEqual((center + step) - center, center - (center - step))
        result = finite_difference_hessian(lambda x: x[0] - center, [center], step)
        np.testing.assert_allclose(result, [[0.0]], atol=1e-12, rtol=0)

    def test_mixed_quadratic_uses_actual_coordinate_distances(self):
        center = np.array([2.0**40, -2.0**40])
        calls = []

        def objective(x):
            calls.append(tuple(x))
            a, b = x - center
            return 3*a*a + 4*a*b + 5*b*b + 7*a - 2*b

        result = finite_difference_hessian(objective, center, 1.25 * 2.0**-13)
        np.testing.assert_allclose(result, [[6.0, 4.0], [4.0, 10.0]], atol=1e-8, rtol=0)
        np.testing.assert_array_equal(result, result.T)
        self.assertEqual(len(calls), 9)
        self.assertEqual(len(set(calls)), 9)

    def test_symmetric_rounded_steps_do_not_use_requested_denominator(self):
        center = 1e12
        result = finite_difference_hessian(lambda x: 3*(x[0]-center)**2, [center], 1e-4)
        np.testing.assert_allclose(result, [[6.0]], atol=1e-12, rtol=0)
