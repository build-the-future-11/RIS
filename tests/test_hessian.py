import math
import unittest

import numpy as np

from main import finite_difference_hessian, irr_fn


class FiniteDifferenceHessianTests(unittest.TestCase):
    def test_matches_analytic_hessian_for_demo_objective(self):
        x = np.array([1.0, 2.0])
        h = finite_difference_hessian(irr_fn, x)

        expected = np.array(
            [
                [-math.sin(1.0), 0.0],
                [0.0, -math.cos(2.0)],
            ]
        )

        np.testing.assert_allclose(h, expected, rtol=0.0, atol=2e-6)

    def test_does_not_mutate_input(self):
        x = np.array([0.25, -0.75])
        before = x.copy()
        finite_difference_hessian(irr_fn, x)
        np.testing.assert_array_equal(x, before)

    def test_output_is_symmetric_for_smooth_scalar_objective(self):
        x = np.array([0.4, 1.3])
        h = finite_difference_hessian(irr_fn, x)
        np.testing.assert_allclose(h, h.T, rtol=0.0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
