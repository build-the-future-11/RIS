import unittest

import numpy as np

from main import finite_difference_hessian


def quadratic(x):
    return 3 * x[0] ** 2 + 4 * x[0] * x[1] + 5 * x[1] ** 2


class HessianContractTests(unittest.TestCase):
    def test_integer_and_low_precision_inputs_match_mixed_analytic_hessian(self):
        expected = np.array([[6.0, 4.0], [4.0, 10.0]])
        for dtype in (np.int32, np.int64, np.float16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                point = np.array([1, 2], dtype=dtype)
                before = point.copy()
                actual = finite_difference_hessian(quadratic, point, eps=1e-3)
                np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
                np.testing.assert_array_equal(point, before)
                self.assertEqual(actual.dtype, np.float64)

    def test_python_sequence_is_supported(self):
        np.testing.assert_allclose(
            finite_difference_hessian(quadratic, [1, 2], eps=1e-3),
            [[6, 4], [4, 10]], atol=1e-6, rtol=0,
        )

    def test_cross_partials_are_symmetric_and_redundant_evaluations_are_avoided(self):
        points = []

        def objective(x):
            points.append(tuple(x))
            return quadratic(x)

        actual = finite_difference_hessian(objective, [0.5, -0.25])
        np.testing.assert_array_equal(actual, actual.T)
        self.assertEqual(len(points), 9)
        self.assertEqual(len(set(points)), 9)

    def test_single_coordinate_diagonal(self):
        actual = finite_difference_hessian(lambda x: 7 * x[0] ** 2 - 3 * x[0], [2.0], eps=1e-3)
        np.testing.assert_allclose(actual, [[14]], atol=1e-6, rtol=0)

    def test_constant_large_finite_offset_has_zero_hessian(self):
        actual = finite_difference_hessian(lambda x: 1e308, [0.0, 0.0])
        np.testing.assert_array_equal(actual, np.zeros((2, 2)))

    def test_invalid_coordinates_fail_before_objective(self):
        for point in ([], [[1, 2]], [np.nan], [np.inf], [1 + 2j], [True], ["1"]):
            with self.subTest(point=point):
                calls = []
                with self.assertRaises((TypeError, ValueError)):
                    finite_difference_hessian(lambda x: calls.append(x), point)
                self.assertFalse(calls)

    def test_invalid_steps_fail_before_objective(self):
        for eps in (0, -1, np.nan, np.inf, True, "0.1", [0.1], 1j, 1e-200, 1e200):
            with self.subTest(eps=eps):
                calls = []
                with self.assertRaises(ValueError):
                    finite_difference_hessian(lambda x: calls.append(x), [0.0], eps=eps)
                self.assertFalse(calls)

    def test_unresolvable_coordinate_step_fails_before_objective(self):
        calls = []
        with self.assertRaisesRegex(ValueError, "distinct"):
            finite_difference_hessian(lambda x: calls.append(x), [1e30])
        self.assertFalse(calls)

    def test_nonscalar_complex_and_nonfinite_outputs_fail(self):
        for result in ([1.0], np.array([1.0]), 1j, True, np.nan, np.inf):
            with self.subTest(result=result), self.assertRaisesRegex(ValueError, "scalar"):
                finite_difference_hessian(lambda x, value=result: value, [0.0])

    def test_nonfinite_curvature_is_rejected(self):
        with self.assertRaises(FloatingPointError):
            finite_difference_hessian(lambda x: 1e308 if x[0] != 0 else -1e308, [0.0])

    def test_objective_cannot_modify_callers_input(self):
        point = np.array([0.25, -0.5])
        before = point.copy()

        def objective(x):
            value = quadratic(x)
            x[:] = 100
            return value

        actual = finite_difference_hessian(objective, point, eps=1e-3)
        np.testing.assert_array_equal(point, before)
        np.testing.assert_allclose(actual, [[6, 4], [4, 10]], atol=1e-6, rtol=0)
