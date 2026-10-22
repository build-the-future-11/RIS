"""Independent analytic and Decimal oracles for finite stencil range loss."""

import unittest
from decimal import Decimal, localcontext

import numpy as np

from main import finite_difference_hessian


def capture(objective, point, eps):
    calls = []

    def observed(values):
        position = tuple(values)
        result = float(objective(values))
        calls.append((position, result))
        return result

    result = finite_difference_hessian(observed, point, eps)
    return result, calls


def decimal_stencil(calls, point, eps):
    """Differentiate Lagrange basis polynomials over captured float64 samples."""
    point = np.asarray(point, dtype=np.float64)
    coordinates = [(float(value - eps), float(value), float(value + eps)) for value in point]
    samples = {position: Decimal.from_float(value) for position, value in calls}
    with localcontext() as context:
        context.prec = 800
        first, second = [], []
        for axis in coordinates:
            middle = Decimal.from_float(axis[1])
            nodes = [Decimal.from_float(value) - middle for value in axis]
            w1, w2 = [], []
            for index, node in enumerate(nodes):
                a, b = [other for j, other in enumerate(nodes) if j != index]
                denominator = (node - a) * (node - b)
                w1.append(-(a + b) / denominator)
                w2.append(Decimal(2) / denominator)
            first.append(w1)
            second.append(w2)
        result = np.empty((len(point), len(point)))
        for i in range(len(point)):
            total = Decimal(0)
            for k in range(3):
                position = list(point)
                position[i] = coordinates[i][k]
                total += second[i][k] * samples[tuple(position)]
            result[i, i] = float(total)
            for j in range(i + 1, len(point)):
                total = Decimal(0)
                for k in range(3):
                    for m in range(3):
                        position = list(point)
                        position[i], position[j] = coordinates[i][k], coordinates[j][m]
                        total += first[i][k] * first[j][m] * samples[tuple(position)]
                result[i, j] = result[j, i] = float(total)
        return result


def expected_order(point, eps):
    point = np.asarray(point, dtype=np.float64)
    result = [tuple(point)]
    for i in range(len(point)):
        for sign in (1, -1):
            position = point.copy()
            position[i] += sign * eps
            result.append(tuple(position))
    for i in range(len(point)):
        for j in range(i + 1, len(point)):
            for si, sj in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
                position = point.copy()
                position[i] += si * eps
                position[j] += sj * eps
                result.append(tuple(position))
    return result


class StencilArithmeticTests(unittest.TestCase):
    def assert_captured_oracle(self, actual, calls, point, eps):
        self.assertTrue(all(np.isfinite(value) for _, value in calls))
        np.testing.assert_array_equal(actual, decimal_stencil(calls, point, eps))
        self.assertEqual([position for position, _ in calls], expected_order(point, eps))

    def test_large_diagonal_has_finite_curvature_despite_overflowing_difference(self):
        actual, calls = capture(lambda x: (2*(x[0]/1e154)**2-1)*1e308, [0.], 1e154)
        np.testing.assert_allclose(actual, [[4.]], rtol=1e-14, atol=0)
        self.assert_captured_oracle(actual, calls, [0.], 1e154)

    def test_large_mixed_curvature_uses_the_captured_tensor_stencil(self):
        def objective(x):
            u, v = x / 1e154
            return (1.5*u*u + .5*u - 1)*v*1e308

        actual, calls = capture(objective, [0., 0.], 1e154)
        np.testing.assert_allclose(actual, [[0., .5], [.5, 0.]], rtol=1e-14, atol=0)
        self.assert_captured_oracle(actual, calls, [0., 0.], 1e154)

    def test_nonuniform_large_diagonal_uses_both_represented_distances(self):
        scale, eps = 2.0**510, 1.25*2.0**509
        for center in (2.0**562, -2.0**562):
            with self.subTest(center=center):
                actual, calls = capture(
                    lambda x: (2*((x[0]-center)/scale)**2-1)*1e308, [center], eps,
                )
                self.assertNotEqual((center+eps)-center, center-(center-eps))
                np.testing.assert_allclose(actual, [[4*(1e308/(scale*scale))]], rtol=1e-14, atol=0)
                self.assert_captured_oracle(actual, calls, [center], eps)

    def test_nonuniform_mixed_curvature_agrees_with_independent_lagrange_weights(self):
        center, scale, eps = np.array([2.0**562, -2.0**562]), 2.0**510, 1.25*2.0**509

        def objective(x):
            u, v = (x-center)/scale
            return (1.5*u*u + .5*u - 1)*v*1e308

        actual, calls = capture(objective, center, eps)
        cross = .5*(1e308/(scale*scale))
        np.testing.assert_allclose(actual, [[0., cross], [cross, 0.]], rtol=1e-14, atol=0)
        self.assert_captured_oracle(actual, calls, center, eps)

    def test_exact_affine_cancellation_survives_an_overflowing_difference(self):
        center, scale, eps = 2.0**562, 2.0**510, 1.25*2.0**509
        actual, calls = capture(
            lambda x: (2*((x[0]-center)/scale)-.5)*(2.0**1023), [center], eps,
        )
        np.testing.assert_array_equal(actual, [[0.]])
        self.assert_captured_oracle(actual, calls, [center], eps)

    def test_subnormal_diagonal_rounds_once_from_the_exact_stencil(self):
        smallest = float.fromhex('0x0.0000000000001p-1022')
        actual, calls = capture(lambda x: smallest*(x[0]/.75)**2, [0.], .75)
        self.assertEqual(float(actual[0, 0]).hex(), '0x0.0000000000004p-1022')
        self.assert_captured_oracle(actual, calls, [0.], .75)

    def test_representable_subnormal_mixed_curvature_cannot_become_zero(self):
        smallest = float.fromhex('0x0.0000000000001p-1022')
        actual, calls = capture(lambda x: smallest*(x[0]/.75)*(x[1]/.75), [0., 0.], .75)
        self.assertEqual(float(actual[0, 1]).hex(), '0x0.0000000000002p-1022')
        self.assert_captured_oracle(actual, calls, [0., 0.], .75)

    def test_truly_unrepresentable_curvature_is_rejected_without_re_evaluation(self):
        smallest = float.fromhex('0x0.0000000000001p-1022')
        cases = [
            (lambda x: 1e308 if x[0] else -1e308, 1e-4),
            (lambda x: smallest*(x[0]/1e154)**2, 1e154),
        ]
        for fn, eps in cases:
            with self.subTest(eps=eps):
                calls = []

                def objective(x):
                    calls.append(tuple(x))
                    return fn(x)

                with self.assertRaises(FloatingPointError):
                    finite_difference_hessian(objective, [0.], eps)
                self.assertEqual(calls, expected_order([0.], eps))

    def test_ordinary_quadratic_retains_predecessor_bits(self):
        actual = finite_difference_hessian(
            lambda x: 3*x[0]**2 + 4*x[0]*x[1] + 5*x[1]**2, [.5, -.25], 1e-3,
        )
        expected = [
            ['0x1.7fffffffd3df2p+2', '0x1.0000000001194p+2'],
            ['0x1.0000000001194p+2', '0x1.3ffffffff9be7p+3'],
        ]
        self.assertEqual([[float(value).hex() for value in row] for row in actual], expected)

    def test_three_dimensions_preserve_order_budget_symmetry_and_input(self):
        point = np.zeros(3)
        before = point.copy()

        def objective(x):
            u, v, w = x/1e154
            value = (2*u*u - 1 + .25*v*w)*1e308
            x[:] = 42
            return value

        actual, calls = capture(objective, point, 1e154)
        np.testing.assert_array_equal(point, before)
        np.testing.assert_array_equal(actual, actual.T)
        self.assertEqual(len(calls), 19)
        self.assertEqual(len({position for position, _ in calls}), 19)
        self.assertEqual([position for position, _ in calls], expected_order(point, 1e154))
        np.testing.assert_allclose(actual, [[4., 0., 0.], [0., 0., .25], [0., .25, 0.]], rtol=1e-14, atol=0)

    def test_objective_keeps_callers_numpy_error_policy_and_policy_is_restored(self):
        before = np.geterr()
        observed = []

        def objective(x):
            observed.append(np.geterr())
            return (2*(x[0]/1e154)**2-1)*1e308

        with np.errstate(all='raise'):
            policy = np.geterr()
            actual = finite_difference_hessian(objective, [0.], 1e154)
            self.assertEqual(np.geterr(), policy)
        self.assertEqual(observed, [policy]*3)
        self.assertEqual(np.geterr(), before)
        np.testing.assert_allclose(actual, [[4.]], rtol=1e-14, atol=0)

    def test_objective_failure_at_first_corner_is_not_retried(self):
        calls = []
        failure = RuntimeError('retained objective failure')

        def objective(x):
            calls.append(tuple(x))
            if len(calls) == 6:
                raise failure
            return 0.

        with self.assertRaises(RuntimeError) as caught:
            finite_difference_hessian(objective, [0., 0.], 1e154)
        self.assertIs(caught.exception, failure)
        self.assertEqual(calls, expected_order([0., 0.], 1e154)[:6])
