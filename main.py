from numbers import Real

import numpy as np


def irr_fn(x):
    return np.sin(x[0]) + np.cos(x[1])


def finite_difference_hessian(fn, x, eps=1e-4):
    """Approximate a real scalar objective's Hessian using centered differences.

    Inputs are copied to float64 before perturbation. ``eps`` is an absolute
    coordinate step; callers must choose it for their objective's scale.
    Stencil denominators use the actual float64 displacements, which can
    differ on the two sides of a coordinate. Only the upper triangle is
    evaluated, then reflected exactly.
    """
    if not callable(fn):
        raise TypeError("fn must be callable")
    values = np.asarray(x)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("x must be a nonempty one-dimensional vector")
    if values.dtype.kind not in "iuf":
        raise TypeError("x must contain real numeric coordinates")
    with np.errstate(over="ignore", invalid="ignore"):
        point = values.astype(np.float64, copy=True)
    if not np.isfinite(point).all():
        raise ValueError("x must be finite and representable as float64")
    if isinstance(eps, (bool, np.bool_)) or not isinstance(eps, Real):
        raise ValueError("eps must be a positive finite real number")
    try:
        step = float(eps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("eps must be a positive finite real number") from exc
    if not np.isfinite(step) or step <= 0:
        raise ValueError("eps must be a positive finite real number")
    squared_step = step * step
    if not np.isfinite(squared_step) or squared_step == 0:
        raise ValueError("eps squared must be representable as float64")
    with np.errstate(over="ignore", invalid="ignore"):
        above, below = point + step, point - step
    if (
        not np.isfinite(above).all()
        or not np.isfinite(below).all()
        or np.any(above == point)
        or np.any(below == point)
    ):
        raise ValueError("eps must produce finite, distinct perturbed coordinates")

    def evaluate(at):
        result = np.asarray(fn(at.copy()))
        if result.ndim != 0 or result.dtype.kind not in "iuf":
            raise ValueError("fn must return one finite real scalar")
        value = float(result)
        if not np.isfinite(value):
            raise ValueError("fn must return one finite real scalar")
        return value

    size = point.size
    hessian = np.empty((size, size), dtype=np.float64)
    center = evaluate(point)
    positive_steps, negative_steps = above - point, point - below
    positive_values, negative_values = [], []
    for i in range(size):
        plus, minus = point.copy(), point.copy()
        plus[i], minus[i] = above[i], below[i]
        positive_values.append(evaluate(plus))
        negative_values.append(evaluate(minus))

    def first_derivative(positive, middle, negative, coordinate):
        hp, hm = positive_steps[coordinate], negative_steps[coordinate]
        # Three-point derivative on a possibly nonuniform coordinate stencil.
        return ((hm / (hp + hm)) * ((positive - middle) / hp)
                + (hp / (hp + hm)) * ((middle - negative) / hm))

    for i in range(size):
        hp, hm = positive_steps[i], negative_steps[i]
        # Difference secant slopes first: affine functions have zero curvature
        # even at a binade boundary where the represented steps are unequal.
        hessian[i, i] = 2 * (
            (positive_values[i] - center) / hp
            - (center - negative_values[i]) / hm
        ) / (hp + hm)
        for j in range(i + 1, size):
            pp, pm, mp, mm = (point.copy() for _ in range(4))
            pp[i], pp[j] = above[i], above[j]
            pm[i], pm[j] = above[i], below[j]
            mp[i], mp[j] = below[i], above[j]
            mm[i], mm[j] = below[i], below[j]
            at_plus = first_derivative(evaluate(pp), positive_values[j], evaluate(mp), i)
            at_zero = first_derivative(positive_values[i], center, negative_values[i], i)
            at_minus = first_derivative(evaluate(pm), negative_values[j], evaluate(mm), i)
            mixed = first_derivative(at_plus, at_zero, at_minus, j)
            hessian[i, j] = hessian[j, i] = mixed
    if not np.isfinite(hessian).all():
        raise FloatingPointError("finite differences produced a non-finite Hessian")
    return hessian


def demo():
    x = np.array([1.0, 2.0])
    print(finite_difference_hessian(irr_fn, x))


if __name__ == "__main__":
    demo()
