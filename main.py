
import numpy as np

def irr_fn(x):
    return np.sin(x[0]) + np.cos(x[1])

def finite_difference_hessian(fn, x, eps=1e-4):
    n = len(x)
    h = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            x1 = x.copy()
            x2 = x.copy()
            x3 = x.copy()
            x4 = x.copy()

            x1[i] += eps
            x1[j] += eps

            x2[i] += eps
            x2[j] -= eps

            x3[i] -= eps
            x3[j] += eps

            x4[i] -= eps
            x4[j] -= eps

            h[i, j] = (
                fn(x1)
                - fn(x2)
                - fn(x3)
                + fn(x4)
            ) / (4 * eps * eps)

    return h

x = np.array([1.0, 2.0])

print(finite_difference_hessian(irr_fn, x))
