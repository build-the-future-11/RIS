# RIS prototype

RIS currently contains a **minimal numerical prototype**, not a production or institutional-grade financial system.

## Implemented today

The repository has one Python entry point, `main.py`, which:

- defines a two-variable toy objective using sine and cosine;
- approximates its Hessian with centered finite differences;
- evaluates the Hessian at a fixed example point;
- prints the resulting matrix.

The current code uses NumPy. A broader dependency list is retained in `requirements.txt`, but those packages are **not evidence that the corresponding framework features have been implemented**.

## Run the current prototype

```bash
python -m pip install 'numpy>=1.26,<3'
python main.py
```

The repository CI performs an exact-source checkout, compiles the Python entry point, installs the bounded NumPy runtime dependency, and executes the current demo.

## Current evidence boundary

This repository does **not** currently establish:

- an institutional-grade risk or IRR stabilization platform;
- production financial infrastructure;
- validated forecasting or optimization performance;
- trading or investment returns;
- a complete research benchmark or paper result.

Any future claims should be tied to checked-in implementation, tests, retained evaluation artifacts, and a clearly defined financial/research protocol.

## Numerical API contract

`finite_difference_hessian(fn, x, eps=1e-4)` accepts a nonempty real coordinate
vector, including integer arrays and Python lists. It copies coordinates to float64
before perturbation; integer storage must never truncate a finite-difference step.
The objective must return one finite real scalar. Invalid coordinates, nonpositive
or ineffective steps, nonscalar outputs, and non-finite curvature raise an error.

```python
import numpy as np
from main import finite_difference_hessian

point = np.array([1, 2], dtype=np.int64)
hessian = finite_difference_hessian(
    lambda x: 3*x[0]**2 + 4*x[0]*x[1] + 5*x[1]**2,
    point,
    eps=1e-3,
)
# Approximately [[6, 4], [4, 10]]; point is unchanged.
```

The implementation uses centered diagonal and mixed stencils, shares the center
evaluation, and computes one triangle before reflecting it. A successful call in
`n` dimensions uses `2*n*n + 1` objective evaluations; the two-dimensional example
uses 9 instead of the former 16. This is still a finite-difference approximation:
`eps` is an absolute coordinate step, and callers must choose it for the scale and
smoothness of their objective. Input validation does not estimate truncation or
cancellation error. The regression suite checks analytic quadratic/sinusoidal
references, real scalar and finite-value admission, input preservation, and large
constant offsets. These are numerical prototype tests, not financial validation.
