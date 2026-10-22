# Prospective amendment: finite Hessian stencil arithmetic

Status at creation: **specified before implementation**. This is a bounded
engineering correction to the numerical prototype, not a new scientific study.

## Exact predecessor and demonstrated failure

The only open RIS pull request is #4 at
`cae6c530f218b0802ab269f99c5272d71f511048`, with tree
`494cc16d57cdef63526439c30c610539639dff67`. Its `main.py` SHA256 is
`b53175a447bb13e17bfc8e3b58fa7b58ff068b4488316f126a2875b1d1d9ea63`.
The existing represented-displacement repair, all 20 test methods and both
earlier research records are retained.

For `s=1e154`, the smooth objective
`f(x)=(2*(x[0]/s)**2-1)*1e308` at `[0.]`, with `eps=s`, returns finite
values `-1e308`, `+1e308`, `+1e308` in three calls. The predecessor rejects
the Hessian because an intermediate subtraction overflows before division by
the large step. The ideal analytic second derivative is 4; the exact stencil
over the represented float64 samples and steps also has finite curvature near
4. Intermediate arithmetic failure is not proof of unrepresentable curvature.

The retained baseline receipt records this counterexample and related
constructed mixed/nonuniform, cancellation and subnormal cases. These are
development observations; earlier passing checks remain passing checks of
their stated, narrower coverage.

## Amendment and operational boundary

Keep the existing public function, float64 conversion, input validation, absolute
step selection and represented positive/negative coordinate distances. In
particular, this amendment does **not** expand admission to steps whose square
is zero or nonfinite, or to ineffective/nonfinite perturbed coordinates.

Keep the original diagonal secant-slope arithmetic and the nonuniform
first-derivative tensor product for mixed entries. The ordinary arithmetic path
must remain bit-identical when all of its operations complete without a range
exception and its result is finite. Range checking belongs only around stencil
arithmetic; do not change the caller's NumPy error policy during objective calls.

If that arithmetic raises an overflow, underflow, invalid-operation or division
exception, or produces a nonfinite result, recompute that **one entry** using
exact rational arithmetic on the already evaluated float64 objective values and
represented step distances. Convert the exact rational result once to float64:

- A finite nonzero representable result is admitted with float64 rounding.
- Exact zero is admitted, including cancellation after an overflowing
  intermediate difference.
- An overflowing result or a nonzero result that rounds to zero is rejected
  with `FloatingPointError`; do not invent a zero Hessian from range loss.

This is a bounded fallback: a diagonal uses three cached values and two
distances; a mixed entry uses nine cached values and four distances. Every value
is already a finite float64, so rational input size and the number of arithmetic
operations per entry are bounded. No adaptive step search, arbitrary-precision
objective evaluation or unbounded refinement is added.

Evaluate the objective exactly once at each existing stencil point. Preserve
center first, positive/negative axis pairs in coordinate order, and mixed corner
order `pp, mp, pm, mm`. A successful `n`-dimensional call still uses
`2*n*n+1` objective calls, including when several entries require fallback.
Rejected inputs still cause zero objective calls. Rejecting an invalid objective
or unrepresentable entry may stop early; no fallback may invoke the objective
again. Preserve caller inputs and exact output symmetry.

## Independent oracles and falsifiers

Use analytic shifted quadratics, affine functions and a cubic with a known
mixed derivative. Independently calculate selected stencils with 800-digit
`decimal.Decimal` Lagrange weights applied to the captured float64 values and
actual coordinate offsets. The high-precision oracle must not call the
implementation's derivative helpers or evaluate the objective again.

Required cases cover large finite diagonal and mixed curvatures, nonuniform
represented steps, exact affine cancellation, subnormal curvature, genuinely
unrepresentable curvature, ordinary behavior controls, objective order/count,
input preservation, unchanged error policy and step-domain admission. Ordinary
controls retain predecessor output hex values; subnormal comparisons use exact
float64 values. Analytic nonzero controls allow relative error at most `1e-14`
to account for rounding in the objective samples; zero controls must remain
exact where the captured values are exactly affine.

The correction fails its gate if a required finite stencil remains rejected,
an unrepresentable result becomes a false finite answer, prior admission or
ordinary arithmetic changes, any objective is repeated/reordered, or any
existing test is weakened. Preserve baseline failures, implementation failures,
review findings, commands, source hashes and measured outcomes.

## Development budget and remaining scientific limits

Use at most 16 new deterministic test methods and 64 constructed fixtures, with
dimension at most three. Planned executions are a baseline check, a candidate
check, an independent review and the existing small hosted CI gate; record the
reason for any corrective rerun. No provider call, paid work, protected
evaluation, training campaign, deployment or default-branch merge is included.

Exact arithmetic on captured samples does not recover information lost inside
the objective, infer a suitable step, bound truncation error or establish a true
derivative for an arbitrary function. It does not authenticate the objective or
establish financial/research performance. RIS remains a numerical prototype,
with `research_complete=false`; previous scientific dispositions and records
remain unchanged. A new scientific study needs its own hypothesis, methodology,
freeze, data/model identities, budget, evidence and release gates.
