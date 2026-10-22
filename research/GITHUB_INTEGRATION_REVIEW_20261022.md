# RIS complete-descendant integration review — 2026-10-22

## Source and ancestry

PR 5 head `49d167b387981662450a1f8c475f28f4c9d9fe5a`, actual commit tree `784240a52f4a48103fc70f75d048c211d616bfa6`, contains PR 4 head `cae6c530f218b0802ab269f99c5272d71f511048` in full. GitHub compare reports ahead 3/behind 0 with merge-base exactly PR 4. It also contains current main `ddfcb5e532d00362cdd5cf472799a01cb9464a72` (ahead 6/behind 0, merge-base exactly main). PR 5 already targets main. Integrating the reviewed descendant preserves both PR histories and all code; no divergent source selection or extra merge parent is needed.

The review read AGENTS.md, canonical state, every required-session document, the complete main.py diff, all five test modules, replay entry point, README, dependency file and unchanged workflow. All 18 retrieved leaves were checked against their Git blob identities and modes. Six source/evidence SHA256 values from the retained verification report and five embedded independent-review artifact digests also matched. The actual tree identity comes from commit metadata, not a tree endpoint requested by commit ref.

Both PR 4 and PR 5 have no submitted GitHub reviews or inline threads at this checkpoint. This review does not invent a formal external approval or waive any required review/gate.

## Numerical review

The represented-step diagonal computes twice the difference between positive and negative secant slopes divided by the sum of actual distances. The mixed entry composes nonuniform first-derivative weights over the captured 3x3 stencil. At ordinary finite arithmetic, the previous operation sequence is retained. Range checking surrounds stencil arithmetic and restores the caller's NumPy error policy; objective evaluation occurs outside that context.

On a detected range failure, the same fixed stencil is evaluated with exact rational values from the captured binary64 samples and actual distances. Only its final value is rounded to float64. The code refuses final overflow and nonzero curvature that rounds to zero, while allowing exact cancellation. The fallback never invokes the objective. There is one shared center, 2n axis calls and 4 calls for each unordered pair, totaling `2*n*n+1` successful calls, with exact reflected symmetry and protected caller inputs. Invalid objectives or unrepresentable entries may terminate early under the retained prospective contract.

The 12 new methods exercise large and subnormal curvature, asymmetric represented distances, affine cancellation, genuine overflow/nonzero underflow rejection, ordinary bit identity, three-dimensional 19-call order, caller policy and exception preservation. Their 800-digit Decimal Lagrange-basis reference uses captured samples and does not call the implementation's helpers. All 20 preceding methods and tolerances are retained. The historical baseline failures, three assertion failures/eight error events across ten affected methods, remain in the original evidence. The historical independent six probes and eleven captured-value replays are not counted as newly executed evidence in this review.

No concrete blocker was found within this bounded numerical contract. Absolute step selection, truncation error, cancellation already inside the objective and conversion to the stated float64 domain remain limitations. The helper is not a scientific/financial validation system.

## Exact-head hosted verification

Readback of [Python smoke 38031777245](https://github.com/build-the-future-11/RIS/actions/runs/38031777245), job 114154052480, binds the successful run to exact head `49d167b387981662450a1f8c475f28f4c9d9fe5a` and PR 5 targeting main. The source assertion, compilation, NumPy installation, all 32 numerical methods (0.038 seconds), and existing demo passed. Every job step reports success. The complete log and normalized run/job/ancestry receipt are retained in `research/integration_20261022/`.

No test or captured-value replay was rerun locally in this review: immutable exact-head hosted results plus source/evidence inspection already establish the relevant unchanged code checks. This avoids presenting repeated methods as extra coverage or reopening prior budgets. The newly published metadata/documentation checkpoint must still pass the existing unweakened exact-head Python smoke gate before integration.

## Checkpoint changes and preserved research boundary

Only canonical state, this append-only integration record/evidence, and one stale README statement change. `requirements.txt` already declares only NumPy; README previously said a broader dependency list was retained and now reports the actual dependency. Runtime source, all test modules, dependencies, workflow and original research records remain byte-identical to the reviewed head. The exact prior canonical state is retained, and its historical pending statements are kept as dated history.

The current disposition remains `NUMERICAL_PROTOTYPE; no completed scientific study asserted` and `research_complete=false`. No objective measurements, new fixtures, provider calls, protected outcomes, paid work or scientific campaigns occurred in this review. Original failed records and prospective boundaries remain authoritative. Future scientific work needs a separately specified protocol and budget. GitHub integration of these engineering repairs does not itself complete that future work.
