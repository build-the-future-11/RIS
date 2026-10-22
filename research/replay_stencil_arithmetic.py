"""Replay the retained float64 stencils without making new objective measurements."""

import argparse
import hashlib
import importlib.util
import json
import platform
import warnings
from pathlib import Path

import numpy as np


def replay(source, baseline):
    source_bytes, baseline_bytes = source.read_bytes(), baseline.read_bytes()
    specification = importlib.util.spec_from_file_location("ris_stencil_under_test", source)
    module = importlib.util.module_from_spec(specification)
    exec(compile(source_bytes, str(source), "exec"), module.__dict__)
    captured = json.loads(baseline_bytes)
    outcomes = []
    for case in captured["scenarios"]:
        samples = {
            tuple(float.fromhex(value) for value in row["point_hex"]): float.fromhex(row["value_hex"])
            for row in case["calls"]
        }
        calls = []

        def cached_objective(point):
            position = tuple(float(value) for value in point)
            calls.append(position)
            return samples[position]

        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter("always")
            try:
                result = module.finite_difference_hessian(cached_objective, case["point"], case["eps"])
                outcome = {
                    "hessian": result.tolist(),
                    "hessian_hex": [[float(value).hex() for value in row] for row in result],
                }
            except Exception as exc:
                outcome = {"error_type": type(exc).__name__, "error": str(exc)}
        expected = [tuple(float.fromhex(value) for value in row["point_hex"]) for row in case["calls"]]
        outcomes.append({
            "name": case["name"], "outcome": outcome, "call_count": len(calls),
            "same_call_order_as_baseline": calls == expected,
            "same_output_bits_as_baseline": (
                outcome.get("hessian_hex") == case["outcome"].get("hessian_hex")
                if "hessian_hex" in outcome and "hessian_hex" in case["outcome"] else None
            ),
            "warnings": [str(notice.message) for notice in notices],
        })
    return {
        "schema": "ris.stencil-arithmetic.replay.v1",
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__,
        "scope": "Replay retained float64 samples; no new objective measurements or scientific run.",
        "scenarios": outcomes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--source", type=Path, default=root.parent / "main.py")
    parser.add_argument("--baseline", type=Path, default=root / "STENCIL_ARITHMETIC_BASELINE_20261010.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    content = json.dumps(replay(args.source, args.baseline), indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
