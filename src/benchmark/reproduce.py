# =============================================================================
#                     ____        __
#    ____ ___  __  __/ __ )__  __/ /____  _____
#   / __ `__ \/ / / / __  / / / / __/ _ \/ ___/
#  / / / / / / /_/ / /_/ / /_/ / /_/  __(__  )
# /_/ /_/ /_/\__, /_____/\__, /\__/\___/____/
#           /____/      /____/
#
#  myBytes.com
#  Copyright (c) 2026 myBytes GmbH. All rights reserved.
#  Proprietary and confidential.
#
#  File: reproduce.py | Project: soft-commodities-forecast-benchmark | Author: Guido Winger
# =============================================================================

"""
Reproduction harness for the benchmark.

Re-runs train + predict + evaluate for one asset, then asserts the
resulting diagnostics JSON against the version stored in `results/`
within a documented tolerance (see `data_snapshot.json`).

Exits with status code 0 on success, 1 on divergence.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
SNAPSHOT_PATH = REPO_ROOT / "data_snapshot.json"


def _load_snapshot() -> dict:
    with SNAPSHOT_PATH.open() as f:
        return json.load(f)


def _relative_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def _compare_diagnostics(expected: dict, actual: dict, tol: dict) -> list[str]:
    """Return a list of human-readable divergence messages (empty if all clear)."""
    divergences: list[str] = []
    param_tol = tol.get("parameter_relative_diff", 0.01)
    ll_tol = tol.get("loglikelihood_relative_diff", 0.005)

    if (
        "model_params" in expected
        and "model_params" in actual
        and isinstance(expected["model_params"], dict)
    ):
        for name, exp_val in expected["model_params"].items():
            if name not in actual["model_params"]:
                divergences.append(f"missing parameter `{name}` in fresh fit")
                continue
            act_val = actual["model_params"][name]
            try:
                diff = _relative_diff(float(exp_val), float(act_val))
            except (TypeError, ValueError):
                continue
            if diff > param_tol:
                divergences.append(
                    f"parameter `{name}` drifted: expected {exp_val:.6g}, got {act_val:.6g} "
                    f"(relative diff {diff:.3%}, tolerance {param_tol:.1%})"
                )

    if "loglikelihood" in expected and "loglikelihood" in actual:
        try:
            diff = _relative_diff(
                float(expected["loglikelihood"]), float(actual["loglikelihood"])
            )
            if diff > ll_tol:
                divergences.append(
                    f"log-likelihood drifted: expected {expected['loglikelihood']:.4f}, "
                    f"got {actual['loglikelihood']:.4f} "
                    f"(relative diff {diff:.3%}, tolerance {ll_tol:.1%})"
                )
        except (TypeError, ValueError):
            pass

    return divergences


def _run_step(module: str, asset: str) -> None:
    logger.info(f"→ running {module} for {asset}")
    result = subprocess.run(
        [sys.executable, "-m", module, "--asset", asset],
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        logger.error(f"{module} failed for {asset}")
        sys.exit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce one asset's benchmark.")
    parser.add_argument("--asset", required=True, choices=["cocoa", "coffee", "sugar", "cotton"])
    parser.add_argument(
        "--skip-fit",
        action="store_true",
        help="skip train/predict/evaluate; only compare existing diagnostics",
    )
    args = parser.parse_args()

    asset = args.asset
    expected_path = RESULTS_DIR / f"{asset}_diagnostics.json"
    if not expected_path.exists():
        logger.error(f"no stored diagnostics at {expected_path}")
        return 1

    with expected_path.open() as f:
        expected = json.load(f)

    if not args.skip_fit:
        _run_step("benchmark.train", asset)
        _run_step("benchmark.predict", asset)
        _run_step("benchmark.evaluate", asset)

    actual_path = REPO_ROOT / "artifacts" / f"gjr_garch_t_{asset}_diagnostics.json"
    if not actual_path.exists():
        logger.error(f"fresh fit did not produce {actual_path}")
        return 1

    with actual_path.open() as f:
        actual = json.load(f)

    snapshot = _load_snapshot()
    tol = snapshot.get("tolerance", {})

    divergences = _compare_diagnostics(expected, actual, tol)

    if divergences:
        logger.error(f"reproduction divergences for {asset}:")
        for d in divergences:
            logger.error(f"  • {d}")
        return 1

    logger.success(f"reproduction match for {asset} — parameters and log-likelihood within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())