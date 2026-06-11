# =========================================================================
#                     ____        __
#    ____ ___  __  __/ __ )__  __/ /____  _____
#   / __ `__ \/ / / / __  / / / / __/ _ \/ ___/
#  / / / / / / /_/ / /_/ / /_/ / /_/  __(__  )
# /_/ /_/ /_/\__, /_____/\__, /\__/\___/____/
#           /____/      /____/
#
#  myBytes.com
#  Copyright (c) 2026 myBytes GmbH. All rights reserved.
#
#  File: hmm_evaluate.py | Project: soft-commodities-forecast-benchmark
#  Author: Guido Winger
# =========================================================================

"""Layer-2 detection evaluation at the pre-registered endpoint.

Primary signal mechanics ("ratio_multiplier_on_logit", see
configs/global.yaml): we operationalise the logit-ratio as a ratio in
ODDS space - odds_t = p_t / (1 - p_t). A multiplicative threshold on
odds is exactly an additive threshold on the logit (the ratio in odds
equals exp of the logit difference), is strictly monotone in the
posterior, and keeps the signal positive and unbounded - the same
shape the vol-signal calibration machinery expects. Calibration and
lead-time mechanics are byte-identical to layer 1:
``calibrate_detection_threshold(_pooled)`` with EMA(lambda = 0.94)
baseline, sensitive operating point (10 % FPR) on the crisis-free
validation years, 180d lookback, pooled across the four assets.

Sensitivity reporting (pre-registered fallback): fixed posterior
thresholds 0.5 and 0.9 via ``detection_lead_time_probability``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.metrics import (
    calibrate_detection_threshold,
    calibrate_detection_threshold_pooled,
    detection_lead_time,
    detection_lead_time_probability,
)
from benchmark.preprocessing import load_global_config

RESULTS_DIR = Path("results")
ASSETS = ["cocoa", "coffee", "sugar", "cotton"]
EPS = 1e-6


def posterior_to_odds(p: pd.Series) -> pd.Series:
    """Odds transform, clipped away from 0/1 for numerical stability."""
    q = p.clip(EPS, 1.0 - EPS)
    return q / (1.0 - q)


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer-2 HMM detection evaluation.")
    parser.add_argument("--global-config", type=str, default="configs/global.yaml")
    args = parser.parse_args()

    gcfg = load_global_config(args.global_config)
    bs = gcfg["metrics"]["black_swan_detection"]
    splits = gcfg["splits"]
    lookbacks = list(bs["lead_time_lookback_windows_days"])
    operating_points = bs["threshold_calibration"]["operating_points"]
    ema_lambda = 0.94
    baseline_window = 252

    # Load all posteriors, build odds signals
    odds: dict[str, pd.Series] = {}
    for asset in ASSETS:
        post = pd.read_parquet(RESULTS_DIR / f"hmm_posterior_{asset}.parquet")[
            "posterior_stress"
        ]
        odds[asset] = posterior_to_odds(post)

    # Validation slices (crisis-free by construction, as in layer 1)
    val = {
        a: s.loc[splits["val_start"] : splits["val_end"]] for a, s in odds.items()
    }

    report: dict = {
        "model": "gaussian_hmm_k3",
        "signal": "posterior_stress (odds-space, ratio_multiplier_on_logit)",
        "endpoint_note": (
            "Primary: ema_lambda094 / sensitive (10% FPR) / 180d / pooled - "
            "identical to layer 1. Secondary combinations flagged exploratory."
        ),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "by_calibration": {},
        "fixed_threshold_sensitivity": {},
    }

    for calib_name in ["pooled", "single"]:
        block: dict = {}
        for op in operating_points:
            op_name, fpr = op["name"], float(op["target_fpr"])
            if calib_name == "pooled":
                cal = calibrate_detection_threshold_pooled(
                    val, target_fpr=fpr, baseline_window=baseline_window,
                    baseline_statistic="ema", ema_lambda=ema_lambda,
                )
                multipliers = {a: cal["multiplier"] for a in ASSETS}
                cal_info = {k: v for k, v in cal.items() if not isinstance(v, pd.Series)}
            else:
                multipliers, cal_info = {}, {}
                for a in ASSETS:
                    c = calibrate_detection_threshold(
                        val[a], target_fpr=fpr, baseline_window=baseline_window,
                        baseline_statistic="ema", ema_lambda=ema_lambda,
                    )
                    multipliers[a] = c["multiplier"]
                    cal_info[a] = {k: v for k, v in c.items() if not isinstance(v, pd.Series)}

            events_out = {}
            for event in bs["crisis_events"]:
                a = event["asset"]
                per_lb = {}
                for lb in lookbacks:
                    per_lb[f"{lb}d"] = detection_lead_time(
                        risk_signal=odds[a],
                        crisis_start_date=event["start_date"],
                        threshold_multiplier=multipliers[a],
                        baseline_window=baseline_window,
                        baseline_statistic="ema",
                        ema_lambda=ema_lambda,
                        lookback_days=lb,
                    )
                events_out[event["name"]] = {
                    "asset": a, "lookback_windows": per_lb,
                    "multiplier": float(multipliers[a]),
                }
            block[op_name] = {"calibration": cal_info, "events": events_out}
        report["by_calibration"][calib_name] = block

    # Pre-registered sensitivity: fixed posterior thresholds
    for thr in [0.5, 0.9]:
        per_event = {}
        for event in bs["crisis_events"]:
            a = event["asset"]
            post = pd.read_parquet(RESULTS_DIR / f"hmm_posterior_{a}.parquet")[
                "posterior_stress"
            ]
            per_event[event["name"]] = detection_lead_time_probability(
                signal=post,
                crisis_start_date=event["start_date"],
                probability_threshold=thr,
                lookback_days=180,
            )
        report["fixed_threshold_sensitivity"][f"p>{thr}"] = per_event

    out = RESULTS_DIR / "hmm_detection_evaluation.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"written: {out}")

    # Console summary: PRIMARY endpoint
    print("\nPRIMARY (ema_lambda094 / sensitive / 180d / pooled):")
    prim = report["by_calibration"]["pooled"]["sensitive"]["events"]
    for name, ev in prim.items():
        lt = ev["lookback_windows"]["180d"]
        print(f"  {ev['asset']:8s} {name}: lead={lt['lead_time_days']}d det={lt.get('detection_date')}")


if __name__ == "__main__":
    main()
