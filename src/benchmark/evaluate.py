"""
GJR-GARCH(1,1,1) Student-t — Evaluation on val + test.

Reads the parquet produced by ``predict.py`` and produces a multi-layer
evaluation report:

- Volatility-forecast metrics (QLIKE, MSE, MAE) and two flavours of R²_OOS:
  vs. unconditional mean (Campbell-Thompson) and vs. a 5-day rolling-mean
  HAR-RV-style benchmark.
- VaR backtests at 95% and 99% on the full split (Kupiec POF, Christoffersen
  CC) using the per-cohort Student-t ν.
- Pre-Crisis VaR backtests on a 180–365 day window before each known
  crisis event — measures whether the model gave reasonable coverage in the
  early-warning period, separate from the full-sample (backward-looking)
  number.
- Black-swan detection lead time over multiple lookback windows
  (30/90/180 calendar days) and across two baselines (median 252d, EMA
  λ=0.94). 12 lead-time numbers per event per asset → robustness diagnostic.

Everything is logged to MLflow and dumped as JSON next to the forecast.

Usage:
    python -m models.01_gjr_garch_t.evaluate --asset cocoa
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from loguru import logger

from benchmark.metrics import (
    calibrate_detection_threshold,
    calibrate_detection_threshold_pooled,
    christoffersen_cc_test,
    detection_lead_time,
    kupiec_pof_test,
    mae_variance,
    mse_variance,
    pre_crisis_slice,
    qlike,
    r2_oos,
    rolling_mean_benchmark,
    var_from_vol,
)
from benchmark.preprocessing import load_global_config


def _load_pool_val_signals(
    results_dir: Path,
    asset_self: str,
    pool_assets: list[str],
) -> dict[str, pd.Series]:
    """Load val-split cond_vol for each pool asset whose forecast parquet exists."""
    signals: dict[str, pd.Series] = {}
    for asset in pool_assets:
        path = results_dir / f"forecast_{asset}.parquet"
        if not path.exists():
            logger.warning(f"Pool: forecast missing for {asset!r} → {path} (skipping)")
            continue
        f = pd.read_parquet(path)
        val = f.loc[f["split"] == "val", "cond_vol"].dropna()
        if val.empty:
            continue
        signals[asset] = val
    return signals


# ==============================================================================
# Volatility metrics
# ==============================================================================
def _vol_metrics(
    realized_var: np.ndarray,
    forecast_var: np.ndarray,
    rolling_window: int,
) -> dict:
    benchmark = rolling_mean_benchmark(realized_var, window=rolling_window)
    return {
        "qlike": qlike(realized_var, forecast_var),
        "mse": mse_variance(realized_var, forecast_var),
        "mae": mae_variance(realized_var, forecast_var),
        "r2_oos": r2_oos(realized_var, forecast_var),
        "r2_oos_vs_har_rv": r2_oos(realized_var, forecast_var, benchmark=benchmark),
    }


# ==============================================================================
# VaR backtests
# ==============================================================================
def _build_var_series(
    cond_vol: np.ndarray,
    nu: np.ndarray,
    mu: np.ndarray,
    confidence: float,
) -> np.ndarray:
    """Student-t VaR (per refit cohort ν), vectorised across unique ν."""
    var_series = np.empty_like(cond_vol)
    for nu_value in np.unique(nu):
        mask = nu == nu_value
        var_series[mask] = var_from_vol(
            forecast_vol=cond_vol[mask],
            confidence=confidence,
            distribution="studentst",
            df=float(nu_value),
            mean_return=float(np.mean(mu[mask])),
        )
    return var_series


def _var_backtests(
    returns: np.ndarray,
    cond_vol: np.ndarray,
    nu: np.ndarray,
    mu: np.ndarray,
    confidences: list[float],
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for conf in confidences:
        var_series = _build_var_series(cond_vol, nu, mu, conf)
        kup = kupiec_pof_test(returns, var_series, confidence=conf)
        chr_ = christoffersen_cc_test(returns, var_series, confidence=conf)
        out[f"var_{int(conf * 100)}"] = {
            "kupiec_pof": kup.to_dict(),
            "christoffersen_cc": chr_.to_dict(),
        }
    return out


def _pre_crisis_var_backtests(
    forecast: pd.DataFrame,
    crisis_events: list[dict],
    confidences: list[float],
    window_days: tuple[int, int],
) -> dict:
    """
    Slice the forecast to the pre-crisis window per event, then run the
    standard VaR backtests on that slice. Reports a separate Kupiec /
    Christoffersen number, isolating early-warning coverage from the
    aggregate test.
    """
    result: dict = {}
    for event in crisis_events:
        start, end = pre_crisis_slice(
            forecast.index,
            crisis_start_date=event["start_date"],
            window_days=window_days,
        )
        sub = forecast.loc[start:end].dropna(subset=["cond_vol", "return"])
        if len(sub) < 30:
            result[event["name"]] = {
                "window_start": str(start.date()),
                "window_end": str(end.date()),
                "n_obs": int(len(sub)),
                "note": "insufficient observations in pre-crisis window",
            }
            continue
        result[event["name"]] = {
            "window_start": str(start.date()),
            "window_end": str(end.date()),
            "n_obs": int(len(sub)),
            "backtests": _var_backtests(
                returns=sub["return"].values,
                cond_vol=sub["cond_vol"].values,
                nu=sub["nu"].values,
                mu=sub["mu"].values,
                confidences=confidences,
            ),
        }
    return result


# ==============================================================================
# Detection: calibrate → multi-window lead time × multi-baseline
# ==============================================================================
def _detection_block(
    forecast: pd.DataFrame,
    global_config: dict,
    asset: str,
    results_dir: Path | None = None,
) -> dict:
    cfg = global_config["metrics"]["black_swan_detection"]
    cal_cfg = cfg["threshold_calibration"]
    baselines = cal_cfg["baselines"]
    operating_points = cal_cfg["operating_points"]
    lookback_windows = list(cfg["lead_time_lookback_windows_days"])
    crisis_events = [e for e in cfg["crisis_events"] if e["asset"] == asset]

    val_signal = forecast.loc[forecast["split"] == "val", "cond_vol"]
    test_signal = forecast.loc[forecast["split"] == "test", "cond_vol"]

    pool_cfg = cal_cfg.get("pool", {})
    pool_enabled = bool(pool_cfg.get("enabled", False)) and results_dir is not None
    pool_signals: dict[str, pd.Series] = {}
    if pool_enabled:
        pool_signals = _load_pool_val_signals(
            results_dir=results_dir,
            asset_self=asset,
            pool_assets=list(pool_cfg.get("pool_assets", [])),
        )
        if asset not in pool_signals:
            pool_signals[asset] = val_signal.dropna()

    baseline_blocks: dict = {}
    for bl in baselines:
        bl_name = bl["name"]
        bl_stat = bl["statistic"]
        bl_window = int(bl["window_days"])
        bl_ema = float(bl.get("ema_lambda", 0.94))

        ops_block: dict = {}
        for op in operating_points:
            cal = calibrate_detection_threshold(
                signal_val=val_signal,
                target_fpr=float(op["target_fpr"]),
                baseline_window=bl_window,
                baseline_statistic=bl_stat,
                ema_lambda=bl_ema,
            )
            m = cal["multiplier"]
            events_block: dict = {}
            for event in crisis_events:
                event_block: dict = {"lookback_windows": {}}
                for lb_days in lookback_windows:
                    if not np.isfinite(m):
                        event_block["lookback_windows"][f"{lb_days}d"] = {
                            "lead_time_days": np.nan,
                            "note": "no multiplier within FPR grid",
                        }
                        continue
                    lead = detection_lead_time(
                        risk_signal=test_signal,
                        crisis_start_date=event["start_date"],
                        threshold_multiplier=m,
                        baseline_window=bl_window,
                        baseline_statistic=bl_stat,
                        ema_lambda=bl_ema,
                        lookback_days=int(lb_days),
                    )
                    event_block["lookback_windows"][f"{lb_days}d"] = lead
                events_block[event["name"]] = event_block

            ops_block[op["name"]] = {"calibration": cal, "events": events_block}
        baseline_blocks[bl_name] = ops_block

    # 3f: sensitivity — lead-time delta between baselines for the same
    # operating point / event / lookback. NaN-safe.
    sensitivity = _baseline_sensitivity(baseline_blocks, baselines, operating_points)

    pooled_blocks: dict = {}
    pool_vs_single: dict = {}
    if pool_enabled and pool_signals:
        for bl in baselines:
            bl_name, bl_stat = bl["name"], bl["statistic"]
            bl_window = int(bl["window_days"])
            bl_ema = float(bl.get("ema_lambda", 0.94))
            ops_block: dict = {}
            for op in operating_points:
                cal = calibrate_detection_threshold_pooled(
                    signals_by_asset=pool_signals,
                    target_fpr=float(op["target_fpr"]),
                    baseline_window=bl_window,
                    baseline_statistic=bl_stat,
                    ema_lambda=bl_ema,
                )
                m = cal["multiplier"]
                m_single = baseline_blocks[bl_name][op["name"]]["calibration"]["multiplier"]
                pool_vs_single.setdefault(bl_name, {})[op["name"]] = {
                    "multiplier_single": m_single,
                    "multiplier_pooled": m,
                    "delta_abs": (m - m_single)
                    if (np.isfinite(m_single) and np.isfinite(m))
                    else np.nan,
                    "delta_pct": (
                        100.0 * (m - m_single) / m_single
                        if (np.isfinite(m_single) and np.isfinite(m) and m_single != 0)
                        else np.nan
                    ),
                }

                events_block: dict = {}
                for event in crisis_events:
                    event_block: dict = {"lookback_windows": {}}
                    for lb_days in lookback_windows:
                        if not np.isfinite(m):
                            event_block["lookback_windows"][f"{lb_days}d"] = {
                                "lead_time_days": np.nan,
                                "note": "no multiplier within FPR grid (pooled)",
                            }
                            continue
                        lead = detection_lead_time(
                            risk_signal=test_signal,
                            crisis_start_date=event["start_date"],
                            threshold_multiplier=m,
                            baseline_window=bl_window,
                            baseline_statistic=bl_stat,
                            ema_lambda=bl_ema,
                            lookback_days=int(lb_days),
                        )
                        event_block["lookback_windows"][f"{lb_days}d"] = lead
                    events_block[event["name"]] = event_block
                ops_block[op["name"]] = {"calibration": cal, "events": events_block}
            pooled_blocks[bl_name] = ops_block

    return {
        "lookback_windows_days": lookback_windows,
        "baselines": {b["name"]: b for b in baselines},
        "by_baseline": baseline_blocks,
        "baseline_sensitivity": sensitivity,
        "pool_enabled": pool_enabled,
        "pool_assets_loaded": sorted(pool_signals.keys()) if pool_enabled else [],
        "by_baseline_pooled": pooled_blocks,
        "pool_vs_single_multiplier": pool_vs_single,
    }


def _baseline_sensitivity(
    baseline_blocks: dict,
    baselines: list[dict],
    operating_points: list[dict],
) -> dict:
    """For each (op, event, lookback), compute |lead_time_A - lead_time_B| across baselines."""
    if len(baselines) < 2:
        return {"note": "only one baseline configured; no sensitivity"}
    a_name, b_name = baselines[0]["name"], baselines[1]["name"]
    out: dict = {"baseline_a": a_name, "baseline_b": b_name, "comparisons": {}}
    for op in operating_points:
        op_name = op["name"]
        op_a = baseline_blocks[a_name][op_name]
        op_b = baseline_blocks[b_name][op_name]
        # Multipliers may differ — report them too.
        comp = {
            "multiplier_a": op_a["calibration"]["multiplier"],
            "multiplier_b": op_b["calibration"]["multiplier"],
            "events": {},
        }
        for event_name, event_block_a in op_a["events"].items():
            event_block_b = op_b["events"].get(event_name, {})
            per_window = {}
            for lb_key, lead_a in event_block_a["lookback_windows"].items():
                lead_b = event_block_b.get("lookback_windows", {}).get(lb_key, {})
                lt_a = lead_a.get("lead_time_days", np.nan)
                lt_b = lead_b.get("lead_time_days", np.nan)
                delta = (
                    float(lt_a) - float(lt_b)
                    if np.isfinite(lt_a) and np.isfinite(lt_b)
                    else np.nan
                )
                per_window[lb_key] = {
                    "lead_a": lt_a,
                    "lead_b": lt_b,
                    "delta_days": delta,
                }
            comp["events"][event_name] = per_window
        out["comparisons"][op_name] = comp
    return out


# ==============================================================================
# Main evaluation per asset
# ==============================================================================
def evaluate_single_asset(
    asset: str,
    global_config: dict,
    model_config: dict,
    results_dir: Path,
) -> dict:
    forecast_path = results_dir / f"forecast_{asset}.parquet"
    if not forecast_path.exists():
        raise FileNotFoundError(
            f"No forecast for {asset!r} — run predict.py first ({forecast_path})."
        )

    forecast = pd.read_parquet(forecast_path)
    forecast = forecast.dropna(subset=["cond_vol", "return"])

    metrics_cfg = global_config["metrics"]
    confidences = list(metrics_cfg["var_backtest"]["confidence_levels"])
    rolling_window = int(metrics_cfg.get("rolling_mean_benchmark_window", 5))

    report: dict = {"asset": asset, "splits": {}}
    for split_name in ("val", "test"):
        sub = forecast[forecast["split"] == split_name]
        if sub.empty:
            continue
        realised_var = (sub["return"].values) ** 2
        forecast_var = sub["cond_var"].values

        report["splits"][split_name] = {
            "n_obs": int(len(sub)),
            "vol_metrics": _vol_metrics(realised_var, forecast_var, rolling_window),
            "var_backtests": _var_backtests(
                returns=sub["return"].values,
                cond_vol=sub["cond_vol"].values,
                nu=sub["nu"].values,
                mu=sub["mu"].values,
                confidences=confidences,
            ),
        }

    bs_cfg = metrics_cfg["black_swan_detection"]
    crisis_events_asset = [e for e in bs_cfg["crisis_events"] if e["asset"] == asset]
    pre_crisis_window = tuple(bs_cfg["pre_crisis_var_window_days"])  # (180, 365)

    # Pre-crisis VaR coverage uses test-split forecast (events live in test).
    test_forecast = forecast[forecast["split"] == "test"]
    report["pre_crisis_var_coverage"] = _pre_crisis_var_backtests(
        forecast=test_forecast,
        crisis_events=crisis_events_asset,
        confidences=confidences,
        window_days=pre_crisis_window,
    )

    report["black_swan_detection"] = _detection_block(
        forecast, global_config, asset, results_dir=results_dir
    )

    out_path = results_dir / f"evaluation_{asset}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.success(f"✓ Evaluation persisted: {out_path}")
    return report


# ==============================================================================
# MLflow logging
# ==============================================================================
def _safe_metric(name: str, value) -> None:
    if value is None:
        return
    try:
        v = float(value)
    except (TypeError, ValueError):
        return
    if np.isfinite(v):
        mlflow.log_metric(name, v)


def _log_to_mlflow(report: dict) -> None:
    for split_name, payload in report["splits"].items():
        for metric, value in payload["vol_metrics"].items():
            _safe_metric(f"{split_name}_{metric}", value)
        for var_key, tests in payload["var_backtests"].items():
            for test_name, result in tests.items():
                _safe_metric(f"{split_name}_{var_key}_{test_name}_pvalue", result.get("pvalue"))
                _safe_metric(
                    f"{split_name}_{var_key}_{test_name}_violations",
                    result.get("n_violations"),
                )

    for event_name, payload in report.get("pre_crisis_var_coverage", {}).items():
        if "backtests" not in payload:
            continue
        for var_key, tests in payload["backtests"].items():
            for test_name, result in tests.items():
                _safe_metric(
                    f"precrisis_{event_name}_{var_key}_{test_name}_pvalue",
                    result.get("pvalue"),
                )
                _safe_metric(
                    f"precrisis_{event_name}_{var_key}_{test_name}_violations",
                    result.get("n_violations"),
                )

    det = report["black_swan_detection"]
    for bl_name, ops in det["by_baseline"].items():
        for op_name, op_payload in ops.items():
            m = op_payload["calibration"].get("multiplier")
            _safe_metric(f"mult_{bl_name}_{op_name}", m)
            _safe_metric(
                f"realised_fpr_{bl_name}_{op_name}",
                op_payload["calibration"].get("realised_fpr"),
            )
            for event_name, event_block in op_payload["events"].items():
                for lb_key, lead in event_block["lookback_windows"].items():
                    _safe_metric(
                        f"lead_{bl_name}_{op_name}_{event_name}_{lb_key}",
                        lead.get("lead_time_days"),
                    )

    sens = det.get("baseline_sensitivity", {})
    for op_name, comp in sens.get("comparisons", {}).items():
        for event_name, per_window in comp["events"].items():
            for lb_key, vals in per_window.items():
                _safe_metric(
                    f"sens_{op_name}_{event_name}_{lb_key}_delta",
                    vals.get("delta_days"),
                )

    for bl_name, ops in det.get("by_baseline_pooled", {}).items():
        for op_name, op_payload in ops.items():
            m = op_payload["calibration"].get("multiplier")
            _safe_metric(f"pooled_mult_{bl_name}_{op_name}", m)
            _safe_metric(
                f"pooled_fpr_{bl_name}_{op_name}",
                op_payload["calibration"].get("realised_fpr"),
            )
            for event_name, event_block in op_payload["events"].items():
                for lb_key, lead in event_block["lookback_windows"].items():
                    _safe_metric(
                        f"pooled_lead_{bl_name}_{op_name}_{event_name}_{lb_key}",
                        lead.get("lead_time_days"),
                    )
    for bl_name, ops in det.get("pool_vs_single_multiplier", {}).items():
        for op_name, vals in ops.items():
            _safe_metric(f"pool_vs_single_pct_{bl_name}_{op_name}", vals.get("delta_pct"))


# ==============================================================================
# Entrypoint
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GJR-GARCH-t forecasts.")
    parser.add_argument("--config", type=str, default="models/01_gjr_garch_t/config.yaml")
    parser.add_argument("--global-config", type=str, default="configs/global.yaml")
    parser.add_argument("--asset", type=str, default=None)
    args = parser.parse_args()

    global_config = load_global_config(args.global_config)
    with open(args.config) as f:
        model_config = yaml.safe_load(f)

    assets = [args.asset] if args.asset else model_config["data"]["assets"]
    cfg_path = Path(args.config)
    results_dir = cfg_path.parent / "results"

    mlflow_cfg = global_config["mlflow"]
    mlflow.set_tracking_uri(
        mlflow_cfg["tracking_uri_local"]
        if mlflow_cfg.get("default_mode", "local") == "local"
        else mlflow_cfg["tracking_uri_remote"]
    )

    for asset in assets:
        experiment_name = (
            f"{asset}_volatility_{mlflow_cfg['experiment_prefix']}"
            f"{model_config['mlflow']['experiment_suffix']}"
        )
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"evaluate_{asset}"):
            mlflow.set_tag("phase", "evaluate")
            mlflow.set_tag("asset", asset)
            report = evaluate_single_asset(
                asset=asset,
                global_config=global_config,
                model_config=model_config,
                results_dir=results_dir,
            )
            _log_to_mlflow(report)
            mlflow.log_artifact(str(results_dir / f"evaluation_{asset}.json"))


if __name__ == "__main__":
    main()
