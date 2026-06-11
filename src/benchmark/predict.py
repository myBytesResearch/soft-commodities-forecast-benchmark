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
#  File: predict.py | Project: soft-commodities-forecast-benchmark | Author: Guido Winger
# =============================================================================

"""
GJR-GARCH(1,1,1) Student-t — 1-step-ahead forecast on validation + test.

Walk-forward scheme:
- Refit the model every ``refit_frequency_days`` trading days on the expanding
  in-sample window.
- Between refits, the fitted parameters are held fixed and the conditional
  variance recursion is filtered through the new returns to produce honest
  1-step-ahead forecasts σ_t = f(I_{t-1}; θ_refit).

The output is a single ``forecast.parquet`` per asset containing date, the
realised return, σ_t (conditional vol forecast), σ²_t, and the index of the
refit cohort the forecast belongs to.

Usage:
    python -m models.01_gjr_garch_t.predict --asset cocoa
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from arch import arch_model
from loguru import logger

from benchmark.preprocessing import (
    compute_returns,
    load_global_config,
    split_data,
)
from benchmark.yfinance_loader import YFinanceLoader


def _build_model(returns: pd.Series, model_cfg: dict, training_cfg: dict):
    """Instantiate an arch_model object from a config block."""
    return arch_model(
        returns,
        mean=model_cfg["mean"],
        vol=model_cfg["vol"],
        p=model_cfg["p"],
        o=model_cfg["o"],
        q=model_cfg["q"],
        power=model_cfg.get("power", 2.0),
        dist=model_cfg["dist"],
        rescale=training_cfg.get("rescale", False),
    )


def walk_forward_forecast(
    returns: pd.Series,
    eval_index: pd.DatetimeIndex,
    model_cfg: dict,
    training_cfg: dict,
    refit_every: int,
) -> pd.DataFrame:
    """
    Produce 1-step-ahead conditional volatility forecasts on ``eval_index``.

    Args:
        returns: Full returns series (train+val+test) used for both fitting
            and filtering. Must be sorted, no NaNs.
        eval_index: Dates for which we need σ_t forecasts.
        model_cfg: Model-spec block from the model config.
        training_cfg: Training block (cov_type, rescale, etc.).
        refit_every: Refit cadence in trading days.

    Returns:
        DataFrame indexed by date with columns
            ``return``, ``cond_vol``, ``cond_var``, ``refit_cohort``,
            ``nu`` (Student-t df at the active refit) and ``mu``.
    """
    full = returns.dropna().sort_index()
    eval_pos = full.index.get_indexer(eval_index)
    eval_pos = eval_pos[eval_pos >= 0]
    if len(eval_pos) == 0:
        raise ValueError("eval_index has no overlap with returns index")

    chunks: list[pd.DataFrame] = []
    cohort_id = 0

    for chunk_start in range(eval_pos.min(), eval_pos.max() + 1, refit_every):
        chunk_end = min(chunk_start + refit_every, eval_pos.max() + 1)
        train_slice = full.iloc[:chunk_start]

        if len(train_slice) < 250:
            logger.warning(
                f"Refit window too short at position {chunk_start} "
                f"({len(train_slice)} obs); skipping cohort."
            )
            continue

        am_fit = _build_model(train_slice, model_cfg, training_cfg)
        res = am_fit.fit(
            disp="off",
            cov_type=training_cfg.get("cov_type", "robust"),
            show_warning=False,
        )

        # Filter the fitted parameters through training + eval chunk to get σ_t
        # conditional on information up to t-1.
        full_to_chunk_end = full.iloc[:chunk_end]
        am_filter = _build_model(full_to_chunk_end, model_cfg, training_cfg)
        filtered = am_filter.fix(res.params)

        sigma = pd.Series(
            np.asarray(filtered.conditional_volatility),
            index=full_to_chunk_end.index,
            name="cond_vol",
        )
        chunk_index = full.index[chunk_start:chunk_end]
        sigma_chunk = sigma.loc[chunk_index]

        nu = float(res.params.get("nu", np.nan))
        mu = float(res.params.get("mu", 0.0))

        chunk_df = pd.DataFrame(
            {
                "return": full.loc[chunk_index].values,
                "cond_vol": sigma_chunk.values,
                "cond_var": (sigma_chunk.values ** 2),
                "refit_cohort": cohort_id,
                "nu": nu,
                "mu": mu,
            },
            index=chunk_index,
        )
        chunks.append(chunk_df)
        cohort_id += 1
        logger.info(
            f"Cohort {cohort_id - 1}: fit on {len(train_slice)} obs, "
            f"σ̄={sigma_chunk.mean():.3f} over {len(sigma_chunk)} days"
        )

    forecast = pd.concat(chunks).sort_index()
    forecast = forecast.loc[~forecast.index.duplicated(keep="first")]
    return forecast


def predict_single_asset(
    asset: str,
    global_config: dict,
    model_config: dict,
    artifacts_dir: Path,
    results_dir: Path,
) -> pd.DataFrame:
    """Run walk-forward 1-step-ahead forecast on val+test for one asset."""
    logger.info("=" * 80)
    logger.info(f"Forecasting GJR-GARCH-t for: {asset}")
    logger.info("=" * 80)

    loader = YFinanceLoader(cache_dir="data/raw/prices")
    df = loader.load(symbol=asset, start="2000-01-01")
    returns = compute_returns(df["close"], method=model_config["data"]["return_method"])
    split = split_data(returns.to_frame("returns"), config=global_config)

    eval_index = split.val.index.append(split.test.index).sort_values().unique()

    refit_every = int(
        model_config.get("backtest", {}).get(
            "refit_frequency_days",
            global_config["backtest"]["refit_frequency_days"],
        )
    )

    forecast = walk_forward_forecast(
        returns=returns,
        eval_index=eval_index,
        model_cfg=model_config["model"],
        training_cfg=model_config["training"],
        refit_every=refit_every,
    )

    forecast["split"] = np.where(forecast.index <= split.val.index.max(), "val", "test")

    results_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = results_dir / f"forecast_{asset}.parquet"
    forecast.to_parquet(forecast_path)
    logger.success(
        f"✓ Forecast persisted: {forecast_path} "
        f"({len(forecast)} rows, val={int((forecast.split == 'val').sum())}, "
        f"test={int((forecast.split == 'test').sum())})"
    )
    return forecast


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward forecast for GJR-GARCH-t.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--global-config", type=str, default="configs/global.yaml")
    parser.add_argument("--asset", type=str, default=None)
    args = parser.parse_args()

    global_config = load_global_config(args.global_config)
    with open(args.config) as f:
        model_config = yaml.safe_load(f)

    assets = [args.asset] if args.asset else model_config["data"]["assets"]
    cfg_path = Path(args.config)
    artifacts_dir = cfg_path.parent / "artifacts"
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
        with mlflow.start_run(run_name=f"predict_{asset}"):
            mlflow.set_tag("phase", "predict")
            mlflow.set_tag("asset", asset)
            forecast = predict_single_asset(
                asset=asset,
                global_config=global_config,
                model_config=model_config,
                artifacts_dir=artifacts_dir,
                results_dir=results_dir,
            )
            mlflow.log_metric("n_forecast_obs", len(forecast))
            mlflow.log_metric("mean_cond_vol", float(forecast["cond_vol"].mean()))
            mlflow.log_artifact(str(results_dir / f"forecast_{asset}.parquet"))


if __name__ == "__main__":
    main()