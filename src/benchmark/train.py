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
#  File: train.py | Project: soft-commodities-forecast-benchmark | Author: Guido Winger
# =============================================================================

"""
GJR-GARCH(1,1,1) mit Student-t Innovationen — Training Script.

Welle-1-Baseline. Dient gleichzeitig als Template für weitere Modelle:
jedes Modell hat dieselbe train.py-Signatur.

Usage:
    python -m models.01_gjr_garch_t.train --config models/01_gjr_garch_t/config.yaml
    python -m models.01_gjr_garch_t.train --asset cocoa
    python -m models.01_gjr_garch_t.train --asset coffee
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml
from arch import arch_model
from loguru import logger

from benchmark.preprocessing import (
    arch_lm_test,
    compute_returns,
    load_global_config,
    split_data,
    stylized_facts_summary,
)
from benchmark.yfinance_loader import YFinanceLoader


# ==============================================================================
# Helpers
# ==============================================================================
def setup_mlflow(global_config: dict, model_config: dict, asset: str) -> None:
    """MLflow Tracking-Setup."""
    mlflow_cfg = global_config["mlflow"]
    mode = mlflow_cfg.get("default_mode", "local")

    if mode == "local":
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri_local"])
    else:
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri_remote"])

    experiment_name = (
        f"{asset}_volatility_{mlflow_cfg['experiment_prefix']}"
        f"{model_config['mlflow']['experiment_suffix']}"
    )
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow Experiment: {experiment_name} ({mode})")


def load_data(asset: str) -> pd.DataFrame:
    """Lädt Preisdaten via YFinanceLoader."""
    loader = YFinanceLoader(cache_dir="data/raw/prices")
    df = loader.load(symbol=asset, start="2000-01-01")
    return df


# ==============================================================================
# Training
# ==============================================================================
def train_single_asset(
    asset: str,
    global_config: dict,
    model_config: dict,
    artifacts_dir: Path,
) -> dict[str, Any]:
    """
    Trainiert GJR-GARCH-t für ein einzelnes Asset.

    Returns:
        dict mit Trainingsergebnissen (Parameter, In-Sample-Metriken, Diagnostik).
    """
    logger.info(f"=" * 80)
    logger.info(f"Training GJR-GARCH-t für: {asset}")
    logger.info(f"=" * 80)

    # ----- Daten laden -----
    df = load_data(asset)
    returns = compute_returns(
        df["close"],
        method=model_config["data"]["return_method"],
    )
    # Note: compute_returns gibt schon ×100-Returns zurück (arch-Convention)

    split = split_data(returns.to_frame("returns"), config=global_config)
    train_returns = split.train["returns"]

    logger.info(f"Train: {len(train_returns)} Beobachtungen")

    # ----- Stylized Facts + ARCH-LM (Diagnostik) -----
    stylized = stylized_facts_summary(train_returns)
    arch_lm = arch_lm_test(train_returns, lags=10)

    logger.info(f"Mean: {stylized['mean']:.4f}, Std: {stylized['std']:.4f}")
    logger.info(f"Skew: {stylized['skew']:.4f}, Excess Kurt: {stylized['kurtosis_excess']:.4f}")
    logger.info(f"ARCH-LM p-value: {arch_lm['lm_pvalue']:.4g} → {'ARCH effects' if arch_lm['arch_effects_detected'] else 'NO ARCH effects'}")

    # ----- Modell-Spezifikation -----
    m_cfg = model_config["model"]
    t_cfg = model_config["training"]

    am = arch_model(
        train_returns,
        mean=m_cfg["mean"],
        vol=m_cfg["vol"],
        p=m_cfg["p"],
        o=m_cfg["o"],
        q=m_cfg["q"],
        power=m_cfg.get("power", 2.0),
        dist=m_cfg["dist"],
        rescale=t_cfg.get("rescale", False),
    )

    logger.info(f"Modell: {am}")

    # ----- Schätzung -----
    res = am.fit(
        update_freq=t_cfg.get("update_freq", 5),
        show_warning=t_cfg.get("show_warning", False),
        cov_type=t_cfg.get("cov_type", "robust"),
        disp="off",
    )

    logger.info(f"Konvergiert: {res.convergence_flag == 0}")
    logger.info(f"LogLik: {res.loglikelihood:.2f}, AIC: {res.aic:.2f}, BIC: {res.bic:.2f}")
    logger.info(f"\n{res.summary()}")

    # ----- Persistenz-Check -----
    params = res.params
    alpha = float(params.get("alpha[1]", 0))
    gamma = float(params.get("gamma[1]", 0))
    beta = float(params.get("beta[1]", 0))
    persistence = alpha + 0.5 * gamma + beta  # Bei symmetrischer Verteilung
    logger.info(
        f"Persistence α + 0.5γ + β = {persistence:.4f} "
        f"{'(stationary)' if persistence < 1 else '(WARNING: non-stationary!)'}"
    )

    # ----- Artefakte speichern -----
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / f"gjr_garch_t_{asset}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(res, f)

    diagnostics = {
        "asset": asset,
        "timestamp": datetime.now().isoformat(),
        "n_train_obs": len(train_returns),
        "train_start": str(split.train.index.min().date()),
        "train_end": str(split.train.index.max().date()),
        "stylized_facts": stylized,
        "arch_lm_test": arch_lm,
        "model_params": {k: float(v) for k, v in params.items()},
        "model_pvalues": {k: float(v) for k, v in res.pvalues.items()},
        "loglikelihood": float(res.loglikelihood),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "persistence": persistence,
        "convergence_flag": int(res.convergence_flag),
    }

    diag_path = artifacts_dir / f"gjr_garch_t_{asset}_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)

    logger.info(f"✓ Modell gespeichert: {model_path}")
    logger.info(f"✓ Diagnostik gespeichert: {diag_path}")

    return diagnostics


# ==============================================================================
# Main
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Train GJR-GARCH-t baseline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Modell-Config-YAML.",
    )
    parser.add_argument(
        "--global-config",
        type=str,
        default="configs/global.yaml",
    )
    parser.add_argument(
        "--asset",
        type=str,
        default=None,
        help="Einzelnes Asset. Default: alle aus config.",
    )
    args = parser.parse_args()

    # Configs laden
    global_config = load_global_config(args.global_config)
    with open(args.config) as f:
        model_config = yaml.safe_load(f)

    assets = [args.asset] if args.asset else model_config["data"]["assets"]

    # Artifacts-Dir
    artifacts_dir = Path(args.config).parent / "artifacts"

    for asset in assets:
        try:
            setup_mlflow(global_config, model_config, asset)

            with mlflow.start_run(run_name=f"gjr_garch_t_{asset}_{datetime.now():%Y%m%d_%H%M%S}"):
                # Tags
                for k, v in global_config["mlflow"]["default_tags"].items():
                    mlflow.set_tag(k, v)
                mlflow.set_tag("asset", asset)
                mlflow.set_tag("model_family", model_config["model"]["family"])
                mlflow.set_tag("model_name", model_config["model"]["name"])

                # Params
                mlflow.log_params(model_config["model"])

                # Training
                diag = train_single_asset(
                    asset=asset,
                    global_config=global_config,
                    model_config=model_config,
                    artifacts_dir=artifacts_dir,
                )

                # Metrics
                mlflow.log_metric("loglikelihood", diag["loglikelihood"])
                mlflow.log_metric("aic", diag["aic"])
                mlflow.log_metric("bic", diag["bic"])
                mlflow.log_metric("persistence", diag["persistence"])
                mlflow.log_metric("train_n_obs", diag["n_train_obs"])
                mlflow.log_metric("kurtosis_excess", diag["stylized_facts"]["kurtosis_excess"])
                mlflow.log_metric("arch_lm_pvalue", diag["arch_lm_test"]["lm_pvalue"])

                # Artifacts
                mlflow.log_artifact(str(artifacts_dir / f"gjr_garch_t_{asset}.pkl"))
                mlflow.log_artifact(str(artifacts_dir / f"gjr_garch_t_{asset}_diagnostics.json"))

                logger.success(f"✓ {asset} → MLflow Run ID: {mlflow.active_run().info.run_id}")

        except Exception as e:
            logger.exception(f"✗ Training für {asset} fehlgeschlagen: {e}")


if __name__ == "__main__":
    main()