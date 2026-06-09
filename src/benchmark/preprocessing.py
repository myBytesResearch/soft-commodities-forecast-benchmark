"""
Preprocessing — Zentrale Funktionen für Returns, Splits, Cleaning.

Wird von allen Modellen unter models/*/train.py importiert, damit alle
auf demselben Train/Val/Test-Frame arbeiten.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger


@dataclass(frozen=True)
class DataSplit:
    """Train/Val/Test Split-Container."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def __repr__(self) -> str:
        return (
            f"DataSplit("
            f"train={len(self.train)} [{self.train.index.min().date()} → {self.train.index.max().date()}], "
            f"val={len(self.val)} [{self.val.index.min().date()} → {self.val.index.max().date()}], "
            f"test={len(self.test)} [{self.test.index.min().date()} → {self.test.index.max().date()}])"
        )


def load_global_config(config_path: Path | str = "configs/global.yaml") -> dict:
    """Lädt configs/global.yaml."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_returns(
    prices: pd.Series,
    method: str = "log",
    drop_na: bool = True,
) -> pd.Series:
    """
    Berechnet Returns aus Preisreihe.

    Args:
        prices: Preisreihe (z.B. df['close']).
        method: 'log' oder 'simple'.
        drop_na: Erste NaN-Zeile droppen.

    Returns:
        pd.Series der Returns, in Prozent skaliert (×100), wie in arch-Konvention.
        Das macht die GARCH-Schätzung numerisch stabiler.
    """
    if method == "log":
        ret = np.log(prices / prices.shift(1))
    elif method == "simple":
        ret = prices.pct_change()
    else:
        raise ValueError(f"Unbekannte Methode: {method}. Erlaubt: 'log', 'simple'.")

    ret = ret * 100.0  # arch-package Convention: Returns in Prozent

    if drop_na:
        ret = ret.dropna()

    ret.name = f"{method}_return_pct"
    return ret


def split_data(
    df: pd.DataFrame,
    config: dict | None = None,
    config_path: Path | str = "configs/global.yaml",
) -> DataSplit:
    """
    Splittet DataFrame nach den Datums-Konfigurationen aus global.yaml.

    Args:
        df: DataFrame mit DatetimeIndex.
        config: Optional vorgeladene Config. Wenn None → wird aus config_path geladen.

    Returns:
        DataSplit(train, val, test)
    """
    if config is None:
        config = load_global_config(config_path)

    s = config["splits"]

    # Index-Timezone harmonisieren
    if df.index.tz is not None:
        train_start = pd.Timestamp(s["train_start"], tz=df.index.tz)
        train_end = pd.Timestamp(s["train_end"], tz=df.index.tz)
        val_start = pd.Timestamp(s["val_start"], tz=df.index.tz)
        val_end = pd.Timestamp(s["val_end"], tz=df.index.tz)
        test_start = pd.Timestamp(s["test_start"], tz=df.index.tz)
        test_end = pd.Timestamp(s["test_end"], tz=df.index.tz)
    else:
        train_start = pd.Timestamp(s["train_start"])
        train_end = pd.Timestamp(s["train_end"])
        val_start = pd.Timestamp(s["val_start"])
        val_end = pd.Timestamp(s["val_end"])
        test_start = pd.Timestamp(s["test_start"])
        test_end = pd.Timestamp(s["test_end"])

    train = df.loc[train_start:train_end].copy()
    val = df.loc[val_start:val_end].copy()
    test = df.loc[test_start:test_end].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"Mindestens ein Split ist leer. "
            f"train={len(train)}, val={len(val)}, test={len(test)}. "
            f"Datenbereich: {df.index.min().date()} → {df.index.max().date()}"
        )

    split = DataSplit(train=train, val=val, test=test)
    logger.info(f"Data-Split erstellt: {split}")
    return split


def stylized_facts_summary(returns: pd.Series) -> dict:
    """
    Berechnet die klassischen 'stylized facts' für Finanzreihen:
    Mean, Std, Skew, Kurtosis, Jarque-Bera, Ljung-Box auf r und r².

    Returns:
        dict mit den Kennzahlen.
    """
    from scipy import stats
    from statsmodels.stats.diagnostic import acorr_ljungbox

    r = returns.dropna()

    jb_stat, jb_pvalue = stats.jarque_bera(r)
    lb_r = acorr_ljungbox(r, lags=[10, 20], return_df=True)
    lb_r2 = acorr_ljungbox(r**2, lags=[10, 20], return_df=True)

    return {
        "n_obs": len(r),
        "mean": float(r.mean()),
        "std": float(r.std()),
        "min": float(r.min()),
        "max": float(r.max()),
        "skew": float(stats.skew(r)),
        "kurtosis_excess": float(stats.kurtosis(r)),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_pvalue),
        "ljung_box_r_lag10_pvalue": float(lb_r.loc[10, "lb_pvalue"]),
        "ljung_box_r2_lag10_pvalue": float(lb_r2.loc[10, "lb_pvalue"]),
        "n_negative_returns": int((r < 0).sum()),
        "n_extreme_3sigma": int((r.abs() > 3 * r.std()).sum()),
    }


def arch_lm_test(returns: pd.Series, lags: int = 10) -> dict:
    """
    Engle's ARCH-LM Test: Gibt es signifikante ARCH-Effekte?

    H0: Keine ARCH-Effekte (Residuen homoskedastisch)
    H1: ARCH-Effekte vorhanden

    Returns:
        dict mit lm_stat, lm_pvalue, f_stat, f_pvalue.
    """
    from statsmodels.stats.diagnostic import het_arch

    r = returns.dropna()
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(r, nlags=lags)

    return {
        "lags": lags,
        "lm_stat": float(lm_stat),
        "lm_pvalue": float(lm_pvalue),
        "f_stat": float(f_stat),
        "f_pvalue": float(f_pvalue),
        "arch_effects_detected": bool(lm_pvalue < 0.05),
    }
