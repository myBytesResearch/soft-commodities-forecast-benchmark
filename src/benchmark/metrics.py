"""
Metrics — Volatility Forecast Evaluation + VaR/ES Backtest.

Wird von allen Modellen für ein faires, einheitliches Scoring genutzt.

Referenzen:
- QLIKE: Patton (2011), "Volatility forecast comparison using imperfect proxies"
- Kupiec POF: Kupiec (1995), "Techniques for verifying the accuracy of risk measurement models"
- Christoffersen CC: Christoffersen (1998), "Evaluating interval forecasts"
- Engle-Manganelli DQ: Engle & Manganelli (2004), CAViaR
- Acerbi-Szekely ES: Acerbi & Szekely (2014), "Back-testing Expected Shortfall"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ==============================================================================
# Volatility Forecast Metrics
# ==============================================================================
def qlike(realized_var: np.ndarray, forecast_var: np.ndarray) -> float:
    """
    QLIKE Loss (Patton, 2011).

    Robust gegen Noise im Realized-Vol-Proxy.
    Niedriger ist besser.
    """
    realized_var = np.asarray(realized_var, dtype=float)
    forecast_var = np.asarray(forecast_var, dtype=float)

    # Schutz gegen 0 / negative Werte
    mask = (forecast_var > 0) & (realized_var > 0) & np.isfinite(realized_var) & np.isfinite(forecast_var)
    rv = realized_var[mask]
    fv = forecast_var[mask]

    if len(rv) == 0:
        return np.nan

    return float(np.mean(rv / fv - np.log(rv / fv) - 1.0))


def mse_variance(realized_var: np.ndarray, forecast_var: np.ndarray) -> float:
    """MSE auf Varianzen."""
    realized_var = np.asarray(realized_var, dtype=float)
    forecast_var = np.asarray(forecast_var, dtype=float)
    mask = np.isfinite(realized_var) & np.isfinite(forecast_var)
    return float(np.mean((realized_var[mask] - forecast_var[mask]) ** 2))


def mae_variance(realized_var: np.ndarray, forecast_var: np.ndarray) -> float:
    """MAE auf Varianzen."""
    realized_var = np.asarray(realized_var, dtype=float)
    forecast_var = np.asarray(forecast_var, dtype=float)
    mask = np.isfinite(realized_var) & np.isfinite(forecast_var)
    return float(np.mean(np.abs(realized_var[mask] - forecast_var[mask])))


def r2_oos(
    realized_var: np.ndarray,
    forecast_var: np.ndarray,
    benchmark: np.ndarray | None = None,
) -> float:
    """
    Out-of-Sample R² (Campbell-Thompson) of the forecast vs. a benchmark.

    Default benchmark is the historical mean of realised variance. When
    ``benchmark`` is provided (same length as realised), the OOS R² is
    computed against it — useful for comparing against a harder baseline
    such as a 5-day rolling mean (HAR-RV-style) instead of the trivial
    long-run mean.
    """
    realized_var = np.asarray(realized_var, dtype=float)
    forecast_var = np.asarray(forecast_var, dtype=float)
    if benchmark is not None:
        benchmark = np.asarray(benchmark, dtype=float)
        mask = (
            np.isfinite(realized_var)
            & np.isfinite(forecast_var)
            & np.isfinite(benchmark)
        )
        rv = realized_var[mask]
        fv = forecast_var[mask]
        bm = benchmark[mask]
    else:
        mask = np.isfinite(realized_var) & np.isfinite(forecast_var)
        rv = realized_var[mask]
        fv = forecast_var[mask]
        bm = None

    if len(rv) < 2:
        return np.nan

    ss_res = np.sum((rv - fv) ** 2)
    if bm is None:
        ss_tot = np.sum((rv - rv.mean()) ** 2)
    else:
        ss_tot = np.sum((rv - bm) ** 2)

    if ss_tot == 0:
        return np.nan

    return float(1.0 - ss_res / ss_tot)


def rolling_mean_benchmark(
    realized_var: np.ndarray | pd.Series,
    window: int = 5,
) -> np.ndarray:
    """
    HAR-RV-style benchmark: rolling mean of past realised variance.

    Returns the rolling mean shifted by one period so that the benchmark
    for day t uses only information up to t-1 (no look-ahead).
    """
    s = pd.Series(np.asarray(realized_var, dtype=float))
    bm = s.rolling(window=window, min_periods=max(1, window // 2)).mean().shift(1)
    return bm.to_numpy()


# ==============================================================================
# VaR / ES Berechnung aus Volatilitäts-Forecast
# ==============================================================================
def var_from_vol(
    forecast_vol: np.ndarray,
    confidence: float = 0.99,
    distribution: str = "normal",
    df: float | None = None,
    mean_return: float = 0.0,
) -> np.ndarray:
    """
    Berechnet Value-at-Risk aus Volatilitäts-Forecast.

    Args:
        forecast_vol: Forecast Standard Deviation (NICHT Varianz).
        confidence: z.B. 0.99 für 99%-VaR.
        distribution: 'normal' oder 'studentst'.
        df: Degrees of Freedom für Student-t.
        mean_return: Bedingter Mean.

    Returns:
        VaR als negativer Return (Konvention: VaR > 0 = Verlust).
    """
    alpha = 1.0 - confidence

    if distribution == "normal":
        quantile = stats.norm.ppf(alpha)
    elif distribution == "studentst":
        if df is None:
            raise ValueError("df muss bei Student-t angegeben werden.")
        # Skalierungsfaktor: Student-t Vol = sqrt(df / (df-2)) * sigma
        scale_factor = np.sqrt((df - 2.0) / df)
        quantile = stats.t.ppf(alpha, df) * scale_factor
    else:
        raise ValueError(f"Unbekannte Verteilung: {distribution}")

    # VaR = -(mean + quantile * vol), positiv = Verlust
    var = -(mean_return + quantile * forecast_vol)
    return np.asarray(var)


def es_from_vol(
    forecast_vol: np.ndarray,
    confidence: float = 0.975,
    distribution: str = "normal",
    df: float | None = None,
    mean_return: float = 0.0,
) -> np.ndarray:
    """Expected Shortfall (Conditional VaR) aus Volatilitäts-Forecast."""
    alpha = 1.0 - confidence

    if distribution == "normal":
        es_quantile = -stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
    elif distribution == "studentst":
        if df is None:
            raise ValueError("df muss bei Student-t angegeben werden.")
        t_alpha = stats.t.ppf(alpha, df)
        scale_factor = np.sqrt((df - 2.0) / df)
        es_quantile = (
            -(df + t_alpha**2) / (df - 1)
            * stats.t.pdf(t_alpha, df) / alpha
            * scale_factor
        )
    else:
        raise ValueError(f"Unbekannte Verteilung: {distribution}")

    es = -(mean_return + es_quantile * forecast_vol)
    return np.asarray(es)


# ==============================================================================
# VaR Backtest Tests
# ==============================================================================
@dataclass
class BacktestResult:
    """Container für VaR-Backtest-Ergebnisse."""

    test_name: str
    statistic: float
    pvalue: float
    n_violations: int
    n_observations: int
    expected_violations: float
    violation_rate: float
    rejected: bool                # H0 rejected at 5% level?

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "statistic": float(self.statistic),
            "pvalue": float(self.pvalue),
            "n_violations": int(self.n_violations),
            "n_observations": int(self.n_observations),
            "expected_violations": float(self.expected_violations),
            "violation_rate": float(self.violation_rate),
            "rejected": bool(self.rejected),
        }


def kupiec_pof_test(
    returns: np.ndarray,
    var_forecast: np.ndarray,
    confidence: float = 0.99,
) -> BacktestResult:
    """
    Kupiec Proportion of Failures (POF) Test.

    H0: Beobachtete Violation-Rate = (1 - confidence)
    H1: Modell mis-spezifiziert.

    Args:
        returns: Realized Returns (gleicher Sign-Convention wie für VaR).
        var_forecast: VaR als positive Verluste.
        confidence: VaR-Konfidenzniveau (z.B. 0.99).
    """
    returns = np.asarray(returns, dtype=float)
    var_forecast = np.asarray(var_forecast, dtype=float)

    # Violation: realisierter Verlust > VaR  =>  -returns > VaR
    violations = (-returns > var_forecast).astype(int)
    n = len(violations)
    x = int(violations.sum())
    p_expected = 1.0 - confidence

    if x == 0 or x == n:
        # Edge case: keine oder alle Verletzungen
        lr_stat = np.nan
        pvalue = np.nan
    else:
        p_hat = x / n
        lr_stat = -2.0 * (
            (n - x) * np.log(1 - p_expected) + x * np.log(p_expected)
            - (n - x) * np.log(1 - p_hat) - x * np.log(p_hat)
        )
        pvalue = 1.0 - stats.chi2.cdf(lr_stat, df=1)

    return BacktestResult(
        test_name="kupiec_pof",
        statistic=lr_stat,
        pvalue=pvalue,
        n_violations=x,
        n_observations=n,
        expected_violations=n * p_expected,
        violation_rate=x / n if n > 0 else np.nan,
        rejected=bool(pvalue < 0.05) if not np.isnan(pvalue) else False,
    )


def christoffersen_cc_test(
    returns: np.ndarray,
    var_forecast: np.ndarray,
    confidence: float = 0.99,
) -> BacktestResult:
    """
    Christoffersen Conditional Coverage Test.

    Kombiniert Unconditional Coverage (Kupiec) + Independence Test.
    H0: Korrekte Coverage UND Unabhängigkeit der Violations.
    """
    returns = np.asarray(returns, dtype=float)
    var_forecast = np.asarray(var_forecast, dtype=float)
    violations = (-returns > var_forecast).astype(int)
    n = len(violations)

    # Transition Counts
    # n_ij = Anzahl Übergänge von Zustand i zu j
    n00 = n01 = n10 = n11 = 0
    for t in range(1, n):
        prev, curr = violations[t - 1], violations[t]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    # Independence LR
    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if n > 1 else 0

    if pi in (0, 1) or pi_01 in (0, 1) or pi_11 in (0, 1):
        lr_ind = 0.0
    else:
        lr_ind = -2.0 * (
            (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
            - n00 * np.log(1 - pi_01) - n01 * np.log(pi_01)
            - n10 * np.log(1 - pi_11) - n11 * np.log(pi_11)
        )

    # Unconditional Coverage (Kupiec)
    kupiec = kupiec_pof_test(returns, var_forecast, confidence)
    lr_uc = kupiec.statistic if not np.isnan(kupiec.statistic) else 0.0

    # Conditional Coverage = UC + Independence
    lr_cc = lr_uc + lr_ind
    pvalue = 1.0 - stats.chi2.cdf(lr_cc, df=2) if lr_cc > 0 else 1.0

    return BacktestResult(
        test_name="christoffersen_cc",
        statistic=lr_cc,
        pvalue=pvalue,
        n_violations=int(violations.sum()),
        n_observations=n,
        expected_violations=n * (1.0 - confidence),
        violation_rate=violations.sum() / n if n > 0 else np.nan,
        rejected=bool(pvalue < 0.05),
    )


def acerbi_szekely_z1(
    returns: np.ndarray,
    var_forecast: np.ndarray,
    es_forecast: np.ndarray,
    confidence: float = 0.975,
) -> dict:
    """
    Acerbi-Szekely (2014) Z_1 ES backtest statistic.

    Z_1 = (1/N_t) · Σ_t (X_t / ES_t) · I(X_t < -VaR_t) − 1
    where X_t is the realised return and ES_t is the forecast Expected
    Shortfall (positive loss). Under H0 (ES correct), E[Z_1] = 0.
    Z_1 < 0 → realised tail losses exceed predicted ES → ES under-estimates.
    """
    returns = np.asarray(returns, dtype=float)
    var_forecast = np.asarray(var_forecast, dtype=float)
    es_forecast = np.asarray(es_forecast, dtype=float)
    violations = (-returns) > var_forecast
    n_v = int(violations.sum())
    if n_v == 0 or not np.any(np.isfinite(es_forecast[violations])):
        return {
            "z1": float("nan"),
            "n_violations": n_v,
            "expected_violations": float(len(returns) * (1.0 - confidence)),
            "interpretation": "no violations to evaluate",
        }
    contrib = (-returns[violations]) / es_forecast[violations]
    z1 = float(contrib.mean() / 1.0 - 1.0)  # mean(X/ES) - 1; X is loss here
    return {
        "z1": z1,
        "n_violations": n_v,
        "expected_violations": float(len(returns) * (1.0 - confidence)),
        "interpretation": (
            "ES under-predicts tails" if z1 > 0
            else "ES over-predicts tails (conservative)"
        ),
    }


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """
    Holm-Bonferroni step-down for a labelled dict of p-values.

    Returns ``{label: {"pvalue": p, "adjusted": p_adj, "rejected": bool}}``.
    NaNs are passed through unchanged and never reject.
    """
    items = [(k, v) for k, v in pvalues.items()]
    finite = [(k, v) for k, v in items if v is not None and np.isfinite(v)]
    finite.sort(key=lambda kv: kv[1])
    m = len(finite)
    out: dict[str, dict] = {}
    for i, (label, p) in enumerate(finite):
        adj = min(1.0, (m - i) * p)
        out[label] = {"pvalue": float(p), "adjusted": float(adj), "rejected": bool(adj <= alpha)}
    for k, v in items:
        if k not in out:
            out[k] = {"pvalue": v, "adjusted": float("nan"), "rejected": False}
    return out


# ==============================================================================
# Pre-Crisis Window Slicing
# ==============================================================================
def pre_crisis_slice(
    series_index: pd.DatetimeIndex,
    crisis_start_date: str | pd.Timestamp,
    window_days: tuple[int, int] = (180, 365),
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Return the (start, end) timestamps of the pre-crisis window.

    Window is ``[crisis_start - window_days[1], crisis_start - window_days[0]]``.
    Default = 180–365 days before crisis: the period where early-warning
    coverage actually matters, distinct from the always-conservative full-sample
    backtest. The endpoint is BEFORE the crisis so realised returns are still
    pre-stress.
    """
    crisis_start = pd.Timestamp(crisis_start_date)
    if series_index.tz is not None and crisis_start.tz is None:
        crisis_start = crisis_start.tz_localize(series_index.tz)
    start = crisis_start - pd.Timedelta(days=window_days[1])
    end = crisis_start - pd.Timedelta(days=window_days[0])
    return start, end


# ==============================================================================
# Black-Swan Specific Metrics
# ==============================================================================
def _rolling_baseline(
    signal: pd.Series,
    window_days: int,
    statistic: str = "median",
    ema_lambda: float = 0.94,
) -> pd.Series:
    """
    Rolling baseline of the risk signal.

    Supported statistics:
        - ``median`` / ``mean``: simple rolling window of length ``window_days``.
        - ``ema``: exponentially weighted mean with decay factor ``ema_lambda``
          (RiskMetrics convention; λ=0.94 ≈ 12-day half-life). ``window_days``
          is used only to compute the warm-up min_periods so very short
          histories do not produce overconfident baselines.
    """
    min_p = max(30, window_days // 4)
    if statistic == "median":
        return signal.rolling(window=window_days, min_periods=min_p).median()
    if statistic == "mean":
        return signal.rolling(window=window_days, min_periods=min_p).mean()
    if statistic == "ema":
        # In EWMA notation alpha = 1 - lambda.
        if not 0.0 < ema_lambda < 1.0:
            raise ValueError(f"ema_lambda must be in (0,1), got {ema_lambda}")
        return signal.ewm(alpha=(1.0 - ema_lambda), min_periods=min_p, adjust=False).mean()
    raise ValueError(f"Unsupported baseline statistic: {statistic!r}")


def calibrate_detection_threshold(
    signal_val: pd.Series,
    target_fpr: float,
    baseline_window: int = 252,
    baseline_statistic: str = "median",
    ema_lambda: float = 0.94,
    multiplier_grid: np.ndarray | None = None,
) -> dict:
    """
    Calibrates the alarm-threshold multiplier on the validation set under an FPR constraint.

    The validation period MUST be crisis-free by construction (2019–2020). Under that
    assumption every alarm is a false positive, so the empirical FPR equals the alarm
    rate. We pick the smallest multiplier m such that alarm-rate ≤ target_fpr.

    Args:
        signal_val: Risk signal on the validation period (e.g. conditional vol).
        target_fpr: Maximum tolerated false-positive rate.
        baseline_window: Rolling window (days) for the baseline level.
        baseline_statistic: "median" or "mean".
        multiplier_grid: Optional override of the search grid.

    Returns:
        dict with the calibrated multiplier, realised FPR, and diagnostics.
    """
    if not 0.0 < target_fpr < 1.0:
        raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")

    signal = signal_val.dropna().sort_index()
    baseline = _rolling_baseline(signal, baseline_window, baseline_statistic, ema_lambda)
    ratio = (signal / baseline).dropna()

    if ratio.empty:
        return {
            "multiplier": np.nan,
            "realised_fpr": np.nan,
            "target_fpr": target_fpr,
            "n_eval": 0,
            "note": "insufficient validation data for baseline window",
        }

    if multiplier_grid is None:
        multiplier_grid = np.round(np.arange(1.05, 5.01, 0.05), 4)

    # Smallest m with alarm-rate ≤ target_fpr; ratios are monotone in m (fewer alarms ⇒ higher m).
    chosen_m = float("nan")
    realised = float("nan")
    for m in multiplier_grid:
        rate = float((ratio > m).mean())
        if rate <= target_fpr:
            chosen_m = float(m)
            realised = rate
            break

    return {
        "multiplier": chosen_m,
        "realised_fpr": realised,
        "target_fpr": target_fpr,
        "baseline_window": baseline_window,
        "baseline_statistic": baseline_statistic,
        "ema_lambda": float(ema_lambda) if baseline_statistic == "ema" else None,
        "n_eval": int(len(ratio)),
        "ratio_min": float(ratio.min()),
        "ratio_median": float(ratio.median()),
        "ratio_max": float(ratio.max()),
    }


def calibrate_detection_threshold_pooled(
    signals_by_asset: dict[str, pd.Series],
    target_fpr: float,
    baseline_window: int = 252,
    baseline_statistic: str = "median",
    ema_lambda: float = 0.94,
    multiplier_grid: np.ndarray | None = None,
) -> dict:
    """
    Pool the ``signal_t / baseline_t`` ratios across multiple assets, then
    pick the smallest multiplier whose pooled alarm-rate satisfies the FPR
    constraint.

    Ratio-pooling (not raw-signal pooling) is the right operation here:
    each asset's signal lives on its own vol scale, but the ratio is unit-free
    and directly comparable across assets.

    Args:
        signals_by_asset: ``{asset_name: signal_series_on_val}``. Each series
            should already be restricted to its validation slice.
        target_fpr: Maximum tolerated false-positive rate (pooled).
        baseline_window / baseline_statistic / ema_lambda: as in
            ``calibrate_detection_threshold``.
        multiplier_grid: Optional override.
    """
    if not 0.0 < target_fpr < 1.0:
        raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")
    if not signals_by_asset:
        raise ValueError("signals_by_asset is empty")

    per_asset_ratios: dict[str, pd.Series] = {}
    for asset, signal in signals_by_asset.items():
        s = signal.dropna().sort_index()
        if s.empty:
            continue
        baseline = _rolling_baseline(s, baseline_window, baseline_statistic, ema_lambda)
        ratio = (s / baseline).dropna()
        if not ratio.empty:
            per_asset_ratios[asset] = ratio

    if not per_asset_ratios:
        return {
            "multiplier": np.nan,
            "realised_fpr": np.nan,
            "target_fpr": target_fpr,
            "n_eval": 0,
            "note": "no asset produced a non-empty ratio series",
        }

    pooled = pd.concat(per_asset_ratios.values()).sort_index()
    if multiplier_grid is None:
        multiplier_grid = np.round(np.arange(1.05, 5.01, 0.05), 4)

    chosen_m = float("nan")
    realised = float("nan")
    for m in multiplier_grid:
        rate = float((pooled > m).mean())
        if rate <= target_fpr:
            chosen_m = float(m)
            realised = rate
            break

    return {
        "multiplier": chosen_m,
        "realised_fpr": realised,
        "target_fpr": target_fpr,
        "baseline_window": baseline_window,
        "baseline_statistic": baseline_statistic,
        "ema_lambda": float(ema_lambda) if baseline_statistic == "ema" else None,
        "n_eval_total": int(len(pooled)),
        "n_eval_per_asset": {a: int(len(r)) for a, r in per_asset_ratios.items()},
        "pool_assets": sorted(per_asset_ratios.keys()),
        "ratio_min": float(pooled.min()),
        "ratio_median": float(pooled.median()),
        "ratio_max": float(pooled.max()),
    }


def calibrate_probability_threshold(
    signal_val: pd.Series,
    target_fpr: float,
    threshold_grid: np.ndarray | None = None,
) -> dict:
    """
    Calibrate a *fixed* probability cutoff p* on a probability signal in [0, 1].

    Picks the smallest p* such that P(signal_t > p*) ≤ target_fpr on val.
    Used for HMM posterior(state=Crisis) and MS-GARCH state mass — signals
    where the multiplier-on-ratio mechanism does not apply.
    """
    if not 0.0 < target_fpr < 1.0:
        raise ValueError(f"target_fpr must be in (0,1), got {target_fpr}")
    s = signal_val.dropna().sort_index()
    if s.empty:
        return {"threshold": float("nan"), "realised_fpr": float("nan"),
                "target_fpr": target_fpr, "n_eval": 0, "note": "empty signal"}
    if threshold_grid is None:
        threshold_grid = np.round(np.arange(0.05, 0.99, 0.01), 4)
    chosen, realised = float("nan"), float("nan")
    for p_star in threshold_grid:
        rate = float((s > p_star).mean())
        if rate <= target_fpr:
            chosen, realised = float(p_star), rate
            break
    return {
        "threshold": chosen,
        "realised_fpr": realised,
        "target_fpr": target_fpr,
        "n_eval": int(len(s)),
        "signal_min": float(s.min()),
        "signal_median": float(s.median()),
        "signal_max": float(s.max()),
    }


def calibrate_probability_threshold_pooled(
    signals_by_asset: dict[str, pd.Series],
    target_fpr: float,
    threshold_grid: np.ndarray | None = None,
) -> dict:
    """Pooled-FPR version: pool all val probability signals, then pick p*."""
    if not signals_by_asset:
        raise ValueError("signals_by_asset is empty")
    parts = [s.dropna() for s in signals_by_asset.values()]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return {"threshold": float("nan"), "realised_fpr": float("nan"),
                "target_fpr": target_fpr, "n_eval_total": 0,
                "note": "no asset produced non-empty signal"}
    pooled = pd.concat(parts).sort_index()
    if threshold_grid is None:
        threshold_grid = np.round(np.arange(0.05, 0.99, 0.01), 4)
    chosen, realised = float("nan"), float("nan")
    for p_star in threshold_grid:
        rate = float((pooled > p_star).mean())
        if rate <= target_fpr:
            chosen, realised = float(p_star), rate
            break
    return {
        "threshold": chosen,
        "realised_fpr": realised,
        "target_fpr": target_fpr,
        "n_eval_total": int(len(pooled)),
        "n_eval_per_asset": {a: int(len(s.dropna())) for a, s in signals_by_asset.items()},
        "pool_assets": sorted(signals_by_asset.keys()),
        "signal_max": float(pooled.max()),
    }


def detection_lead_time_probability(
    signal: pd.Series,
    crisis_start_date: str | pd.Timestamp,
    probability_threshold: float,
    lookback_days: int = 180,
) -> dict:
    """
    Lead time for a probability signal under a fixed cutoff.

    Alarm fires at the first day t in [crisis_start − lookback, crisis_start]
    where ``signal_t > probability_threshold``.
    """
    s = signal.dropna().sort_index()
    crisis_start = pd.Timestamp(crisis_start_date)
    if s.index.tz is not None and crisis_start.tz is None:
        crisis_start = crisis_start.tz_localize(s.index.tz)
    lb_start = crisis_start - pd.Timedelta(days=lookback_days)
    window = s.loc[lb_start:crisis_start]
    if window.empty:
        return {"lead_time_days": np.nan, "probability_threshold": float(probability_threshold),
                "lookback_days": int(lookback_days), "detection_date": None,
                "note": "no signal in lookback window"}
    alarms = window[window > float(probability_threshold)]
    if alarms.empty:
        return {"lead_time_days": 0, "probability_threshold": float(probability_threshold),
                "lookback_days": int(lookback_days), "detection_date": None,
                "note": "no detection in lookback window"}
    detection_date = alarms.index[0]
    return {
        "lead_time_days": int((crisis_start - detection_date).days),
        "probability_threshold": float(probability_threshold),
        "lookback_days": int(lookback_days),
        "detection_date": str(detection_date.date()),
        "signal_at_detection": float(alarms.iloc[0]),
    }


def detection_lead_time(
    risk_signal: pd.Series,
    crisis_start_date: str | pd.Timestamp,
    threshold_multiplier: float,
    baseline_window: int = 252,
    baseline_statistic: str = "median",
    ema_lambda: float = 0.94,
    lookback_days: int = 90,
) -> dict:
    """
    Lead time (calendar days) between first alarm and a known crisis start.

    Implementation aligns with ``calibrate_detection_threshold``: the
    baseline is computed pointwise (rolling median/mean or EWMA) over the
    full signal, then the alarm fires on the first day t inside the
    lookback window where ``signal_t > multiplier * baseline_t``.

    The threshold multiplier is NOT a free hyperparameter — it must come from
    calibration on the validation set. Passing a hardcoded multiplier here is
    a look-ahead smell.

    Args:
        risk_signal: Risk signal (e.g. conditional vol) over the evaluation period.
        crisis_start_date: Known crisis-start date.
        threshold_multiplier: Calibrated multiplier vs. local baseline.
        baseline_window: Window length (days) for the local baseline.
        baseline_statistic: "median" | "mean" | "ema".
        ema_lambda: Decay factor when baseline_statistic == "ema".
        lookback_days: How many calendar days before crisis we are willing to
            count as a "lead" detection. Outside that window, alarms are
            considered unrelated noise and ignored.
    """
    signal = risk_signal.dropna().sort_index()
    crisis_start = pd.Timestamp(crisis_start_date)
    if signal.index.tz is not None and crisis_start.tz is None:
        crisis_start = crisis_start.tz_localize(signal.index.tz)

    baseline = _rolling_baseline(signal, baseline_window, baseline_statistic, ema_lambda)
    ratio = (signal / baseline).dropna()

    lookback_start = crisis_start - pd.Timedelta(days=lookback_days)
    window = ratio.loc[lookback_start:crisis_start]

    if window.empty:
        return {
            "lead_time_days": np.nan,
            "threshold_multiplier": float(threshold_multiplier),
            "lookback_days": int(lookback_days),
            "detection_date": None,
            "note": "no signal in lookback window",
        }

    alarms = window[window > float(threshold_multiplier)]
    if alarms.empty:
        return {
            "lead_time_days": 0,
            "threshold_multiplier": float(threshold_multiplier),
            "lookback_days": int(lookback_days),
            "detection_date": None,
            "note": "no detection in lookback window",
        }

    detection_date = alarms.index[0]
    lead_time = (crisis_start - detection_date).days
    detection_signal = float(signal.loc[detection_date])
    detection_baseline = float(baseline.loc[detection_date])

    return {
        "lead_time_days": int(lead_time),
        "threshold_multiplier": float(threshold_multiplier),
        "lookback_days": int(lookback_days),
        "detection_date": str(detection_date.date()),
        "signal_at_detection": detection_signal,
        "baseline_at_detection": detection_baseline,
        "ratio_at_detection": float(alarms.iloc[0]),
    }
