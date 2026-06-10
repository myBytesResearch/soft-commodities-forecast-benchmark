<!--
=============================================================================
                    ____        __
   ____ ___  __  __/ __ )__  __/ /____  _____
  / __ `__ \/ / / / __  / / / / __/ _ \/ ___/
 / / / / / / /_/ / /_/ / /_/ / /_/  __(__  )
/_/ /_/ /_/\__, /_____/\__, /\__/\___/____/
          /____/      /____/

 myBytes.com
 Copyright (c) 2026 myBytes GmbH. All rights reserved.
=============================================================================
-->

# Methodology

Full description of the model specification, data preparation,
walk-forward mechanics and evaluation discipline. This document is
the companion to the repository code; the operational summary lives
in the [README](../README.md).

---

## 1 · Model specification

### 1.1 GJR-GARCH(1,1) with Student-t innovations

Mean and volatility equations:

```
r_t = mu + eps_t

sigma_t^2 = omega + (alpha + gamma * 1[eps_{t-1} < 0]) * eps_{t-1}^2 + beta * sigma_{t-1}^2

eps_t / sigma_t ~ Student-t(nu)
```

We use the standard implementation from the
[`arch` package](https://arch.readthedocs.io/) (Sheppard 2024). Returns
are scaled to percent (×100) for numerical stability, following the
`arch` convention.

The `gamma` parameter captures the asymmetric volatility response to
negative returns. In equity markets `gamma` is typically positive
(leverage effect); in soft commodities our estimates often show the
inverse effect — see `results/<asset>_diagnostics.json` for the
empirical values per commodity.

### 1.2 Pre-registered specification

We use the **same** specification for all four commodities:
GJR-GARCH(1,1) with Student-t innovations, constant mean,
Bollerslev–Wooldridge robust standard errors. There is **no
specification search** — no hyperparameter tuning, no per-asset p/o/q
variations. This is a deliberate discipline. Specification searches
over walk-forward backtests are a known source of look-ahead bias and
data snooping.

### 1.3 What the specification is not

- Not Markov-switching GARCH (belongs to the next research stage)
- Not EVT-POT on residuals (separate stage)
- Not GARCH-MIDAS with exogenous low-frequency factors (weather, COT,
  ENSO)
- Not foundation models on volatility targets

Each of these layers is planned in the myBytes research programme
and will receive its own companion repository.

---

## 2 · Data and preparation

### 2.1 Data source

Daily closing prices for four ICE Continuous Futures, fetched via
`yfinance` (Yahoo Finance):

- `CC=F` — ICE Cocoa
- `KC=F` — ICE Coffee (Arabica)
- `SB=F` — ICE Sugar No. 11
- `CT=F` — ICE Cotton No. 2

License caveat: Yahoo Finance Terms of Service prohibit data
redistribution. We ship code, not data. Details:
[`LICENSES.md`](../LICENSES.md).

### 2.2 Return calculation

Logarithmic returns, scaled to percent:

```
r_t = 100 * log(P_t / P_{t-1})
```

Missing days (weekends, holidays) are dropped, not interpolated.

### 2.3 Training and test periods

| Period | Start | End |
|---|---|---|
| Training | 2000-01-01 | 2018-12-28 |
| Test     | 2019-01-01 | 2024-12-31 |

Walk-forward over the test period with an expanding training window
(initial window ≈ 10 years, refit every 21 trading days). Forecast
horizon: 1 day ahead.

---

## 3 · Evaluation

### 3.1 VaR backtests

Per walk-forward window we evaluate the conditional 1-day VaR
against the realised returns:

- **Kupiec POF** ([Kupiec 1995](https://doi.org/10.3905/jod.1995.407942))
  tests violation frequency against the nominal level
- **Christoffersen CC** ([Christoffersen 1998](https://doi.org/10.2307/2527341))
  additionally tests independence of violations
- Levels: 95 % and 99 %

We report violation share, test statistic, p-value and rejection
flag per test.

### 3.2 Pre-crisis-window VaR coverage

**A separate discipline alongside aggregate coverage.** GARCH models
adapt quickly after a crisis, so aggregate VaR coverage can look fine
even though the model saw nothing *before* the crisis. We therefore
evaluate a **pre-crisis window** of roughly 126 trading days before
each known stress event, separately:

| Commodity | Stress event | Window |
|---|---|---|
| Cocoa  | 2023/24 supply shock     | 2022-09-01 → 2023-03-05 |
| Coffee | 2024 Brazil drought      | 2023-09-02 → 2024-03-05 |
| Sugar  | 2023 India export curb   | 2023-04-01 → 2023-10-01 |
| Cotton | 2022 supply shock        | 2022-02-01 → 2022-08-01 |

On these windows we apply the same Kupiec POF and Christoffersen CC
tests. Results live in `results/evaluation_<asset>.json` under
`pre_crisis_var_coverage`.

### 3.3 Lead-time statement

For the lead-time statement highlighted in the companion article
(*"zero lead time"*) we measure whether the conditional volatility
forecast rises significantly above its 60-day trend in the 30 trading
days before the spike day. For single-layer GJR-GARCH on the
2023/24 cocoa spike the answer is no — volatility rises only *on*
the spike day. This is the expected finding because GJR-GARCH carries
no regime detection.

### 3.4 R²_OOS against squared returns

We additionally report the out-of-sample R² of the conditional
variance forecast against realised squared returns. Values between
−0.01 and +0.03 are expected in the volatility forecasting literature
and are not a model failure but a consequence of the high noise
component of squared returns as a volatility proxy
([Andersen/Bollerslev 1998](https://www.jstor.org/stable/2527343)).

---

## 4 · Reproducibility

### 4.1 Snapshot pinning

Reproducibility requires a fixed end-date for the Yahoo data. We pin
this in `data_snapshot.json` and check the fresh fit parameters
against `results/<asset>_diagnostics.json` within a documented
tolerance.

Tolerances:
- Parameter drift: 1 % relative
- Log-likelihood drift: 0.5 % relative

Tolerances are deliberately small but non-zero. Yahoo occasionally
revises historical closing prices retroactively, and arch optimisation
can differ in the eighth decimal place depending on the BLAS backend.
Larger drift is a genuine signal, not noise.

### 4.2 Seed discipline

All stochastic components (NumPy, Python `random`) are seeded at 42.
GARCH optimisation itself is deterministic for identical data and
identical arch version.

### 4.3 MLflow tracking

A MLflow run is created per walk-forward refit, with configuration
hash, data snapshot hash, all parameters, all metrics and forecast
Parquet files as artefacts. Default backend is a local SQLite file
(`mlflow.db`); remote backends can be set via `MLFLOW_TRACKING_URI`
(see `.env.example`).

---

## 5 · Truth-check protocol

This methodology is documented under the myBytes truth-check protocol
(seven steps: claim extraction, classification, anchor mapping,
reproducibility, steel-man, limitations, independent review). The
status of the seven steps for the companion article is reported in the
article's frontmatter.

→ [Truth-check protocol (German)](https://mybytes.com/research/truth-check-protocol)

---

## 6 · Reading list

1. [Glosten, Jagannathan, Runkle 1993, *On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks*](https://www.jstor.org/stable/2329067) — GJR-GARCH original
2. [Bollerslev 1986, *Generalized Autoregressive Conditional Heteroskedasticity*](https://doi.org/10.1016/0304-4076(86)90063-1) — GARCH original
3. [Andersen/Bollerslev 1998, *Answering the Skeptics: Yes, Standard Volatility Models Do Provide Accurate Forecasts*](https://www.jstor.org/stable/2527343)
4. [Kupiec 1995, *Techniques for Verifying the Accuracy of Risk Measurement Models*](https://doi.org/10.3905/jod.1995.407942)
5. [Christoffersen 1998, *Evaluating Interval Forecasts*](https://doi.org/10.2307/2527341)
6. [Sheppard 2024, *arch package documentation*](https://arch.readthedocs.io/)
