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

# soft-commodities-forecast-benchmark

**Companion repository to the myBytes Research methodology note on
GJR-GARCH baseline backtests across four soft commodities.**

→ Methodology note (German): https://mybytes.com/research/garch-soft-commodities-baseline-backtest

---

## Scope

This repository contains the code, configurations and reproduction
tooling for a multi-year walk-forward backtest of a classical
GJR-GARCH(1,1) model with Student-t innovations on four ICE
Continuous Futures:

- **ICE Cocoa** (CC=F)
- **ICE Coffee** (KC=F, Arabica)
- **ICE Sugar** (SB=F, No. 11)
- **ICE Cotton** (CT=F, No. 2)

GJR-GARCH is the industry-standard model for asymmetric volatility.
We build it deliberately as the baseline layer of a multi-year
research programme on soft-commodity volatility. What this baseline
delivers and what it structurally does not deliver is the subject of
the linked methodology note.

## What this repository reproduces

- All per-commodity model parameters cited in the note (mean, omega,
  alpha, gamma, beta, nu) under identical walk-forward split mechanics
- All VaR backtests (Kupiec-POF, Christoffersen-CC) on the aggregate
  test period 2019–2024
- Pre-crisis-window VaR coverage as a separate discipline (aggregate
  coverage measures backward-looking adaptation; the pre-crisis
  window measures what was visible *before* the stress event)
- The lead-time finding highlighted in the note (zero lead time for
  single-layer GJR-GARCH on the 2023/24 cocoa supply shock and on the
  corresponding stress episodes for coffee, sugar and cotton)

One-command reproduction:

```bash
make reproduce
```

The command fetches the data fresh via `yfinance` at a snapshot
end-date pinned in `data_snapshot.json`, runs the full pipeline
(training, walk-forward prediction, evaluation) and asserts the
freshly estimated parameters and log-likelihoods against the values
stored in `results/<asset>_diagnostics.json`. The tolerance for
small Yahoo-Finance data drifts is documented in the snapshot file.

## What this repository does not contain

Three points explicitly:

1. **No data.** Yahoo Finance Terms of Service prohibit redistribution
   of fetched data. You fetch the data yourself via `yfinance`. The
   code is pinned to a fixed snapshot end-date so reproduction stays
   deterministic. License details: [`LICENSES.md`](LICENSES.md).
2. **No follow-on layers.** HMM regime detection, GARCH-MIDAS with
   weather and COT data, foundation models for volatility targets —
   all of these are planned in the myBytes research programme, none
   of them belong in this baseline repository. When those layers are
   built, they will land in their own companion repositories.
3. **Not an investment or hedging recommendation.** The backtest is
   methodological material, not a trading system. See the disclaimer
   at the end.

## Quickstart in 10 minutes

Prerequisite: Python 3.11 or 3.12, a fresh virtual environment.

```bash
# 1. Clone the repository and install runtime dependencies
git clone https://github.com/myBytesResearch/soft-commodities-forecast-benchmark.git
cd soft-commodities-forecast-benchmark
make install

# 2. Derive your environment file from the example
cp .env.example .env
# (defaults work for most setups; see comments inside .env.example)

# 3. Reproduce one commodity
make reproduce-cocoa

# 4. Reproduce all four
make reproduce
```

If the run finishes with `reproduction match for cocoa` (and the
equivalent for the other three commodities), reproduction succeeded.
On drift warnings, first verify that your snapshot end-date matches
`data_snapshot.json`.

## Repository layout

```
src/benchmark/        Model code (train, predict, evaluate, reproduce)
configs/              One YAML per commodity plus base.yaml
results/              Diagnostics JSON per commodity (gold-standard values)
notebooks/            Research notebooks with the full methodological walk
docs/                 Standalone methodology and limitations documents
tests/                Unit and integration tests
artifacts/            Output of fresh runs (gitignored)
data_snapshot.json    Reproducibility pin (tickers, end-date, tolerance)
```

## Research notebooks

`notebooks/` contains a detailed research-style notebook walking from
data exploration to VaR-discipline evaluation. It is deliberately not
a terse tutorial: it shows stylized facts, diagnostics, the model
fit, walk-forward backtest mechanics, VaR tests, the pre-crisis
window and the methodological self-criticism side by side.

## On the methodology note

The companion note on mybytes.com explains:

- why classical GARCH passes the VaR discipline and simultaneously
  produces zero lead time before each of the four stress episodes
- why this is not a bug but the expected finding
- which second model layer is needed for early warning, and on what
  schedule we plan to publish it

→ [The single-GARCH limit on soft commodities (German)](https://mybytes.com/research/garch-soft-commodities-baseline-backtest)
→ [Truth-check protocol (German)](https://mybytes.com/research/truth-check-protocol)

## Disclaimer

This implementation and the backtest figures it produces describe a
walk-forward backtest from our own research practice. They are not an
investment recommendation and not a hedging recommendation. The
quoted performance figures refer to a specific test-setup
configuration and are not transferable without further analysis to
other use cases.
