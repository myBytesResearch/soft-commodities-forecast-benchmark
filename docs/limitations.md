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

# Limitations

What this backtest **does not** deliver. A standalone documentation
of the known weaknesses of the implementation, separated from the
methodology description. Anyone planning further work on top of our
results should read this first.

---

## 1 · A single stress episode per commodity

The pre-crisis-window evaluation per commodity rests on a single
stress episode (2023/24 cocoa supply shock, 2024 coffee Brazil
drought, 2023 sugar India export curb, 2022 cotton supply shock).
**One episode is not a sample.** The lead-time statement *"zero
lead time"* refers to that one episode per commodity. Validation on
further stress episodes is pending and is the subject of the next
research stage.

## 2 · Look-ahead bias in event definition

The event windows were chosen manually from market history. We know
*now* that the 2023/24 cocoa spike occurred; a model that should have
detected it *then* would not have known the date. This asymmetry is
deliberate in the procedure - the lead-time statement is a
**conditional-on-event** finding, not a real-time out-of-sample
detection.

## 3 · GJR-GARCH specification is fixed

We use GJR(1,1)-t for all four commodities. A specification search
(p, o, q or distribution choice) could yield marginally better fits
for individual commodities. We deliberately abstained from it because
specification searches over walk-forward backtests are a known source
of data snooping. Anyone experimenting with alternative
specifications should do so in a separate repository with their own
snapshot pins.

## 4 · Yahoo Finance data quality

Yahoo data for continuous futures are not Tier-1 market data. They
are less clean than ICE Data Services or Bloomberg feeds: occasional
retroactive revisions, missing half-day trading sessions, time-zone
inconsistencies on Asian sessions. For **methodological demonstration**
the quality is sufficient. For **production risk applications** a
licensed vendor feed should be used (see `LICENSES.md`).

## 5 · Continuous-futures roll mechanics

Yahoo's continuous-futures series use a proprietary roll method that
is not publicly documented. Roll adjustments can create short-term
volatility jumps that are roll artefacts, not market signal. We do
not filter these explicitly. For very fine tail-risk analysis a
self-implemented roll method with documented adjustment rules is
preferable.

## 6 · Backend dependence of the optimisation

The GARCH MLE optimisation in the `arch` package uses SciPy
optimisers (default: SLSQP). Depending on the BLAS backend (OpenBLAS,
MKL, Accelerate) the estimated parameters can differ in the eighth
decimal place. We set tolerances in `data_snapshot.json` so that this
drift is tolerated. For bit-exact reproducibility a reproducible
BLAS configuration is required.

## 7 · What this document does not cover

- Comparison against alternative volatility models (EGARCH, FIGARCH,
  Realized GARCH, MS-GARCH, HMM, GARCH-MIDAS) - material for the
  next research stages
- Cross-correlations between the four commodities - a separate
  research topic
- Options-market validation against implied volatility - see the
  forthcoming realised-versus-implied volatility note for procurement
- Application to procurement hedging - methodological material, not
  an operational hedging recommendation

---

This list is not exhaustive. Anyone identifying an additional
limitation is invited to open an issue in the repository.
