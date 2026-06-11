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
#  File: hmm_regime.py | Project: soft-commodities-forecast-benchmark
#  Author: Guido Winger
# =========================================================================

"""Layer 2 - Gaussian HMM regime detection (model 17).

Walk-forward regime filtering on daily log returns. The model answers
exactly one pre-registered question: does a regime layer deliver a
detection lead time > 0 days before the four known stress episodes
where the GJR-GARCH-t baseline (layer 1) delivered zero?

Pre-registered specification (fixed before any run; see
configs/hmm.yaml and configs/global.yaml `hmm_signal`):

- Gaussian HMM, K = 3 regimes (calm / elevated / stress), diagonal
  covariance on the single return series.
- Stress regime := the state with the largest fitted variance.
- Signal: filtered posterior P(stress | data up to t) - strictly
  causal (forward filter only, no smoothing).
- Walk-forward protocol identical to layer 1: expanding window,
  initial >= 2520 trading days, refit every 21 trading days,
  the filter runs out-of-sample between refits.
- Primary detection statistic: logit of the posterior, evaluated
  with the same ratio-vs-EMA-baseline mechanics and the same
  pre-registered endpoint (sensitive, 10 % FPR, 180d lookback,
  pooled calibration) as layer 1. Raw posterior with fixed
  threshold is reported as sensitivity only.

No per-asset specification search. One specification, four assets,
seed 42.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from hmmlearn.hmm import GaussianHMM
from loguru import logger

from benchmark.preprocessing import compute_returns, load_global_config
from benchmark.yfinance_loader import YFinanceLoader

RESULTS_DIR = Path("results")
SEED = 42


# ------------------------------------------------------------------ #
# Fitting                                                            #
# ------------------------------------------------------------------ #
def fit_hmm(returns: np.ndarray, n_states: int, seed: int = SEED) -> GaussianHMM:
    """Fit a Gaussian HMM with EM on a 1-D return array."""
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=200,
        tol=1e-4,
        random_state=seed,
        init_params="stmc",
    )
    model.fit(returns.reshape(-1, 1))
    return model


def stress_state_index(model: GaussianHMM) -> int:
    """The stress regime is the state with the largest fitted variance."""
    return int(np.argmax(model.covars_.ravel()))


def filtered_stress_posterior(
    model: GaussianHMM, returns: np.ndarray, stress_idx: int
) -> np.ndarray:
    """Strictly causal filtered posterior P(state = stress | r_1..r_t).

    Own scaled forward filter (no hmmlearn private API): the
    normalised forward variables alpha_t are exactly the filtered
    state probabilities. Emission densities are univariate Gaussians
    with the fitted means and (diagonal) variances.
    """
    r = returns.ravel()
    means = model.means_.ravel()
    variances = model.covars_.reshape(model.n_components, -1)[:, 0]
    startprob = model.startprob_
    transmat = model.transmat_

    # emission likelihoods N(r_t; mu_k, sigma2_k), shape (T, K)
    norm = 1.0 / np.sqrt(2.0 * np.pi * variances)
    emit = norm * np.exp(-0.5 * (r[:, None] - means[None, :]) ** 2 / variances[None, :])
    emit = np.maximum(emit, 1e-300)

    alpha = startprob * emit[0]
    alpha /= alpha.sum()
    out = np.empty(len(r))
    out[0] = alpha[stress_idx]
    for t in range(1, len(r)):
        alpha = (alpha @ transmat) * emit[t]
        alpha /= alpha.sum()
        out[t] = alpha[stress_idx]
    return out


# ------------------------------------------------------------------ #
# Walk-forward                                                       #
# ------------------------------------------------------------------ #
def walk_forward_posterior(
    returns: pd.Series,
    initial_window: int,
    refit_every: int,
    n_states: int,
) -> pd.DataFrame:
    """Expanding-window walk-forward regime filter.

    At each refit date the HMM is re-estimated on data up to that
    date. Between refits the *frozen* model filters incoming returns
    causally. Output: one filtered stress posterior per out-of-sample
    day, plus refit bookkeeping.
    """
    r = returns.dropna()
    values = r.to_numpy()
    n = len(values)
    if n <= initial_window:
        raise ValueError(f"need more than {initial_window} observations, got {n}")

    out_idx, out_post, out_refit = [], [], []
    refit_count = 0

    for block_start in range(initial_window, n, refit_every):
        # refit on all data strictly before the block
        model = fit_hmm(values[:block_start], n_states=n_states)
        stress_idx = stress_state_index(model)
        refit_count += 1
        block_end = min(block_start + refit_every, n)
        # one causal filter pass over history + block with the frozen
        # model; the filtered posterior at each in-block day uses only
        # data up to that day (forward filter is causal by construction)
        post = filtered_stress_posterior(model, values[:block_end], stress_idx)
        for t in range(block_start, block_end):
            out_idx.append(r.index[t])
            out_post.append(post[t])
            out_refit.append(refit_count)

    return pd.DataFrame(
        {"posterior_stress": out_post, "refit_id": out_refit},
        index=pd.DatetimeIndex(out_idx, name="date"),
    )


# ------------------------------------------------------------------ #
# CLI                                                                #
# ------------------------------------------------------------------ #
def main() -> None:
    parser = argparse.ArgumentParser(description="Layer-2 Gaussian HMM regime filter.")
    parser.add_argument("--config", type=str, default="configs/hmm.yaml")
    parser.add_argument("--global-config", type=str, default="configs/global.yaml")
    parser.add_argument("--asset", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    gcfg = load_global_config(args.global_config)

    n_states = int(cfg["model"]["n_states"])
    initial_window = int(gcfg["backtest"]["initial_window_days"])
    refit_every = int(gcfg["backtest"]["refit_frequency_days"])

    np.random.seed(SEED)

    loader = YFinanceLoader(cache_dir="data/raw/prices")
    df = loader.load(symbol=args.asset, start=gcfg["splits"]["train_start"])
    df = df.loc[: gcfg["splits"]["test_end"]]
    returns = compute_returns(df["close"], method=cfg["data"]["return_method"]) * 1.0

    logger.info(
        f"{args.asset}: {len(returns)} returns, walk-forward from obs {initial_window}, "
        f"refit every {refit_every}d, K={n_states}"
    )
    t0 = datetime.now()
    wf = walk_forward_posterior(returns, initial_window, refit_every, n_states)
    elapsed = (datetime.now() - t0).total_seconds()
    logger.info(f"{args.asset}: walk-forward done in {elapsed:.1f}s, {wf['refit_id'].max()} refits")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_parquet = RESULTS_DIR / f"hmm_posterior_{args.asset}.parquet"
    wf.to_parquet(out_parquet)

    meta = {
        "asset": args.asset,
        "model": "gaussian_hmm_k3",
        "n_states": n_states,
        "covariance_type": "diag",
        "seed": SEED,
        "initial_window_days": initial_window,
        "refit_frequency_days": refit_every,
        "n_oos_days": int(len(wf)),
        "n_refits": int(wf["refit_id"].max()),
        "walkforward_seconds": round(elapsed, 1),
        "first_oos_date": str(wf.index[0].date()),
        "last_oos_date": str(wf.index[-1].date()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    out_meta = RESULTS_DIR / f"hmm_meta_{args.asset}.json"
    out_meta.write_text(json.dumps(meta, indent=2))
    logger.success(f"✓ {args.asset}: {out_parquet} + {out_meta}")


if __name__ == "__main__":
    main()
