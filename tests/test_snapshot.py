"""
Tests for the reproducibility snapshot file.

These tests run without external dependencies and verify that
`data_snapshot.json` is internally consistent.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = REPO_ROOT / "data_snapshot.json"
ASSETS = ["cocoa", "coffee", "sugar", "cotton"]


@pytest.fixture(scope="module")
def snapshot() -> dict:
    with SNAPSHOT_PATH.open() as f:
        return json.load(f)


def test_snapshot_exists() -> None:
    assert SNAPSHOT_PATH.exists()


def test_snapshot_has_all_four_assets(snapshot: dict) -> None:
    for asset in ASSETS:
        assert asset in snapshot["tickers"], f"missing ticker for {asset}"
        assert asset in snapshot["expected_first_pass_stats"], f"missing stats for {asset}"


def test_snapshot_dates_parseable(snapshot: dict) -> None:
    datetime.strptime(snapshot["snapshot_end_date"], "%Y-%m-%d")
    datetime.strptime(snapshot["fetch_start_date"], "%Y-%m-%d")


def test_snapshot_tolerance_reasonable(snapshot: dict) -> None:
    tol = snapshot["tolerance"]
    assert 0 < tol["parameter_relative_diff"] < 0.1, (
        "parameter tolerance should be tight (< 10 %) but not zero"
    )
    assert 0 < tol["loglikelihood_relative_diff"] < 0.05, (
        "log-likelihood tolerance should be tighter than parameter tolerance"
    )


@pytest.mark.parametrize("asset", ASSETS)
def test_snapshot_ticker_format(asset: str, snapshot: dict) -> None:
    ticker = snapshot["tickers"][asset]["yahoo_symbol"]
    assert ticker.endswith("=F"), f"{asset} ticker `{ticker}` not in Yahoo continuous-futures format"
