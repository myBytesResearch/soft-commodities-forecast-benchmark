"""
Integrity tests for repository configurations.

These tests do not require external data and run in milliseconds.
They check that every commodity has a complete configuration tree
and that the stored diagnostics file structure is intact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"
ASSETS = ["cocoa", "coffee", "sugar", "cotton"]


@pytest.fixture(scope="module")
def base_config() -> dict:
    with (CONFIG_DIR / "base.yaml").open() as f:
        return yaml.safe_load(f)


def test_base_config_has_required_sections(base_config: dict) -> None:
    for section in ("model", "data", "training", "backtest", "mlflow"):
        assert section in base_config, f"missing section `{section}` in base.yaml"


def test_base_config_model_is_gjr_garch_t(base_config: dict) -> None:
    assert base_config["model"]["name"] == "gjr_garch_t"
    assert base_config["model"]["family"] == "garch"
    assert base_config["model"]["p"] == 1
    assert base_config["model"]["o"] == 1
    assert base_config["model"]["q"] == 1
    assert base_config["model"]["dist"] == "studentst"


def test_base_config_returns_scale(base_config: dict) -> None:
    assert base_config["data"]["return_scale"] == 100, (
        "Returns must be scaled to percent for arch-package numerical stability"
    )


def test_base_config_random_seed_default(base_config: dict) -> None:
    assert base_config["random_seed"] == 42


@pytest.mark.parametrize("asset", ASSETS)
def test_asset_config_exists(asset: str) -> None:
    path = CONFIG_DIR / f"{asset}.yaml"
    assert path.exists(), f"missing config for {asset}"


@pytest.mark.parametrize("asset", ASSETS)
def test_asset_config_has_required_fields(asset: str) -> None:
    with (CONFIG_DIR / f"{asset}.yaml").open() as f:
        cfg = yaml.safe_load(f)
    assert "asset" in cfg
    assert cfg["asset"]["name"] == asset
    assert cfg["asset"]["ticker"].endswith("=F"), (
        "tickers follow Yahoo's continuous-futures convention `XX=F`"
    )
    assert "pre_crisis_window" in cfg
    pcw = cfg["pre_crisis_window"]
    assert pcw["event_id"], "pre-crisis window must name the event"
    assert pcw["window_start"]
    assert pcw["window_end"]


@pytest.mark.parametrize("asset", ASSETS)
def test_diagnostics_file_exists(asset: str) -> None:
    path = RESULTS_DIR / f"{asset}_diagnostics.json"
    assert path.exists(), f"stored diagnostics missing for {asset}"


@pytest.mark.parametrize("asset", ASSETS)
def test_diagnostics_file_has_required_keys(asset: str) -> None:
    path = RESULTS_DIR / f"{asset}_diagnostics.json"
    with path.open() as f:
        d = json.load(f)
    for key in ("asset", "stylized_facts", "arch_lm_test", "model_params", "loglikelihood"):
        assert key in d, f"diagnostics for {asset} missing key `{key}`"
    assert d["asset"] == asset


@pytest.mark.parametrize("asset", ASSETS)
def test_diagnostics_arch_effects_detected(asset: str) -> None:
    """Every soft commodity must show ARCH effects — otherwise GARCH is not justified."""
    path = RESULTS_DIR / f"{asset}_diagnostics.json"
    with path.open() as f:
        d = json.load(f)
    assert d["arch_lm_test"]["arch_effects_detected"] is True, (
        f"{asset}: ARCH-LM test did not detect ARCH effects — GARCH modelling not justified here"
    )
    assert d["arch_lm_test"]["lm_pvalue"] < 0.05, (
        f"{asset}: ARCH-LM p-value not < 0.05; GARCH modelling not justified"
    )


@pytest.mark.parametrize("asset", ASSETS)
def test_diagnostics_garch_persistence_below_unity(asset: str) -> None:
    """Persistence (alpha + 0.5*gamma + beta) below 1 means the variance process is stationary."""
    path = RESULTS_DIR / f"{asset}_diagnostics.json"
    with path.open() as f:
        d = json.load(f)
    persistence = d.get("persistence")
    assert persistence is not None, f"{asset}: persistence not recorded"
    assert persistence < 1.0, (
        f"{asset}: persistence {persistence:.6f} ≥ 1 — non-stationary GARCH, results are suspect"
    )
