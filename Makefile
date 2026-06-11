# =========================================================================
# soft-commodities-forecast-benchmark — Makefile
# =========================================================================
# Standard reproducibility entry points. Run `make help` for an overview.

PYTHON ?= python
PIP ?= pip
ASSETS := cocoa coffee sugar cotton

.PHONY: help install dev-install fetch train predict evaluate \
        reproduce reproduce-cocoa reproduce-coffee reproduce-sugar reproduce-cotton \
        test lint format clean

help:
	@echo "soft-commodities-forecast-benchmark"
	@echo ""
	@echo "Setup:"
	@echo "  make install         install runtime dependencies"
	@echo "  make dev-install     install runtime + development dependencies"
	@echo ""
	@echo "Reproduction:"
	@echo "  make reproduce       run the full benchmark for all four commodities"
	@echo "                       and assert results match the stored diagnostics"
	@echo "  make reproduce-cocoa run only one asset (also: -coffee, -sugar, -cotton)"
	@echo ""
	@echo "Pipeline (per asset):"
	@echo "  make train ASSET=cocoa     fit the GJR-GARCH-t model"
	@echo "  make predict ASSET=cocoa   walk-forward forecast"
	@echo "  make evaluate ASSET=cocoa  VaR backtests + diagnostics JSON"
	@echo ""
	@echo "Quality:"
	@echo "  make test            run pytest"
	@echo "  make lint            ruff check"
	@echo "  make format          black + ruff format"
	@echo "  make clean           remove build artefacts and caches"

install:
	$(PIP) install -e .

dev-install:
	$(PIP) install -e ".[dev]"

ASSET ?= cocoa

train:
	$(PYTHON) -m benchmark.train --asset $(ASSET)

predict:
	$(PYTHON) -m benchmark.predict --asset $(ASSET)

evaluate:
	$(PYTHON) -m benchmark.evaluate --asset $(ASSET)

hmm:
	$(PYTHON) -m benchmark.hmm_regime --asset $(ASSET)

hmm-evaluate:
	$(PYTHON) -m benchmark.hmm_evaluate

hmm-all:
	for a in $(ASSETS); do $(PYTHON) -m benchmark.hmm_regime --asset $$a; done
	$(PYTHON) -m benchmark.hmm_evaluate

reproduce: reproduce-cocoa reproduce-coffee reproduce-sugar reproduce-cotton
	@echo ""
	@echo "All four commodities reproduced successfully."

reproduce-cocoa:
	$(PYTHON) -m benchmark.reproduce --asset cocoa

reproduce-coffee:
	$(PYTHON) -m benchmark.reproduce --asset coffee

reproduce-sugar:
	$(PYTHON) -m benchmark.reproduce --asset sugar

reproduce-cotton:
	$(PYTHON) -m benchmark.reproduce --asset cotton

test:
	pytest -q

lint:
	ruff check src tests

format:
	black src tests
	ruff format src tests

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
