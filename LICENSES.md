# Licenses - Code, Data, Models, Libraries

This repository ships **code only**. Data is fetched at runtime from
Yahoo Finance via `yfinance`. Reading this file is part of the
reproduction procedure.

## 1 · Repository code

**License:** MIT (see `LICENSE`).
You may use, modify, redistribute, and commercialise the code, provided
you preserve the copyright notice. There is no warranty.

## 2 · Data sources

### 2.1 ICE Continuous Futures via `yfinance`

- **Source:** Yahoo Finance (CC=F, KC=F, SB=F, CT=F).
- **Fetched via:** `yfinance` (third-party Python package; not affiliated
  with Yahoo).
- **Licensing of the data itself:** Yahoo Finance Terms of Service
  prohibit redistribution of fetched data. You may use the data for
  personal, non-commercial research; commercial use generally requires
  a separate license from a market-data vendor (Bloomberg, Refinitiv,
  ICE Data Services, …).
- **This repository's posture:** we ship **code only**, never data
  snapshots. The user fetches data themselves under their own terms.
- **For commercial users:** consult Yahoo Finance ToS and ICE
  licensing. For genuine production volatility work, license the
  underlying ICE Continuous Futures data from ICE Data Services or an
  authorised redistributor.

### 2.2 Reproducibility snapshot

The `data_snapshot.json` file in the repository root pins the dataset
end-date and records the expected first-pass statistics. Running
`make reproduce` re-fetches the data at the pinned end-date and asserts
the model output against the stored diagnostics. No raw data is
distributed; only the hash of the expected outputs.

## 3 · Python libraries (third-party dependencies)

The benchmark depends on the following libraries. Each retains its own
license; see the library's own documentation for full terms. All listed
licenses are permissive (MIT, BSD, Apache-2.0) and allow commercial
use:

| Library | License | Purpose |
|---|---|---|
| numpy, pandas, scipy | BSD-3-Clause | numerical and tabular foundation |
| matplotlib, seaborn | PSF / BSD-3 | plotting |
| arch | NCSA | GARCH-family estimator |
| statsmodels | BSD-3 | statistical tests |
| yfinance | Apache-2.0 | Yahoo Finance fetcher (data ToS apply, see §2.1) |
| scikit-learn | BSD-3 | helper utilities |
| mlflow | Apache-2.0 | experiment tracking |
| pyyaml | MIT | configuration parsing |
| loguru | MIT | logging |
| python-dotenv | BSD-3 | environment loading |
| pyarrow | Apache-2.0 | Parquet I/O |

## 4 · Method references

The methods used in this benchmark are published, peer-reviewed
research. We cite the original authors in the methodology note
(`docs/methodology.md`) and in the companion article on
`mybytes.com`. No proprietary algorithms are used.

## 5 · What this means for you, the reader

- **Personal, academic, methodological reproduction:** permitted under
  the MIT license and Yahoo Finance ToS, with the caveats above.
- **Commercial production use:** you may use the code, but you must
  license the underlying market data from an authorised redistributor.
- **Republishing the data:** not permitted. Republishing your derived
  *results* (diagnostics, plots, articles) is permitted.

If any of this is unclear for your specific use case, consult your own
legal counsel. This file is documentation, not legal advice.
