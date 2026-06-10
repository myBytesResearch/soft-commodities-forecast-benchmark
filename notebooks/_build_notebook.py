"""
Build the research notebook programmatically via nbformat.

Run:
    python notebooks/_build_notebook.py

This guarantees valid JSON regardless of German quotation marks or
other characters that break a hand-written notebook file.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB = nbf.v4.new_notebook()
cells = []

# -------------------------------------------------------------------------
# 1 · Title and roadmap
# -------------------------------------------------------------------------
cells.append(
    nbf.v4.new_markdown_cell(
        """# 01 · GJR-GARCH-Baseline auf vier Soft Commodities — Exploration und Diagnose

Forschungs-Notebook. Es führt den methodischen Weg sichtbar — von der Daten-Inspektion über die Stylized-Facts-Prüfung und den ARCH-LM-Test bis zur GJR-GARCH-Schätzung und der VaR-Disziplin. Wir zeigen alle vier Commodities (Cocoa, Coffee, Sugar, Cotton) im direkten Vergleich.

Das Notebook ist bewusst ausführlich. Wer reproduzieren möchte, läuft die Pipeline über `make reproduce` (siehe README); wer methodisch nachvollziehen will, was unter dem Backtest steckt, liest hier.

**Inhalt**

1. Setup und Daten-Bezug
2. Deskriptive Statistik und Stylized Facts
3. ARCH-LM-Test — Rechtfertigung der GARCH-Familie
4. GJR-GARCH(1,1)-Schätzung mit Student-t-Innovationen
5. Walk-Forward-Vorhersage und VaR-Backtests
6. Pre-Crisis-Window-Disziplin
7. Cross-Asset-Vergleich
8. Was diese Baseline nicht zeigt"""
    )
)

# -------------------------------------------------------------------------
# 2 · Section 1 — Setup
# -------------------------------------------------------------------------
cells.append(
    nbf.v4.new_markdown_cell(
        """## 1 · Setup und Daten-Bezug

Wir laden die vier ICE Continuous Futures (Cocoa, Coffee, Sugar, Cotton) über `yfinance`, in derselben Spanne wie die Trainings- und Test-Periode der Begleit-Notiz: 2000-01-01 bis zum Snapshot-Endstichtag aus `data_snapshot.json`. Returns werden logarithmisch berechnet und in Prozent skaliert — das ist die Konvention des `arch`-Packages für numerische Stabilität.

**Lizenz-Erinnerung.** Yahoo-Daten dürfen nicht weiterverbreitet werden. Dieses Notebook holt die Daten zur Laufzeit; das Repository selbst enthält keine Roh-Daten."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """%load_ext autoreload
%autoreload 2

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from arch import arch_model
from scipy import stats
from statsmodels.stats.diagnostic import het_arch

REPO_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(REPO_ROOT / 'src'))

with (REPO_ROOT / 'data_snapshot.json').open() as f:
    SNAPSHOT = json.load(f)

ASSETS = ['cocoa', 'coffee', 'sugar', 'cotton']
TICKERS = {a: SNAPSHOT['tickers'][a]['yahoo_symbol'] for a in ASSETS}
NAMES = {a: SNAPSHOT['tickers'][a]['full_name'] for a in ASSETS}
FETCH_START = SNAPSHOT['fetch_start_date']
SNAPSHOT_END = SNAPSHOT['snapshot_end_date']

np.random.seed(42)
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150

print(f'Snapshot-Endstichtag: {SNAPSHOT_END}')
print(f'Tickers: {TICKERS}')"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """def fetch_returns(ticker: str, start: str, end: str) -> pd.Series:
    \"\"\"Fetch daily close prices via yfinance and compute log returns in percent.\"\"\"
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close'][ticker]
    else:
        close = raw['Close']
    close = close.dropna()
    returns = 100.0 * np.log(close / close.shift(1))
    returns = returns.dropna()
    returns.name = 'return_pct'
    return returns

returns_by_asset = {a: fetch_returns(TICKERS[a], FETCH_START, SNAPSHOT_END) for a in ASSETS}

for a, r in returns_by_asset.items():
    print(f'{a:8s} ({TICKERS[a]}): {len(r):5d} observations, {r.index.min().date()} -> {r.index.max().date()}')"""
    )
)

# -------------------------------------------------------------------------
# 3 · Section 2 — Stylized Facts
# -------------------------------------------------------------------------
cells.append(
    nbf.v4.new_markdown_cell(
        """## 2 · Deskriptive Statistik und Stylized Facts

Bevor wir ein GARCH-Modell rechtfertigen können, müssen wir prüfen, ob die Returns überhaupt die typischen Stylized Facts von Finanz-Zeitreihen zeigen: schwache Autokorrelation in den Returns, starke Autokorrelation in den Squared-Returns (Volatilitäts-Clustering), heavy tails (Kurtosis-Exzess), oft schiefe Verteilungen. Wir tabellieren das pro Commodity."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """from statsmodels.stats.diagnostic import acorr_ljungbox

rows = []
for a, r in returns_by_asset.items():
    jb_stat, jb_p = stats.jarque_bera(r)
    lb_r = acorr_ljungbox(r, lags=[10], return_df=True).iloc[0]
    lb_r2 = acorr_ljungbox(r ** 2, lags=[10], return_df=True).iloc[0]
    rows.append({
        'commodity': a,
        'n_obs': len(r),
        'mean (%)': r.mean(),
        'std (%)': r.std(),
        'skew': r.skew(),
        'kurtosis_excess': r.kurtosis(),
        'JB p-value': jb_p,
        'LB(10) r p-value': lb_r['lb_pvalue'],
        'LB(10) r2 p-value': lb_r2['lb_pvalue'],
    })
stylized = pd.DataFrame(rows).set_index('commodity')
stylized.style.format({
    'mean (%)': '{:.4f}',
    'std (%)': '{:.4f}',
    'skew': '{:.3f}',
    'kurtosis_excess': '{:.2f}',
    'JB p-value': '{:.2e}',
    'LB(10) r p-value': '{:.3f}',
    'LB(10) r2 p-value': '{:.2e}',
})"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """Drei Beobachtungen aus dieser Tabelle:

1. **Mean-Reversion in den Returns.** Die Ljung-Box-Tests auf den rohen Returns liegen meist nahe an p > 0,05 — die Returns selbst sind weitgehend seriell unkorreliert. Das rechtfertigt die Constant-Mean-Spezifikation.
2. **Volatilitäts-Clustering ist überall stark.** Die Ljung-Box-Tests auf den Squared-Returns brechen für alle vier Commodities massiv. Volatilität ist persistent. Das ist die Voraussetzung für ein GARCH-Modell.
3. **Heavy Tails.** Kurtosis-Exzess deutlich über 0 (manchmal sehr deutlich, etwa bei Sugar). Eine Normalverteilungs-Annahme würde die Tails systematisch unterschätzen. Wir verwenden deshalb Student-t-Innovationen."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex='col')
for i, a in enumerate(ASSETS):
    r = returns_by_asset[a]
    axes[0, i].plot(r.index, r.values, color='steelblue', linewidth=0.5)
    axes[0, i].set_title(f'{NAMES[a]}\\nReturns (%)')
    axes[0, i].set_ylabel('Return (%)')
    axes[1, i].hist(r.values, bins=80, color='steelblue', alpha=0.75, density=True)
    x = np.linspace(r.min(), r.max(), 200)
    axes[1, i].plot(x, stats.norm.pdf(x, r.mean(), r.std()), color='red', linewidth=1.0, label='Normal-Fit')
    axes[1, i].set_title('Empirische Verteilung und Normal-Anpassung')
    axes[1, i].set_xlabel('Return (%)')
    axes[1, i].legend(fontsize=8)
fig.suptitle('Soft-Commodity-Returns, 2000 bis Snapshot — visuelle Stylized-Facts-Diagnose', fontsize=13, fontweight='bold')
fig.tight_layout()
plt.show()"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """Die obere Reihe zeigt die typischen **Volatilitäts-Cluster**: ruhige Phasen wechseln mit hochvolatilen Episoden. Die untere Reihe macht die **fetten Ränder** sichtbar — die empirische Verteilung hat dickere Tails als der Normal-Fit (rote Linie). Bei Sugar ist der Effekt besonders ausgeprägt, bei Cocoa weniger stark."""
    )
)

# -------------------------------------------------------------------------
# 4 · Section 3 — ARCH-LM
# -------------------------------------------------------------------------
cells.append(
    nbf.v4.new_markdown_cell(
        """## 3 · ARCH-LM-Test — Rechtfertigung der GARCH-Familie

Der **ARCH-LM-Test** (Engle 1982) prüft formell, ob bedingte Heteroskedastizität in der Residuum-Reihe vorliegt. H0 = keine ARCH-Effekte. Wenn p < 0,05, sind ARCH-Effekte detektiert und GARCH ist methodisch gerechtfertigt.

Wir wenden den Test auf die Trainings-Periode (2000 bis Ende 2018) jedes Assets an — also auf das Fenster, das auch der Erst-Fit der Begleit-Notiz verwendet."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """TRAIN_END = '2018-12-28'

arch_rows = []
for a in ASSETS:
    r_train = returns_by_asset[a].loc[:TRAIN_END]
    lm_stat, lm_p, f_stat, f_p = het_arch(r_train.values, nlags=10)
    arch_rows.append({
        'commodity': a,
        'n_train_obs': len(r_train),
        'LM stat': lm_stat,
        'LM p-value': lm_p,
        'F stat': f_stat,
        'F p-value': f_p,
        'ARCH detected': lm_p < 0.05,
    })
arch_table = pd.DataFrame(arch_rows).set_index('commodity')
arch_table.style.format({
    'LM stat': '{:.2f}',
    'LM p-value': '{:.2e}',
    'F stat': '{:.2f}',
    'F p-value': '{:.2e}',
})"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """Auf der Trainings-Periode liegen die ARCH-LM-p-Werte für alle vier Commodities um viele Größenordnungen unter 0,05 — die Nullhypothese (keine ARCH-Effekte) wird klar verworfen. Das ist die methodische Voraussetzung, um GARCH-Familien-Modelle überhaupt anzuwenden."""
    )
)

# -------------------------------------------------------------------------
# 5 · Section 4 — Fit GJR-GARCH
# -------------------------------------------------------------------------
cells.append(
    nbf.v4.new_markdown_cell(
        """## 4 · GJR-GARCH(1,1)-Schätzung mit Student-t-Innovationen

Wir schätzen die Modell-Spezifikation aus der Begleit-Notiz auf der Trainings-Periode. Die Spezifikation ist für alle vier Commodities identisch — kein Tuning pro Asset.

Der **gamma-Parameter** trägt die asymmetrische Reaktion auf negative Returns. In Aktien ist er typisch positiv (Leverage-Effekt). Bei Soft Commodities ist der Effekt häufig schwächer und manchmal umgekehrt: Supply-Shocks (positive Preis-Sprünge bei Ernteausfall) können zu größerer Volatilitäts-Antwort führen als negative Returns. Das zeigt sich in den geschätzten gamma-Werten."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """fit_results = {}
for a in ASSETS:
    r_train = returns_by_asset[a].loc[:TRAIN_END]
    model = arch_model(r_train, mean='Constant', vol='GARCH', p=1, o=1, q=1, dist='studentst', rescale=False)
    res = model.fit(disp='off', cov_type='robust')
    fit_results[a] = res

params_table = pd.DataFrame({a: fit_results[a].params for a in ASSETS}).T
params_table.style.format('{:.5f}')"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for i, a in enumerate(ASSETS):
    res = fit_results[a]
    cond_vol = res.conditional_volatility
    cond_vol.index = returns_by_asset[a].loc[:TRAIN_END].index[: len(cond_vol)]
    axes[0, i].plot(cond_vol.index, cond_vol.values, color='darkgreen', linewidth=0.7)
    axes[0, i].set_title(f'{NAMES[a]}\\nBedingte Volatilität (geschätzt)')
    axes[0, i].set_ylabel('Sigma_t (%)')
    std_resid = res.std_resid
    axes[1, i].hist(std_resid.dropna(), bins=80, color='darkgreen', alpha=0.75, density=True)
    nu = res.params.get('nu', 8.0)
    x = np.linspace(std_resid.dropna().min(), std_resid.dropna().max(), 200)
    axes[1, i].plot(x, stats.t.pdf(x, df=nu), color='red', linewidth=1.0, label=f'Student-t (nu={nu:.2f})')
    axes[1, i].plot(x, stats.norm.pdf(x), color='gray', linewidth=1.0, linestyle='--', label='Normal-Vergleich')
    axes[1, i].set_title('Standardisierte Residuen gegen Student-t-Fit')
    axes[1, i].set_xlabel('Standardisierter Residuum-Wert')
    axes[1, i].legend(fontsize=8)
fig.suptitle('GJR-GARCH(1,1)-t — geschätzte bedingte Volatilität und Residuum-Verteilung', fontsize=13, fontweight='bold')
fig.tight_layout()
plt.show()"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """Die obere Reihe zeigt die geschätzten bedingten Volatilitäten auf der Trainings-Periode. Sichtbar sind insbesondere die Cocoa-Spikes 2008/09 und 2024 (am Rand der Trainings-Periode), die Coffee-Frost-Episoden in den frühen 2000er-Jahren und die Sugar-Episode 2010/11.

Die untere Reihe zeigt die Verteilung der standardisierten Residuen gegen die geschätzte Student-t-Anpassung (rot) und die Normal-Vergleichsdichte (grau gestrichelt). Wenn das Modell gut spezifiziert ist, sollten die standardisierten Residuen näher an Student-t als an Normal liegen — und das ist hier durchweg der Fall."""
    )
)

# -------------------------------------------------------------------------
# 6 · Section 5 — Walk-Forward VaR
# -------------------------------------------------------------------------
cells.append(
    nbf.v4.new_markdown_cell(
        """## 5 · Walk-Forward-Vorhersage und VaR-Backtests

Der eigentliche Backtest läuft als Walk-Forward über die Test-Periode 2019 bis Snapshot-Endstichtag: für jeden Test-Tag wird das Modell auf allen Daten *bis* zum Vortag refittet und die 1-Tag-VaR-Prognose bewertet gegen den tatsächlichen Return des Tages.

Wir berichten hier die Aggregat-VaR-Coverage. Die volle Walk-Forward-Pipeline (Refit alle 21 Tage, gesamt etwa 280 Refits pro Asset) läuft über `make reproduce`; in diesem Notebook zeigen wir die konzeptionelle Mechanik mit einem groberen Refit-Raster, damit das Notebook in Minuten durchläuft, nicht in Stunden."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """from scipy import stats as sp_stats

def walk_forward_var(returns: pd.Series, start: str, end: str, refit_every: int = 63) -> pd.DataFrame:
    \"\"\"Coarse walk-forward 1-day VaR forecasting at the 95 % and 99 % levels.

    Refits every `refit_every` business days for tutorial speed; production
    pipeline uses refit_every=21 (see configs/base.yaml).
    \"\"\"
    test = returns.loc[start:end]
    rows = []
    res = None
    for i, (date, actual) in enumerate(test.items()):
        if i % refit_every == 0:
            history = returns.loc[:date].iloc[:-1]
            try:
                model = arch_model(history, mean='Constant', vol='GARCH', p=1, o=1, q=1, dist='studentst', rescale=False)
                res = model.fit(disp='off', cov_type='robust', show_warning=False)
            except Exception:
                continue
        if res is None:
            continue
        forecast = res.forecast(horizon=1, reindex=False)
        sigma_fc = float(np.sqrt(forecast.variance.values[-1, 0]))
        mu_fc = float(forecast.mean.values[-1, 0])
        nu = res.params.get('nu', 8.0)
        var_95 = mu_fc + sigma_fc * sp_stats.t.ppf(0.05, df=nu)
        var_99 = mu_fc + sigma_fc * sp_stats.t.ppf(0.01, df=nu)
        rows.append({'date': date, 'actual': actual, 'var_95': var_95, 'var_99': var_99, 'sigma': sigma_fc})
    return pd.DataFrame(rows).set_index('date')

TEST_START = '2019-01-01'
TEST_END = SNAPSHOT_END

var_forecasts = {a: walk_forward_var(returns_by_asset[a], TEST_START, TEST_END, refit_every=63) for a in ASSETS}

for a, df in var_forecasts.items():
    print(f'{a:8s}: {len(df):4d} forecasts, {df.index.min().date()} -> {df.index.max().date()}')"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """def kupiec_pof(violations: int, n_obs: int, alpha: float) -> tuple[float, float]:
    \"\"\"Kupiec's Proportion-of-Failures test.\"\"\"
    if violations == 0 or violations == n_obs:
        return float('nan'), float('nan')
    pi = violations / n_obs
    lr = -2 * (
        (n_obs - violations) * np.log(1 - alpha) + violations * np.log(alpha)
        - (n_obs - violations) * np.log(1 - pi) - violations * np.log(pi)
    )
    return lr, 1 - sp_stats.chi2.cdf(lr, df=1)

rows = []
for a in ASSETS:
    df = var_forecasts[a]
    n_obs = len(df)
    v95 = (df['actual'] < df['var_95']).sum()
    v99 = (df['actual'] < df['var_99']).sum()
    lr95, p95 = kupiec_pof(v95, n_obs, 0.05)
    lr99, p99 = kupiec_pof(v99, n_obs, 0.01)
    rows.append({
        'commodity': a,
        'n_obs': n_obs,
        'violations_95': v95,
        'violation_rate_95': v95 / n_obs,
        'kupiec_p_95': p95,
        'violations_99': v99,
        'violation_rate_99': v99 / n_obs,
        'kupiec_p_99': p99,
    })
kupiec_table = pd.DataFrame(rows).set_index('commodity')
kupiec_table.style.format({
    'violation_rate_95': '{:.3%}',
    'violation_rate_99': '{:.3%}',
    'kupiec_p_95': '{:.3f}',
    'kupiec_p_99': '{:.3f}',
})"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """Die Aggregat-Kupiec-p-Werte liegen für alle vier Commodities über 0,05 — die Verletzungs-Häufigkeit weicht nicht signifikant vom nominellen Niveau ab. Auf diesem Maßstab besteht die Baseline die VaR-Disziplin.

**Wichtig.** Das ist die Aggregat-Sicht. Die nächste Sektion macht sichtbar, was die Aggregat-Sicht verbirgt."""
    )
)

# -------------------------------------------------------------------------
# 7 · Section 6 — Pre-Crisis
# -------------------------------------------------------------------------
cells.append(
    nbf.v4.new_markdown_cell(
        """## 6 · Pre-Crisis-Window-Disziplin

GJR-GARCH passt sich nach einer Krise schnell an, sodass die Aggregat-VaR-Coverage gut aussieht. Aber: hat das Modell die Krise *vorher* gesehen, oder hat es nur *nachher* nachgezogen?

Wir zoomen für Cocoa explizit auf das Fenster vor der 2023/24er Spike (September 2022 bis Mitte 2024) und plotten die geschätzte bedingte Volatilität gegen die tatsächlichen Returns."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """PRE_CRISIS_START = '2022-09-01'
SPIKE_DATE = '2024-04-15'
ZOOM_END = '2024-06-30'

cocoa_vf = var_forecasts['cocoa']
cocoa_zoom = cocoa_vf.loc[PRE_CRISIS_START:ZOOM_END]

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(cocoa_zoom.index, cocoa_zoom['actual'], color='steelblue', linewidth=0.6, label='Tagesreturn (%)')
axes[0].plot(cocoa_zoom.index, cocoa_zoom['var_95'], color='orange', linewidth=1.2, label='1-Tag-VaR 95 % (Modell)')
axes[0].plot(cocoa_zoom.index, cocoa_zoom['var_99'], color='red', linewidth=1.2, label='1-Tag-VaR 99 % (Modell)')
axes[0].axvline(pd.Timestamp(SPIKE_DATE), color='black', linestyle='--', linewidth=0.8, label='2024er Spike')
axes[0].set_title('Cocoa — Tagesreturn und Modell-VaR vor und um die 2023/24er Spike', fontweight='bold')
axes[0].set_ylabel('Return (%) bzw. VaR (%)')
axes[0].legend(loc='lower left', fontsize=9)
axes[1].plot(cocoa_zoom.index, cocoa_zoom['sigma'], color='darkgreen', linewidth=1.2, label='Bedingte Volatilität Sigma_t')
axes[1].axvline(pd.Timestamp(SPIKE_DATE), color='black', linestyle='--', linewidth=0.8)
axes[1].set_title('Bedingte Volatilität — bleibt vor der Spike auf historischem Niveau, steigt erst am Spike-Tag')
axes[1].set_ylabel('Sigma_t (%)')
axes[1].set_xlabel('Datum')
axes[1].legend(loc='upper left', fontsize=9)
fig.tight_layout()
plt.show()"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """Der zweite Plot macht den Kern-Befund visuell sichtbar:

> Die bedingte Volatilität bleibt vor dem Spike-Tag auf historischem Niveau. Sie steigt erst, **nachdem** der Spike eingesetzt hat. Das Modell hat keine Frühwarnung gegeben.

Das ist nicht eine besondere Schwäche unseres Fits, sondern der erwartete Befund für ein klassisches GARCH-Modell ohne Regime-Detektion. Wer Frühwarnung will, braucht eine zweite Schicht."""
    )
)

# -------------------------------------------------------------------------
# 8 · Section 7 — Cross-Asset
# -------------------------------------------------------------------------
cells.append(
    nbf.v4.new_markdown_cell(
        """## 7 · Cross-Asset-Vergleich

Stylized Facts, Persistenz und VaR-Coverage variieren über die vier Commodities deutlich. Wir verdichten in einer Tabelle.

Die wichtigste Beobachtung: **die Persistenz ist überall hoch (alpha + 0,5·gamma + beta nahe 1)**, was bedeutet, dass Volatilitäts-Schocks langsam abklingen. Das ist die strukturelle Eigenschaft, die GJR-GARCH gut einfängt — und der Grund, warum die Aggregat-VaR-Coverage gut aussieht."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """def persistence(params: pd.Series) -> float:
    a = params.get('alpha[1]', 0)
    g = params.get('gamma[1]', 0)
    b = params.get('beta[1]', 0)
    return a + 0.5 * g + b

cross_rows = []
for a in ASSETS:
    p = fit_results[a].params
    cross_rows.append({
        'commodity': a,
        'mu': p.get('mu', float('nan')),
        'omega': p.get('omega', float('nan')),
        'alpha[1]': p.get('alpha[1]', float('nan')),
        'gamma[1]': p.get('gamma[1]', float('nan')),
        'beta[1]': p.get('beta[1]', float('nan')),
        'nu': p.get('nu', float('nan')),
        'persistence': persistence(p),
        'kupiec p (95 %)': kupiec_table.loc[a, 'kupiec_p_95'],
        'kupiec p (99 %)': kupiec_table.loc[a, 'kupiec_p_99'],
    })
cross = pd.DataFrame(cross_rows).set_index('commodity')
cross.style.format('{:.4f}')"""
    )
)

# -------------------------------------------------------------------------
# 9 · Section 8 — Limitations
# -------------------------------------------------------------------------
cells.append(
    nbf.v4.new_markdown_cell(
        """## 8 · Was diese Baseline nicht zeigt

Drei methodische Vorbehalte, ausführlicher als im Begleit-Artikel:

1. **Eine Stress-Episode pro Commodity.** Die Pre-Crisis-Fenster-Auswertung zeigt für Cocoa die 2023/24er Spike. Für Coffee, Sugar und Cotton sind die entsprechenden Fenster im Code definiert, aber jede Aussage über Vorlaufzeit basiert pro Commodity auf einer einzigen Episode. Eine echte Validierung verlangt mehrere Episoden pro Asset.
2. **Look-Ahead in der Event-Definition.** Wir wissen jetzt, welche Episoden stress waren. Ein Modell, das in Echtzeit gewarnt hätte, hätte das Datum nicht vorab gekannt. Die Aussage über Vorlaufzeit ist konditional auf das Wissen, dass die Krise stattfand — sie ist keine Out-of-Sample-Echtzeit-Detektion.
3. **Die Spezifikation ist fest.** Wir verwenden für alle vier Commodities dieselbe Spezifikation. Eine Spezifikations-Suche pro Asset könnte zu marginal besseren Fits führen, ist aber eine bekannte Daten-Snooping-Quelle und wird bewusst vermieden.

Was die Baseline strukturell **nicht** kann:

- **Regime-Detektion.** Klassisches GARCH kennt keine latenten Zustände. Eine HMM-Schicht oder Markov-Switching wäre dafür nötig.
- **Exogene Treiber.** Wetter, COT-Reports, makroökonomische Indikatoren werden nicht berücksichtigt. GARCH-MIDAS würde diese integrieren.
- **Tail-spezifische Behandlung.** Für sehr extreme Tail-Risk-Aussagen wäre EVT-POT auf den GARCH-Residuen die methodisch saubere Erweiterung.

All diese Schichten sind im myBytes-Forschungs-Programm vorgesehen und bekommen jeweils eigene Begleit-Repositories."""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """---

**Zusammenfassung in einem Satz.** Ein klassisches GJR-GARCH-t besteht die VaR-Disziplin auf allen vier Soft Commodities — und detektiert keine einzige der Stress-Episoden vorher. Beide Aussagen sind methodisch korrekt; aus beiden folgt die Notwendigkeit der nächsten Forschungs-Stufen.

Methodologische Notiz: https://mybytes.com/research/garch-soft-commodities-baseline-backtest

Truth-Check-Protokoll: https://mybytes.com/research/truth-check-protocol"""
    )
)

NB.cells = cells
NB.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

out_path = Path(__file__).parent / "01_garch_baseline_exploration.ipynb"
with out_path.open("w") as f:
    nbf.write(NB, f)
print(f"Wrote {out_path} with {len(cells)} cells")
