# soft-commodities-forecast-benchmark

**Begleit-Repository zur methodischen Notiz von myBytes Research zu
GJR-GARCH-Baseline-Backtests auf vier Soft Commodities.**

> Companion repository to the myBytes Research methodology note on
> GJR-GARCH baseline backtests across four soft commodities. See the
> English Quickstart below for non-German speakers.

---

## Worum es geht

Dieses Repository enthält den Code, die Konfigurationen und die
Reproduktions-Werkzeuge für einen mehrjährigen Walk-Forward-Backtest
eines klassischen GJR-GARCH(1,1)-Modells mit Student-t-Innovationen
auf vier ICE-Soft-Commodity-Continuous-Futures:

- **ICE Cocoa** (CC=F)
- **ICE Coffee** (KC=F, Arabica)
- **ICE Sugar** (SB=F, No. 11)
- **ICE Cotton** (CT=F, No. 2)

Methodisch ist das Modell der Branchen-Standard für asymmetrische
Volatilitäts-Modellierung. Wir bauen es bewusst als Baseline-Schicht
eines mehrjährig angelegten Forschungs-Programms zu Soft-Commodity-
Volatilität. Was diese Baseline leistet und was sie strukturell nicht
leistet, ist Gegenstand der dazugehörigen
[methodischen Notiz](https://mybytes.com/research/garch-soft-commodities-baseline-backtest).

## Was hier reproduzierbar ist

- Alle in der Notiz zitierten Modell-Parameter pro Commodity
  (Mean, Omega, Alpha, Gamma, Beta, Nu) mit identischer
  Walk-Forward-Split-Mechanik
- Alle VaR-Backtests (Kupiec-POF, Christoffersen-CC) auf den
  Aggregat-Perioden 2019–2024
- Pre-Crisis-Fenster-VaR-Coverage als zusätzliche Disziplin
  (Aggregat-Coverage misst Backward-Looking-Anpassung; das
  Pre-Crisis-Fenster misst, was vor der Krise messbar war)
- Die für den Artikel hervorgehobene Vorlaufzeit-Aussage
  („Vorlaufzeit gleich null für GJR-GARCH allein vor der
  2023/24er Cocoa-Spike") als reproduzierbarer Output

Reproduktion mit einem Befehl:

```bash
make reproduce
```

Der Befehl holt die Daten frisch via `yfinance` an einem in
`data_snapshot.json` eingefrorenen Endstichtag, läuft die volle
Pipeline (Training, Walk-Forward-Vorhersage, Evaluation) und
vergleicht die frisch geschätzten Parameter und Log-Likelihoods
gegen die in `results/<asset>_diagnostics.json` hinterlegten
Werte. Die Toleranz für kleine Yahoo-Daten-Driften ist im
Snapshot dokumentiert.

## Was hier nicht enthalten ist

Drei Dinge ausdrücklich:

1. **Keine Daten.** Yahoo Finance verbietet Redistribution gescraper
   Daten. Sie holen die Daten selbst über `yfinance`. Der Code in
   diesem Repository ist auf einen festen Snapshot-Endstichtag gepinnt,
   damit die Reproduktion deterministisch bleibt. Lizenz-Details:
   [`LICENSES.md`](LICENSES.md).
2. **Keine Folgestufen.** HMM-Regime-Detektion, GARCH-MIDAS mit
   Wetter- und COT-Daten, Foundation-Modelle für Volatilitäts-
   Targets — alles im myBytes-Forschungs-Programm vorgesehen, alles
   nicht in diesem Baseline-Repository. Wenn diese Schichten gebaut
   sind, kommen sie in eigene Begleit-Repositories.
3. **Keine Anlage- oder Hedging-Empfehlung.** Der Backtest ist
   methodisches Material, kein Handels-System. Siehe Disclaimer am
   Ende.

## Schnellstart in 10 Minuten

Voraussetzung: Python 3.11 oder 3.12, ein frisches virtuelles
Umfeld.

```bash
# 1. Repository klonen und Abhängigkeiten installieren
git clone https://github.com/myBytesResearch/soft-commodities-forecast-benchmark.git
cd soft-commodities-forecast-benchmark
make install

# 2. Umgebungs-Datei aus dem Beispiel ableiten
cp .env.example .env
# (Default-Werte funktionieren für die meisten Setups; siehe Kommentare in .env.example)

# 3. Eine einzelne Commodity reproduzieren
make reproduce-cocoa

# 4. Alle vier Commodities reproduzieren
make reproduce
```

Wenn am Ende der Reproduktion „reproduction match for cocoa" und
analog für die anderen drei Commodities erscheint, ist die
Reproduktion gelungen. Bei Drift-Meldungen prüfen Sie zuerst, ob
Ihr Snapshot-Endstichtag mit `data_snapshot.json` übereinstimmt.

## Struktur

```
src/benchmark/        — Modell-Code (train, predict, evaluate, reproduce)
configs/              — Pro Commodity ein Konfigurations-File, plus base.yaml
results/              — Diagnostics-JSON pro Commodity (Goldstandard-Werte)
notebooks/            — Forschungs-Notebooks mit ausführlicher Methodik
docs/                 — Methodologie und Limitations als eigenständige Dokumente
tests/                — Unit-Tests und Integrations-Tests
artifacts/            — Output frischer Läufe (gitignored)
data_snapshot.json    — Reproduzierbarkeits-Pin (Tickers, Endstichtag, Toleranz)
```

## Forschungs-Notebooks

Im Verzeichnis `notebooks/` liegen ausführliche Forschungs-Notebooks,
die den methodischen Weg von der Daten-Exploration bis zur Bewertung
der VaR-Disziplin sichtbar führen. Sie sind bewusst kein knappes
Tutorial: sie zeigen Stylized Facts, Diagnostics, Modell-Fit,
Walk-Forward-Backtest, VaR-Tests, Pre-Crisis-Fenster und die
methodische Selbstkritik nebeneinander.

## Zur methodischen Notiz

Die zugehörige Notiz auf mybytes.com erklärt:

- warum klassisches GARCH die VaR-Disziplin besteht und gleichzeitig
  null Vorlaufzeit vor der Cocoa-Spike 2023/24 produziert
- warum das kein Bug ist, sondern der erwartete Befund
- welche zweite Modell-Schicht für Frühwarnung nötig ist und in
  welchem Zeitrahmen wir sie veröffentlichen

→ [Das Single-GARCH-Limit auf Soft Commodities](https://mybytes.com/research/garch-soft-commodities-baseline-backtest)

## Disclaimer

Diese Implementation und die zitierten Backtest-Zahlen beschreiben
einen Walk-Forward-Backtest aus unserer eigenen Forschungs-Praxis.
Sie sind keine Anlage- und keine Hedging-Empfehlung. Die genannten
Performance-Aussagen beziehen sich auf eine spezifische Test-Setup-
Konfiguration und sind nicht ohne weiteres auf andere Anwendungs-
Szenarien übertragbar.

---

# English Quickstart

This repository reproduces the GJR-GARCH(1,1)-with-Student-t-innovations
baseline benchmark across four ICE Soft Commodity Continuous Futures
(Cocoa, Coffee, Sugar, Cotton). It is the companion to the myBytes
Research methodology note linked below.

```bash
git clone https://github.com/myBytesResearch/soft-commodities-forecast-benchmark.git
cd soft-commodities-forecast-benchmark
make install
cp .env.example .env
make reproduce
```

The `make reproduce` target fetches the data fresh via `yfinance` at
a snapshot end-date pinned in `data_snapshot.json`, runs the full
pipeline for all four commodities, and asserts the fresh parameters
and log-likelihoods against the stored diagnostics within a documented
relative tolerance.

**This repository ships code only.** Yahoo Finance Terms of Service
forbid data redistribution; you fetch the data yourself. See
`LICENSES.md` for the full license trifecta (code MIT, data sources,
third-party libraries).

For non-English methodology: the article and the methodology
documentation in `docs/` are in German. The code, configuration, and
operational commands are in English.

→ [Methodology note](https://mybytes.com/research/garch-soft-commodities-baseline-backtest)
→ [Truth-Check protocol](https://mybytes.com/research/truth-check-protocol)
