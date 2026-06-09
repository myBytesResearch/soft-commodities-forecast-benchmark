# Methodologie

Vollständige Beschreibung der Modell-Spezifikation, der Daten-
Aufbereitung, der Walk-Forward-Mechanik und der Evaluations-
Disziplin. Diese Notiz ist die Begleit-Dokumentation zum
Repository-Code; die operative Zusammenfassung steht im
[README](../README.md).

---

## 1 · Modell-Spezifikation

### 1.1 GJR-GARCH(1,1) mit Student-t-Innovationen

Die Mean- und Volatility-Equation:

$$r_t = \mu + \varepsilon_t$$

$$\sigma_t^2 = \omega + \left(\alpha + \gamma \cdot \mathbb{1}[\varepsilon_{t-1} < 0]\right) \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

$$\varepsilon_t / \sigma_t \sim t_\nu$$

Wir verwenden die Standard-Implementation aus dem
[`arch`-Package](https://arch.readthedocs.io/) (Sheppard 2024).
Returns werden zur numerischen Stabilität in Prozent skaliert
(×100), wie es die `arch`-Konvention vorsieht.

Der `gamma`-Parameter trägt die asymmetrische Volatilitäts-
Reaktion auf negative Returns. In Aktienmärkten ist `gamma` typisch
positiv (Leverage-Effekt); bei Soft Commodities zeigen unsere
Schätzungen oft den umgekehrten Effekt (siehe
`results/<asset>_diagnostics.json` für die echten Werte pro
Commodity).

### 1.2 Pre-registrierte Spezifikation

Wir verwenden für alle vier Commodities **dieselbe** Spezifikation:
GJR-GARCH(1,1) mit Student-t-Innovationen, Constant-Mean,
robusten Bollerslev-Wooldridge-Standardfehlern. Es findet **keine
Spezifikations-Suche** statt — kein Hyperparameter-Tuning, keine
Asset-spezifischen p/o/q-Variationen. Diese Disziplin ist bewusst:
Spezifikations-Suchen über Walk-Forward-Backtests sind ein
bekanntes Quelle von Look-Ahead-Bias und Daten-Snooping.

### 1.3 Was die Spezifikation nicht ist

- Kein **Markov-Switching-GARCH** (gehört zur nächsten
  Forschungs-Stufe)
- Kein **EVT-POT auf Residuen** (eigene Stufe)
- Kein **GARCH-MIDAS** mit exogenen Niedrig-Frequenz-Faktoren
  (Wetter, COT, ENSO)
- Keine **Foundation-Modelle** für Volatilitäts-Targets

Diese Schichten sind im myBytes-Forschungs-Programm vorgesehen
und bekommen jeweils eigene Begleit-Repositories.

---

## 2 · Daten und Aufbereitung

### 2.1 Datenquelle

Tägliche Schluss-Preise für vier ICE Continuous Futures, abgerufen
über `yfinance` (Yahoo Finance):

- `CC=F` — ICE Cocoa
- `KC=F` — ICE Coffee (Arabica)
- `SB=F` — ICE Sugar No. 11
- `CT=F` — ICE Cotton No. 2

Lizenz-Vorbehalt: Yahoo Finance Terms of Service verbieten
Daten-Redistribution. Wir versenden Code, keine Daten. Details:
[`LICENSES.md`](../LICENSES.md).

### 2.2 Return-Berechnung

Logarithmische Returns, in Prozent skaliert:

$$r_t = 100 \cdot \log(P_t / P_{t-1})$$

Fehlende Tage (Wochenende, Feiertage) werden weggelassen, nicht
interpoliert.

### 2.3 Trainings- und Test-Periode

| Periode | Start | Ende |
|---|---|---|
| Training | 2000-01-01 | 2018-12-28 |
| Test     | 2019-01-01 | 2024-12-31 |

Walk-Forward über die Test-Periode mit einem expandierenden
Trainings-Fenster (Initial-Fenster ≈ 10 Jahre, Refit alle 21
Handelstage). Vorhersage-Horizont: 1 Tag voraus.

---

## 3 · Evaluation

### 3.1 VaR-Backtests

Pro Walk-Forward-Fenster bewerten wir die bedingte 1-Tag-VaR
gegen die tatsächlich beobachteten Returns:

- **Kupiec-POF** ([Kupiec 1995](https://doi.org/10.3905/jod.1995.407942))
  prüft die Verletzungs-Häufigkeit gegen das nominelle Niveau
- **Christoffersen-CC** ([Christoffersen 1998](https://doi.org/10.2307/2527341))
  prüft zusätzlich die Unabhängigkeit der Verletzungen
- Niveaus: 95 % und 99 %

Wir berichten Verletzungs-Anteil, Teststatistik, p-Wert und
Verworfen-Flag pro Test.

### 3.2 Pre-Crisis-Window-VaR-Coverage

**Eine eigene Disziplin neben der Aggregat-Coverage.** GARCH-
Modelle passen sich nach einer Krise schnell an, sodass die
Aggregat-VaR-Coverage gut aussehen kann, obwohl das Modell vor
der Krise *nichts* gesehen hat. Wir bewerten deshalb separat ein
**Pre-Crisis-Fenster** von etwa 126 Handelstagen vor jedem
bekannten Stress-Event:

| Commodity | Stress-Event | Fenster |
|---|---|---|
| Cocoa  | 2023/24 supply shock | 2022-09-01 → 2023-03-05 |
| Coffee | 2024 Brazil drought  | 2023-09-02 → 2024-03-05 |
| Sugar  | 2023 India export curb | 2023-04-01 → 2023-10-01 |
| Cotton | 2022 supply shock    | 2022-02-01 → 2022-08-01 |

Auf diesen Fenstern wendet sich derselbe Kupiec-POF- und
Christoffersen-CC-Test an. Die Ergebnisse liegen in
`results/evaluation_<asset>.json` unter
`pre_crisis_var_coverage`.

### 3.3 Vorlaufzeit-Aussage

Für die im Begleit-Artikel hervorgehobene Aussage *„Vorlaufzeit
gleich null"* messen wir, ob die bedingte Volatilitäts-Prognose
in den 30 Handelstagen vor dem Spike-Tag signifikant über ihren
60-Tage-Trend steigt. Bei GJR-GARCH allein ist die Antwort für
die Cocoa-2023/24-Spike: nein, die Volatilität steigt erst *am*
Spike-Tag. Das ist der erwartete Befund, weil GJR-GARCH keine
Regime-Detektion enthält.

### 3.4 R²_OOS gegen Squared-Returns

Wir berichten zusätzlich den Out-of-Sample-R² der bedingten
Varianz-Prognose gegen die nachträglich beobachteten Squared-
Returns. Werte zwischen −0,01 und +0,03 sind in der Volatilitäts-
Forecasting-Literatur erwartbar und kein Modell-Mangel, sondern
Folge der hohen Rausch-Komponente von Squared-Returns als
Volatilitäts-Proxy
([Andersen/Bollerslev 1998](https://www.jstor.org/stable/2527343)).

---

## 4 · Reproduzierbarkeit

### 4.1 Snapshot-Pinning

Reproduzierbarkeit verlangt einen festen Endstichtag der Yahoo-
Daten. Wir pinnen diesen in `data_snapshot.json` und prüfen die
frischen Fit-Parameter gegen `results/<asset>_diagnostics.json`
innerhalb einer dokumentierten Toleranz.

Toleranzen:
- Parameter-Drift: 1 % relativ
- Log-Likelihood-Drift: 0,5 % relativ

Die Toleranzen sind bewusst klein, aber nicht null. Yahoo
korrigiert gelegentlich historische Schlusskurse rückwirkend, und
arch-Optimierung kann je nach BLAS-Backend in der achten
Nachkommastelle abweichen. Größere Drift ist ein echtes Signal,
nicht Rauschen.

### 4.2 Seed-Disziplin

Alle stochastischen Komponenten (Numpy, Python `random`) sind auf
Seed 42 gesetzt. Die GARCH-Optimierung selbst ist deterministisch
für identische Daten und identische arch-Version.

### 4.3 MLflow-Tracking

Pro Walk-Forward-Fenster wird ein MLflow-Run erzeugt, mit
Konfigurations-Hash, Daten-Snapshot-Hash, allen Parametern, allen
Metriken und den Forecast-Parquets als Artefakt. Default-Backend
ist eine lokale SQLite-Datei (`mlflow.db`); Remote-Backends lassen
sich über `MLFLOW_TRACKING_URI` setzen (siehe `.env.example`).

---

## 5 · Truth-Check-Protokoll

Diese Methodologie ist nach dem myBytes-Truth-Check-Protokoll
dokumentiert (sieben Schritte: Claim-Extraktion, Klassifizierung,
Anker-Mapping, Reproduzierbarkeit, Steel-Man, Limitations,
unabhängiger Review). Der Status der sieben Schritte für die
Begleit-Notiz steht im Frontmatter des Artikels.

→ [Truth-Check-Protokoll](https://mybytes.com/research/truth-check-protocol)

---

## 6 · Reading List

1. [Glosten, Jagannathan, Runkle 1993, *On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks*](https://www.jstor.org/stable/2329067) — GJR-GARCH-Original
2. [Bollerslev 1986, *Generalized Autoregressive Conditional Heteroskedasticity*](https://doi.org/10.1016/0304-4076(86)90063-1) — GARCH-Original
3. [Andersen/Bollerslev 1998, *Answering the Skeptics: Yes, Standard Volatility Models Do Provide Accurate Forecasts*](https://www.jstor.org/stable/2527343)
4. [Kupiec 1995, *Techniques for Verifying the Accuracy of Risk Measurement Models*](https://doi.org/10.3905/jod.1995.407942)
5. [Christoffersen 1998, *Evaluating Interval Forecasts*](https://doi.org/10.2307/2527341)
6. [Sheppard 2024, *arch package documentation*](https://arch.readthedocs.io/)
