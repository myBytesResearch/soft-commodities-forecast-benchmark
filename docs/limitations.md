# Limitations

Was dieser Backtest **nicht** leistet. Eine eigenständige Dokumentation
der bekannten Schwächen der Implementation, getrennt von der
methodischen Beschreibung. Wer mit unseren Ergebnissen weiterarbeitet,
sollte zuerst hier lesen.

---

## 1 · Einzelne Stress-Episode pro Commodity

Die Pre-Crisis-Fenster-Auswertung basiert pro Commodity auf einer
einzigen Stress-Episode (2023/24 Cocoa-Spike, 2024 Coffee-Brazil-
Drought, 2023 Sugar-India-Export-Curb, 2022 Cotton-Supply-Shock).
**Eine Episode ist keine Stichprobe.** Die Aussage *„Vorlaufzeit
gleich null"* bezieht sich auf diese eine Episode pro Commodity.
Eine Validierung auf weiteren Stress-Episoden steht aus und ist
Gegenstand der nächsten Forschungs-Stufe.

## 2 · Look-Ahead-Bias in der Event-Definition

Die Event-Fenster sind manuell aus der Marktgeschichte ausgewählt.
Wir wissen *jetzt*, dass die Cocoa-Spike 2023/24 stattgefunden hat;
ein Modell, das sie *damals* hätte detektieren sollen, hätte das
Datum nicht gekannt. Diese Asymmetrie ist im Verfahren bewusst — die
Aussage über Vorlaufzeit ist ein **konditional-auf-Event**-Befund,
keine Out-of-Sample-Detektion in Echtzeit.

## 3 · GJR-GARCH-Spezifikation ist fest

Wir verwenden für alle vier Commodities GJR(1,1) mit Student-t.
Eine Spezifikations-Suche (p, o, q oder Verteilungs-Variante)
könnte für einzelne Commodities zu marginal besseren Fits führen.
Wir haben darauf bewusst verzichtet, weil Spezifikations-Suchen
über Walk-Forward-Backtests eine bekannte Quelle von Daten-
Snooping sind. Wer mit anderen Spezifikationen experimentiert,
sollte das in einem separaten Repository mit eigenen Snapshot-
Pins tun.

## 4 · Yahoo-Finance-Datenqualität

Yahoo-Daten für Continuous Futures sind keine Tier-1-Marktdaten.
Sie sind im Vergleich zu ICE Data Services oder Bloomberg-Feeds
weniger sauber: gelegentliche rückwirkende Korrekturen, fehlende
Halbtags-Handelstage, Zeitzonen-Inkonsistenzen bei Asien-Sessions.
Für **methodische Demonstration** ist die Qualität ausreichend. Für
**produktive Risk-Anwendungen** sollte ein lizenzierter Vendor-Feed
verwendet werden (siehe `LICENSES.md`).

## 5 · Continuous-Futures-Roll-Mechanik

Yahoo's Continuous-Futures-Zeitreihen verwenden eine eigene Roll-
Methode, die nicht öffentlich dokumentiert ist. Roll-Anpassungen
können kurzfristige Volatilitäts-Sprünge erzeugen, die nicht
Markt- sondern Roll-Artefakte sind. Wir filtern diese nicht
explizit. Für sehr feine Tail-Risk-Analysen ist eine
selbst-implementierte Roll-Methode mit dokumentierten Adjustment-
Regeln vorzuziehen.

## 6 · Backend-Abhängigkeit der Optimierung

Die GARCH-MLE-Optimierung im `arch`-Package nutzt SciPy-Optimierer
(Default: SLSQP). Je nach BLAS-Backend (OpenBLAS, MKL, Accelerate)
können die geschätzten Parameter in der achten Nachkommastelle
abweichen. Wir setzen Toleranzen in `data_snapshot.json` so, dass
diese Drift toleriert wird. Wer exakt-bit-reproduzierbar sein muss,
sollte eine reproduzierbare BLAS-Konfiguration verwenden.

## 7 · Was diese Notiz nicht abdeckt

- Vergleich gegen alternative Volatilitäts-Modelle (EGARCH, FIGARCH,
  Realized-GARCH, MS-GARCH, HMM, GARCH-MIDAS) — Gegenstand der
  nächsten Forschungs-Stufen
- Kreuzkorrelationen zwischen den vier Commodities — eigenes
  Forschungs-Thema
- Optionsmarkt-Validierung gegen implizite Volatilität — siehe
  künftige Notiz zur Realized-vs-Implied-Vola für den Einkauf
- Anwendung auf Procurement-Hedging — methodisches Material,
  keine operative Hedging-Empfehlung

---

Diese Liste ist nicht abschließend. Wer eine weitere Limitation
findet, ist eingeladen, ein Issue im Repository zu öffnen.
