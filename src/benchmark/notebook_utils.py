"""
Notebook Utilities — Dynamische Erklärungs-Komponenten für Jupyter Notebooks.

Wird in JEDEM Notebook genutzt, um konsistente, aus Daten generierte
Insights, Objectives und Conclusions zu rendern. Kein hardcoded Markdown.

Usage:
    from benchmark.notebook_utils import (
        render_objective,
        render_insight,
        render_conclusion,
        render_methodology_note,
    )

    render_objective(
        notebook_title="Data Inventory",
        objective="Verifiziere yfinance-Datenlage für Kakao & Kaffee.",
        approach="Lade Returns, prüfe Stylized Facts, teste ARCH-Effekte.",
        expected_outcomes=["Datenqualitäts-Check", "ARCH-LM Ergebnis", "Split-Validierung"],
    )

    render_insight(
        title="ARCH-Effekte detektiert",
        findings={"Cocoa LM p-value": 0.0001, "Coffee LM p-value": 0.0003},
        severity="success",
        next_action="Welle-1-Modelle trainieren",
    )
"""

from __future__ import annotations

from typing import Any, Literal

from IPython.display import HTML, Markdown, display

Severity = Literal["success", "warning", "critical", "info"]


# ==============================================================================
# Color & Style Definitions (myBytes Brand Colors)
# ==============================================================================
_COLORS: dict[Severity, dict[str, str]] = {
    "success": {
        "border": "#8eb600",       # myBytes Green
        "bg": "#f4f8e8",
        "icon": "✓",
        "label": "Insight",
    },
    "warning": {
        "border": "#f0a800",
        "bg": "#fff8e8",
        "icon": "⚠",
        "label": "Warning",
    },
    "critical": {
        "border": "#d63031",
        "bg": "#fdf0f0",
        "icon": "✗",
        "label": "Critical",
    },
    "info": {
        "border": "#058ab9",       # myBytes Blue
        "bg": "#e8f4f9",
        "icon": "ℹ",
        "label": "Info",
    },
}

_NAVY = "#0a1e2e"   # myBytes Dark Navy


def _format_value(value: Any) -> str:
    """Format any value for display."""
    if isinstance(value, float):
        if abs(value) < 0.0001 or abs(value) > 1e6:
            return f"{value:.4g}"
        return f"{value:.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, bool):
        return "✓ True" if value else "✗ False"
    if value is None:
        return "<i>None</i>"
    return str(value)


# ==============================================================================
# Executive Objective (top of notebook)
# ==============================================================================
def render_objective(
    notebook_title: str,
    objective: str,
    approach: str,
    expected_outcomes: list[str],
    related_models: list[str] | None = None,
) -> None:
    """
    Render the Executive Objective block at the top of a notebook.

    Args:
        notebook_title: Notebook title for the H1 header.
        objective: 2-3 sentence description of WHY this notebook exists.
        approach: Methodology / data / models being used.
        expected_outcomes: Bullet list of concrete deliverables.
        related_models: Optional list of model folder names this NB feeds.
    """
    outcomes_html = "".join(f"<li>{o}</li>" for o in expected_outcomes)
    related_html = ""
    if related_models:
        related_html = (
            "<p style='margin: 8px 0 0 0;'><b>Related models:</b> "
            + ", ".join(f"<code>{m}</code>" for m in related_models)
            + "</p>"
        )

    html = f"""
    <div style="
        border-left: 4px solid {_NAVY};
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 16px 20px;
        margin: 16px 0;
        border-radius: 4px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    ">
        <h2 style="margin: 0 0 12px 0; color: {_NAVY};">
            🎯 {notebook_title}
        </h2>
        <div style="margin-bottom: 10px;">
            <b style="color: {_NAVY};">Objective</b><br>
            <span style="color: #333;">{objective}</span>
        </div>
        <div style="margin-bottom: 10px;">
            <b style="color: {_NAVY};">Approach</b><br>
            <span style="color: #333;">{approach}</span>
        </div>
        <div>
            <b style="color: {_NAVY};">Expected Outcomes</b>
            <ul style="margin: 4px 0 0 0; padding-left: 20px; color: #333;">
                {outcomes_html}
            </ul>
        </div>
        {related_html}
    </div>
    """
    display(HTML(html))


# ==============================================================================
# Section Objective (before each analytical section)
# ==============================================================================
def render_section(title: str, rationale: str) -> None:
    """
    Lightweight section header with rationale.

    Use before each analytical sub-section in a notebook.
    """
    html = f"""
    <div style="
        border-left: 3px solid #058ab9;
        padding: 8px 14px;
        margin: 24px 0 12px 0;
        background: #f7fbfd;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    ">
        <h3 style="margin: 0 0 4px 0; color: #058ab9;">{title}</h3>
        <span style="color: #555; font-size: 0.92em;">{rationale}</span>
    </div>
    """
    display(HTML(html))


# ==============================================================================
# Insight Panel (after analytical operations)
# ==============================================================================
def render_insight(
    title: str,
    findings: dict[str, Any],
    severity: Severity = "info",
    next_action: str | None = None,
    caveats: str | None = None,
) -> None:
    """
    Render a dynamic insight panel after an analytical operation.

    Args:
        title: Short headline.
        findings: Dict of {label: value}. Values are auto-formatted.
        severity: 'success' | 'warning' | 'critical' | 'info'.
        next_action: Optional recommended next step.
        caveats: Optional methodological caveats.
    """
    style = _COLORS[severity]

    findings_html = "".join(
        f"""<tr>
            <td style="padding: 4px 12px 4px 0; color: #555; vertical-align: top;">{k}</td>
            <td style="padding: 4px 0; font-family: 'SF Mono', Consolas, monospace; color: {_NAVY};">
                <b>{_format_value(v)}</b>
            </td>
        </tr>"""
        for k, v in findings.items()
    )

    next_action_html = ""
    if next_action:
        next_action_html = f"""
        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px dashed #ccc;">
            <span style="color: #555; font-size: 0.9em;"><b>Next:</b> {next_action}</span>
        </div>
        """

    caveats_html = ""
    if caveats:
        caveats_html = f"""
        <div style="margin-top: 8px; padding: 8px 12px; background: #fff; border-radius: 3px;
                    font-size: 0.85em; color: #666; font-style: italic;">
            <b>⚠ Caveats:</b> {caveats}
        </div>
        """

    html = f"""
    <div style="
        border-left: 4px solid {style['border']};
        background: {style['bg']};
        padding: 12px 16px;
        margin: 12px 0;
        border-radius: 4px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    ">
        <div style="font-weight: 600; color: {_NAVY}; margin-bottom: 8px; font-size: 1.05em;">
            <span style="color: {style['border']};">{style['icon']}</span>
            &nbsp;{style['label']}: {title}
        </div>
        <table style="border-collapse: collapse; margin: 0;">
            {findings_html}
        </table>
        {next_action_html}
        {caveats_html}
    </div>
    """
    display(HTML(html))


# ==============================================================================
# Methodology Note (inline reminders)
# ==============================================================================
def render_methodology_note(text: str) -> None:
    """Render a small inline methodology reminder."""
    html = f"""
    <div style="
        border-left: 3px solid #999;
        padding: 6px 12px;
        margin: 8px 0;
        background: #fafafa;
        font-size: 0.88em;
        color: #555;
        font-style: italic;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    ">
        <b>Method:</b> {text}
    </div>
    """
    display(HTML(html))


# ==============================================================================
# Executive Conclusion (end of notebook)
# ==============================================================================
def render_conclusion(
    key_findings: list[str],
    methodological_caveats: list[str],
    next_steps: list[str | dict[str, str]],
    overall_status: Severity = "success",
) -> None:
    """
    Render the Executive Conclusion at the end of a notebook.

    Args:
        key_findings: 2-4 bullet points summarizing actual results.
        methodological_caveats: What was NOT shown, assumptions, data gaps.
        next_steps: Concrete actions. Strings or dicts with {'text': ..., 'code': ...}.
        overall_status: Overall severity for the notebook outcome.
    """
    style = _COLORS[overall_status]

    findings_html = "".join(f"<li>{f}</li>" for f in key_findings)
    caveats_html = "".join(f"<li>{c}</li>" for c in methodological_caveats)

    next_steps_items = []
    for step in next_steps:
        if isinstance(step, dict):
            text = step.get("text", "")
            code = step.get("code", "")
            item = f"""<li>{text}
                <pre style="background: #1e1e1e; color: #d4d4d4; padding: 8px 12px;
                            margin: 6px 0 0 0; border-radius: 3px;
                            font-family: 'SF Mono', Consolas, monospace; font-size: 0.85em;
                            overflow-x: auto;">{code}</pre>
            </li>"""
        else:
            item = f"<li>{step}</li>"
        next_steps_items.append(item)
    next_steps_html = "".join(next_steps_items)

    html = f"""
    <div style="
        border: 2px solid {_NAVY};
        background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
        padding: 20px 24px;
        margin: 24px 0;
        border-radius: 6px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    ">
        <h2 style="margin: 0 0 16px 0; color: {_NAVY};
                   border-bottom: 2px solid {style['border']}; padding-bottom: 8px;">
            📈 Executive Conclusion
            <span style="float: right; font-size: 0.75em; color: {style['border']};">
                {style['icon']} {style['label']}
            </span>
        </h2>

        <h3 style="color: {_NAVY}; margin: 16px 0 8px 0;">Key Findings</h3>
        <ul style="margin: 0; padding-left: 20px; color: #333; line-height: 1.6;">
            {findings_html}
        </ul>

        <h3 style="color: {_NAVY}; margin: 20px 0 8px 0;">⚠ Methodological Caveats</h3>
        <ul style="margin: 0; padding-left: 20px; color: #555;
                   line-height: 1.6; font-size: 0.95em;">
            {caveats_html}
        </ul>

        <h3 style="color: {_NAVY}; margin: 20px 0 8px 0;">→ Recommended Next Steps</h3>
        <ol style="margin: 0; padding-left: 20px; color: #333; line-height: 1.6;">
            {next_steps_html}
        </ol>
    </div>
    """
    display(HTML(html))


# ==============================================================================
# Convenience: Tabular Insight (for comparing values across assets/models)
# ==============================================================================
def render_comparison_table(
    title: str,
    headers: list[str],
    rows: list[list[Any]],
    severity: Severity = "info",
    note: str | None = None,
) -> None:
    """
    Render a comparison table as an insight panel.

    Example:
        render_comparison_table(
            title="Stylized Facts: Cocoa vs. Coffee",
            headers=["Metric", "Cocoa", "Coffee"],
            rows=[
                ["Mean (%)", 0.012, 0.008],
                ["Std (%)", 1.82, 2.04],
                ["Excess Kurtosis", 4.5, 6.1],
            ],
        )
    """
    style = _COLORS[severity]

    headers_html = "".join(
        f"<th style='padding: 8px 12px; background: {_NAVY}; color: white; "
        f"text-align: left; font-weight: 600;'>{h}</th>"
        for h in headers
    )

    rows_html = ""
    for i, row in enumerate(rows):
        bg = "#fafafa" if i % 2 == 0 else "#ffffff"
        cells = "".join(
            f"<td style='padding: 6px 12px; border-bottom: 1px solid #eee; "
            f"font-family: \"SF Mono\", monospace;'>{_format_value(v)}</td>"
            for v in row
        )
        rows_html += f"<tr style='background: {bg};'>{cells}</tr>"

    note_html = ""
    if note:
        note_html = f"<div style='margin-top: 8px; color: #555; font-size: 0.88em;'><i>{note}</i></div>"

    html = f"""
    <div style="
        border-left: 4px solid {style['border']};
        background: white;
        padding: 12px 16px;
        margin: 12px 0;
        border-radius: 4px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    ">
        <div style="font-weight: 600; color: {_NAVY}; margin-bottom: 10px; font-size: 1.05em;">
            <span style="color: {style['border']};">{style['icon']}</span>
            &nbsp;{title}
        </div>
        <table style="border-collapse: collapse; width: 100%;">
            <thead><tr>{headers_html}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        {note_html}
    </div>
    """
    display(HTML(html))
