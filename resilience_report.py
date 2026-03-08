"""
resilience_report.py
====================

Generates the self-contained HTML resilience assessment report.

Follows the existing pattern from visualize.generate_html_report() but with
richer content: executive summary, MFD analysis, resilience breakdown,
weak point analysis, scenario results, and recommendations.

All images are referenced by relative path (figures/ subdirectory), so the
report is portable when the entire output directory is moved together.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from mfd_analysis import ResilienceScore
from scenario_generator import Scenario

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grade colour mapping
# ---------------------------------------------------------------------------

GRADE_COLORS = {
    "A+": "#27ae60",
    "A": "#2ecc71",
    "B": "#3498db",
    "C": "#f39c12",
    "D": "#e67e22",
    "F": "#e74c3c",
}


# ---------------------------------------------------------------------------
# Helper: embed image as base64 for self-contained report
# ---------------------------------------------------------------------------


def _img_tag(path: str, alt: str) -> str:
    """Return an <img> tag. Uses relative path if exists, else placeholder."""
    if os.path.exists(path):
        # Use relative path from the report location.
        relpath = os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)
        return f"<img src='{relpath}' alt='{alt}' loading='lazy'>"
    return f"<p class='missing-img'>[Image not available: {alt}]</p>"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_resilience_report(
    output_dir: str,
    resilience_score: ResilienceScore,
    scenarios: list[Scenario],
    all_results: list[dict],
    config: dict,
    figures: dict[str, str],
    *,
    claude_analysis: str | None = None,
    per_type_fits: dict | None = None,
) -> str:
    """
    Generate the complete HTML resilience assessment report.

    Args:
        output_dir:       Root assessment output directory.
        resilience_score: Computed ResilienceScore object.
        scenarios:        List of all Scenario objects.
        all_results:      List of result dicts from parallel execution.
        config:           Base configuration dict.
        figures:          Dict mapping figure name to file path.
        claude_analysis:  Optional AI-generated analysis text (markdown).
        per_type_fits:    Optional dict from fit_greenshields_per_scenario_type().
                          If None, loaded from mfd_per_type_fits.json if present.

    Returns:
        Path to the generated report.html file.
    """
    # Load per-type fits from disk if not supplied directly.
    if per_type_fits is None:
        _fits_path = os.path.join(output_dir, "mfd_per_type_fits.json")
        if os.path.exists(_fits_path):
            with open(_fits_path, encoding="utf-8") as _f:
                per_type_fits = json.load(_f)
    grade = resilience_score.grade
    grade_color = GRADE_COLORS.get(grade, "#999999")
    score = resilience_score.overall_score
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Count successes / failures.
    n_success = sum(1 for r in all_results if r.get("status") == "success")
    n_total = len(scenarios)

    # Demand levels and incident types.
    demand_levels = sorted(set(s.period for s in scenarios))
    incident_types = sorted(set(s.scenario_type for s in scenarios))

    html = []

    # ── Head ──
    html.append("<!DOCTYPE html>")
    html.append('<html lang="en">')
    html.append("<head>")
    html.append('  <meta charset="UTF-8">')
    html.append('  <meta name="viewport" content="width=device-width, initial-scale=1.0">')
    html.append("  <title>Resilience Assessment Report</title>")
    html.append("  <style>")
    html.append(_css())
    html.append("  </style>")
    html.append("</head>")
    html.append("<body>")
    html.append("<div class='container'>")

    # ── Header ──
    html.append("<h1>One-Click Resilience Assessment Report</h1>")
    html.append(f"<p class='subtitle'>Generated {timestamp}</p>")

    # ── Section 1: Executive Summary ──
    html.append("<h2>1. Executive Summary</h2>")
    html.append("<div class='executive-summary'>")
    html.append("  <div class='grade-section'>")
    html.append(f"    <div class='grade-badge' style='background: {grade_color};'>{grade}</div>")
    html.append(f"    <div class='score-value'>{score:.2f} / 1.00</div>")
    html.append(f"    <div class='score-interp'>{resilience_score.interpretation}</div>")
    html.append("  </div>")
    html.append("  <div class='summary-stats'>")
    html.append(
        f"    <div class='stat'><span class='stat-val'>{n_total}</span><span class='stat-label'>Scenarios</span></div>"
    )
    html.append(
        f"    <div class='stat'><span class='stat-val'>{n_success}</span><span class='stat-label'>Successful</span></div>"
    )
    html.append(
        f"    <div class='stat'><span class='stat-val'>{len(demand_levels)}</span><span class='stat-label'>Demand Levels</span></div>"
    )
    html.append(
        f"    <div class='stat'><span class='stat-val'>{len(incident_types)}</span><span class='stat-label'>Incident Types</span></div>"
    )
    html.append("  </div>")
    html.append("</div>")

    # Key findings.
    html.append("<div class='key-findings'>")
    html.append("  <h3>Key Findings</h3>")
    html.append("  <ul>")
    html.append(
        f"    <li>Speed resilience: <strong>{resilience_score.speed_resilience:.2f}</strong> — "
        f"{'minimal' if resilience_score.speed_resilience > 0.85 else 'some' if resilience_score.speed_resilience > 0.7 else 'significant'} "
        f"speed degradation under incidents</li>"
    )
    html.append(
        f"    <li>Throughput resilience: <strong>{resilience_score.throughput_resilience:.2f}</strong> — "
        f"{'flow well maintained' if resilience_score.throughput_resilience > 0.85 else 'moderate flow impact' if resilience_score.throughput_resilience > 0.7 else 'significant flow reduction'}</li>"
    )
    html.append(
        f"    <li>Recovery: <strong>{resilience_score.recovery_resilience:.2f}</strong> — "
        f"{'excellent' if resilience_score.recovery_resilience > 0.85 else 'good' if resilience_score.recovery_resilience > 0.6 else 'slow'} "
        f"post-incident recovery (AI = {resilience_score.ai_aggregate:.3f})</li>"
    )
    html.append(
        f"    <li>Robustness: <strong>{resilience_score.robustness:.2f}</strong> — "
        f"{'strong' if resilience_score.robustness > 0.85 else 'moderate' if resilience_score.robustness > 0.6 else 'weak'} "
        f"tolerance to increasing incident rates</li>"
    )
    if resilience_score.weak_points:
        top_wp = resilience_score.weak_points[0]
        html.append(
            f"    <li>Most vulnerable edge: <strong>{top_wp.edge_id}</strong> "
            f"({top_wp.accident_count} incidents, vulnerability index {top_wp.vulnerability_index:.3f})</li>"
        )
    html.append("  </ul>")
    html.append("</div>")

    # ── Section 2: Network Overview ──
    html.append("<h2>2. Network Overview</h2>")
    html.append("<div class='metadata'>")
    html.append(
        f"  <div class='metadata-row'><span class='metadata-label'>Demand levels tested:</span>"
        f"<span class='metadata-value'>{', '.join(f'p={p}' for p in demand_levels)}</span></div>"
    )
    html.append(
        f"  <div class='metadata-row'><span class='metadata-label'>Scenario types:</span>"
        f"<span class='metadata-value'>{', '.join(incident_types)}</span></div>"
    )
    html.append("</div>")

    # Scenario matrix table.
    html.append("<h3>Scenario Matrix</h3>")
    html.append("<table class='scenario-table'>")
    html.append("  <tr><th>Demand Level</th>")
    for it in incident_types:
        html.append(f"    <th>{it.replace('_', ' ').title()}</th>")
    html.append("  </tr>")
    for period in demand_levels:
        html.append(f"  <tr><td>p={period}</td>")
        for it in incident_types:
            matching = [s for s in scenarios if s.period == period and s.scenario_type == it]
            n = len(matching)
            html.append(f"    <td>{n} run{'s' if n != 1 else ''}</td>")
        html.append("  </tr>")
    html.append("</table>")

    # ── Section 3: Macroscopic Fundamental Diagram ──
    html.append("<h2>3. Macroscopic Fundamental Diagram</h2>")
    html.append(
        "<p class='section-note'>Density–flow and density–speed relationships across all "
        "demand levels and incident configurations. Each point is one 5-minute "
        "measurement window. Colours denote scenario type.</p>"
    )
    html.append("<div class='image-grid'>")
    for key in ["mfd_density_flow", "mfd_density_speed"]:
        if key in figures:
            html.append(_img_tag(figures[key], key.replace("_", " ").title()))
        else:
            html.append(f"<p class='missing-img'>[{key.replace('_', ' ').title()} not available]</p>")
    html.append("</div>")

    # Theoretical MFD figure.
    html.append(
        "<h3>3a. Theoretical Greenshields Curves per Incident Level</h3>"
        "<p class='section-note'>"
        "Greenshields model fitted separately for each scenario type. "
        "Solid lines = observed density range; dashed = theoretical extrapolation to jam density. "
        "Diamond markers indicate each curve's capacity point (k<sub>c</sub>, q<sub>max</sub>). "
        "Incident scenarios show reduced free-flow speed and capacity relative to baseline, "
        "quantifying the MFD degradation caused by each disturbance level."
        "</p>"
    )
    if "mfd_theoretical" in figures:
        html.append(_img_tag(figures["mfd_theoretical"], "Theoretical MFD per Incident Level"))
    else:
        html.append("<p class='missing-img'>[Theoretical MFD figure not available]</p>")

    # Per-type fit parameter table.
    if per_type_fits:
        _type_order = [
            "baseline", "low_incident", "default_incident", "high_incident", "extreme_incident"
        ]
        _type_labels = {
            "baseline": "Baseline",
            "low_incident": "Low Incident",
            "default_incident": "Default Incident",
            "high_incident": "High Incident",
            "extreme_incident": "Extreme Incident",
        }
        html.append(
            "<table style='margin-top:12px;width:100%;border-collapse:collapse;font-size:0.88em'>"
            "<thead><tr style='background:#2c3e50;color:white'>"
            "<th style='padding:7px 10px;text-align:left'>Scenario Type</th>"
            "<th style='padding:7px 10px;text-align:right'>u<sub>f</sub> (km/h)</th>"
            "<th style='padding:7px 10px;text-align:right'>k<sub>j</sub> (veh/km)</th>"
            "<th style='padding:7px 10px;text-align:right'>k<sub>c</sub> (veh/km)</th>"
            "<th style='padding:7px 10px;text-align:right'>q<sub>max</sub> (veh/h)</th>"
            "<th style='padding:7px 10px;text-align:right'>R²</th>"
            "<th style='padding:7px 10px;text-align:right'>N points</th>"
            "</tr></thead><tbody>"
        )
        for i, stype in enumerate(_type_order):
            if stype not in per_type_fits:
                continue
            f = per_type_fits[stype]
            bg = "#f8f9fa" if i % 2 == 0 else "white"
            label = _type_labels.get(stype, stype)
            r2_color = "#27ae60" if f["r_squared"] >= 0.7 else (
                "#f39c12" if f["r_squared"] >= 0.4 else "#e74c3c"
            )
            html.append(
                f"<tr style='background:{bg}'>"
                f"<td style='padding:6px 10px;font-weight:500'>{label}</td>"
                f"<td style='padding:6px 10px;text-align:right'>{f['free_flow_speed_kmh']:.2f}</td>"
                f"<td style='padding:6px 10px;text-align:right'>{f['jam_density_veh_per_km']:.2f}</td>"
                f"<td style='padding:6px 10px;text-align:right'>{f['critical_density_veh_per_km']:.2f}</td>"
                f"<td style='padding:6px 10px;text-align:right'>{f['capacity_veh_per_hour']:.1f}</td>"
                f"<td style='padding:6px 10px;text-align:right;color:{r2_color};font-weight:bold'>"
                f"{f['r_squared']:.3f}</td>"
                f"<td style='padding:6px 10px;text-align:right;color:#666'>{f['n_points']:,}</td>"
                "</tr>"
            )
        html.append("</tbody></table>")
        html.append(
            "<p style='font-size:0.8em;color:#666;margin-top:6px'>"
            "R² colour coding: "
            "<span style='color:#27ae60;font-weight:bold'>≥0.70 good</span> · "
            "<span style='color:#f39c12;font-weight:bold'>0.40–0.70 moderate</span> · "
            "<span style='color:#e74c3c;font-weight:bold'>&lt;0.40 poor fit</span>"
            "</p>"
        )

    # Overall Greenshields parameters (baseline-only fit).
    if "free_flow_speed_kmh" in (mfd_p := resilience_score.mfd_parameters):
        html.append("<div class='metadata'>")
        html.append(
            f"  <div class='metadata-row'><span class='metadata-label'>Free-flow speed (baseline):</span>"
            f"<span class='metadata-value'>{mfd_p['free_flow_speed_kmh']:.1f} km/h</span></div>"
        )
        html.append(
            f"  <div class='metadata-row'><span class='metadata-label'>Jam density:</span>"
            f"<span class='metadata-value'>{mfd_p['jam_density_veh_per_km']:.1f} veh/km</span></div>"
        )
        html.append(
            f"  <div class='metadata-row'><span class='metadata-label'>Capacity:</span>"
            f"<span class='metadata-value'>{mfd_p['capacity_veh_per_hour']:.0f} veh/h</span></div>"
        )
        html.append(
            f"  <div class='metadata-row'><span class='metadata-label'>Greenshields R\u00b2:</span>"
            f"<span class='metadata-value'>{mfd_p['r_squared']:.3f}</span></div>"
        )
        html.append("</div>")

    # ── Section 4: Network Dynamics ──
    html.append("<h2>4. Network Dynamics</h2>")
    html.append(
        "<p class='section-note'>Ensemble of all non-baseline simulation runs "
        "(individual traces + mean ± IQR). Speed shown as 5-minute rolling average.</p>"
    )
    if "network_dynamics" in figures:
        html.append(_img_tag(figures["network_dynamics"], "Network Dynamics"))
    else:
        html.append("<p class='missing-img'>[Network dynamics figure not available]</p>")

    # ── Section 4: Resilience Statistics ──
    html.append("<h2>5. Resilience Statistics</h2>")
    html.append(
        "<p class='section-note'>AI distribution, accident counts, AI–accident correlation, "
        "and antifragility category breakdown across all incident scenarios.</p>"
    )
    if "resilience_statistics" in figures:
        html.append(_img_tag(figures["resilience_statistics"], "Resilience Statistics"))
    else:
        html.append("<p class='missing-img'>[Resilience statistics figure not available]</p>")

    # ── Section 5: Accident Characteristics ──
    html.append("<h2>6. Accident Characteristics</h2>")
    html.append(
        "<p class='section-note'>Severity distribution, incident durations, temporal "
        "trigger patterns, and vehicle impact across all incident scenarios.</p>"
    )
    if "accident_characteristics" in figures:
        html.append(_img_tag(figures["accident_characteristics"], "Accident Characteristics"))
    else:
        html.append("<p class='missing-img'>[Accident characteristics figure not available]</p>")

    # ── Section 6: Per-Event AI Analysis ──
    html.append("<h2>7. Per-Event Antifragility Analysis</h2>")
    html.append(
        "<p class='section-note'>Per-event AI values by severity class and pre/post-incident "
        "speed comparisons.</p>"
    )
    if "per_event_ai" in figures:
        html.append(_img_tag(figures["per_event_ai"], "Per-Event AI"))
    else:
        html.append("<p class='missing-img'>[Per-event AI figure not available]</p>")

    # ── Section 7: Spatial Heatmap ──
    html.append("<h2>8. Spatial Accident Heatmap</h2>")
    html.append(
        "<p class='section-note'>Kernel-density estimated accident hotspots and severity "
        "scatter plot overlaid on the Thessaloniki road network.</p>"
    )
    if "spatial_heatmap" in figures:
        html.append(_img_tag(figures["spatial_heatmap"], "Spatial Heatmap"))
    else:
        html.append("<p class='missing-img'>[Spatial heatmap figure not available]</p>")

    # ── Section 8: Weak Point Analysis ──
    html.append("<h2>9. Weak Point Analysis</h2>")
    if "weak_point_map" in figures:
        html.append(_img_tag(figures["weak_point_map"], "Weak Point Map"))

    if resilience_score.weak_points:
        html.append("<h3>Top Vulnerable Edges</h3>")
        html.append("<table class='scenario-table'>")
        html.append(
            "  <tr><th>#</th><th>Edge ID</th><th>Incidents</th>"
            "<th>Mean Duration (s)</th><th>Mean Affected</th>"
            "<th>Importance</th><th>Vulnerability</th></tr>"
        )
        for i, wp in enumerate(resilience_score.weak_points, 1):
            html.append(
                f"  <tr><td>{i}</td><td>{wp.edge_id}</td>"
                f"<td>{wp.accident_count}</td>"
                f"<td>{wp.mean_duration_seconds:.0f}</td>"
                f"<td>{wp.mean_vehicles_affected:.1f}</td>"
                f"<td>{wp.edge_importance:.3f}</td>"
                f"<td>{wp.vulnerability_index:.4f}</td></tr>"
            )
        html.append("</table>")
    else:
        html.append("<p>No weak points identified (no accident data available).</p>")

    # ── Section 9: Scenario Results ──
    html.append("<h2>10. Scenario Results</h2>")
    for period in demand_levels:
        period_scenarios = [s for s in scenarios if s.period == period]
        html.append("<details>")
        html.append(f"  <summary>Demand Level p={period} ({len(period_scenarios)} runs)</summary>")
        html.append("  <table class='scenario-table'>")
        html.append(
            "    <tr><th>Scenario</th><th>Status</th><th>Accidents</th>"
            "<th>Mean Speed (km/h)</th><th>AI</th></tr>"
        )
        for s in scenarios:
            if s.period != period:
                continue
            r = all_results[scenarios.index(s)] if scenarios.index(s) < len(all_results) else {}
            status = r.get("status", "missing")
            summary = r.get("summary", {})
            accidents = summary.get("total_accidents", "-")
            speed = summary.get("mean_speed_kmh")
            speed_str = f"{speed:.1f}" if speed is not None else "-"
            ai = summary.get("antifragility_index")
            ai_str = f"{ai:.3f}" if ai is not None else "-"
            status_icon = "OK" if status == "success" else "FAIL"
            html.append(
                f"    <tr><td>{s.scenario_id}</td><td>{status_icon}</td>"
                f"<td>{accidents}</td><td>{speed_str}</td><td>{ai_str}</td></tr>"
            )
        html.append("  </table>")
        html.append("</details>")

    # ── Section 10: Recommendations ──
    html.append("<h2>11. Recommendations</h2>")
    html.append("<div class='recommendations'>")
    recommendations = _generate_recommendations(resilience_score)
    html.append("  <ol>")
    for rec in recommendations:
        html.append(f"    <li><strong>{rec['title']}</strong>: {rec['description']}</li>")
    html.append("  </ol>")
    html.append("</div>")

    # ── Section 11: Claude AI-Assisted Analysis ──
    html.append("<h2>12. AI-Assisted Expert Analysis</h2>")
    if claude_analysis:
        html.append("<div class='claude-analysis'>")
        html.append(
            "  <p class='claude-badge'>&#129302; Generated by Claude (Anthropic) — "
            "expert analysis of methodology, findings, and recommended interventions</p>"
        )
        # Convert basic markdown to HTML (headings, bold, lists)
        html.append("  <div class='claude-body'>")
        html.append(_markdown_to_html(claude_analysis))
        html.append("  </div>")
        html.append("</div>")
    else:
        html.append("<div class='claude-analysis claude-missing'>")
        html.append(
            "  <p>AI-assisted analysis not available for this run.<br>"
            "  To enable: set the <code>ANTHROPIC_API_KEY</code> environment variable "
            "  before running the assessment.</p>"
        )
        html.append("</div>")

    # ── Section 12: Appendix ──
    html.append("<details>")
    html.append(
        "  <summary><h2 style='display:inline'>13. Appendix — Configuration</h2></summary>"
    )
    html.append("  <pre class='config-block'>")
    # Sanitize config for display (remove file paths that might be sensitive).
    display_config = {k: v for k, v in config.items() if k != "resilience_assessment"}
    html.append(json.dumps(display_config, indent=2, default=str))
    html.append("  </pre>")
    html.append("</details>")

    # ── Footer ──
    html.append("<div class='footer'>")
    html.append("  Generated by SAS One-Click Resilience Assessment<br>")
    html.append(
        "  <a href='https://github.com/tvlahopanagiotis/sumo-accident-simulation'>"
        "github.com/tvlahopanagiotis/sumo-accident-simulation</a>"
    )
    html.append("</div>")

    html.append("</div>")  # container
    html.append("</body>")
    html.append("</html>")

    html_content = "\n".join(html)
    report_path = os.path.join(output_dir, "resilience_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("Resilience report saved → %s", report_path)
    return report_path


# ---------------------------------------------------------------------------
# Markdown → HTML (lightweight, no external deps)
# ---------------------------------------------------------------------------


def _markdown_to_html(text: str) -> str:
    """
    Convert a subset of Markdown to HTML for the Claude analysis section.

    Handles:
      - ### Heading 3 → <h3>
      - **bold** → <strong>
      - Bullet lines starting with - or * → <ul><li>
      - Numbered lines → <ol><li>
      - Blank lines → paragraph breaks
    """
    import re

    lines = text.split("\n")
    out: list[str] = []
    in_ul = False
    in_ol = False

    def close_lists() -> None:
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    def inline(s: str) -> str:
        # **bold**
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        # *italic*
        s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
        # `code`
        s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        return s

    for line in lines:
        stripped = line.strip()

        # Headings
        if stripped.startswith("### "):
            close_lists()
            out.append(f"<h3>{inline(stripped[4:])}</h3>")
        elif stripped.startswith("## "):
            close_lists()
            out.append(f"<h4>{inline(stripped[3:])}</h4>")
        elif stripped.startswith("# "):
            close_lists()
            out.append(f"<h4>{inline(stripped[2:])}</h4>")
        # Horizontal rule
        elif stripped in ("---", "***", "___"):
            close_lists()
            out.append("<hr>")
        # Unordered list item
        elif re.match(r"^[-*]\s+", stripped):
            if in_ol:
                out.append("</ol>")
                in_ol = False
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            content = re.sub(r"^[-*]\s+", "", stripped)
            out.append(f"<li>{inline(content)}</li>")
        # Ordered list item
        elif re.match(r"^\d+\.\s+", stripped):
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if not in_ol:
                out.append("<ol>")
                in_ol = True
            content = re.sub(r"^\d+\.\s+", "", stripped)
            out.append(f"<li>{inline(content)}</li>")
        # Blank line → paragraph break
        elif stripped == "":
            close_lists()
            out.append("<br>")
        # Normal paragraph text
        else:
            close_lists()
            out.append(f"<p>{inline(stripped)}</p>")

    close_lists()
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Recommendations engine
# ---------------------------------------------------------------------------


def _generate_recommendations(score: ResilienceScore) -> list[dict[str, str]]:
    """Generate prioritised recommendations based on score components."""
    recs: list[dict[str, str]] = []

    # Priority: address the weakest component first.
    components = [
        ("speed", score.speed_resilience, "Speed"),
        ("throughput", score.throughput_resilience, "Throughput"),
        ("recovery", score.recovery_resilience, "Recovery"),
        ("robustness", score.robustness, "Robustness"),
    ]
    components.sort(key=lambda x: x[1])

    for key, val, label in components:
        if val < 0.6:
            if key == "speed":
                recs.append(
                    {
                        "title": "Improve speed maintenance under incidents",
                        "description": "Consider adding alternative routes, dynamic rerouting, "
                        "or variable message signs to distribute traffic away from incident zones. "
                        f"Current speed resilience ({val:.2f}) indicates significant speed drops during incidents.",
                    }
                )
            elif key == "throughput":
                recs.append(
                    {
                        "title": "Increase throughput resilience",
                        "description": "Review bottleneck junctions and consider signal timing optimisation, "
                        "lane management, or capacity improvements at critical intersections. "
                        f"Current throughput resilience ({val:.2f}) shows substantial flow reduction under stress.",
                    }
                )
            elif key == "recovery":
                recs.append(
                    {
                        "title": "Improve post-incident recovery",
                        "description": "Reduce incident response times through better emergency coordination, "
                        "dedicated incident management teams, and faster clearance protocols. "
                        f"Current recovery score ({val:.2f}) indicates slow return to normal operations.",
                    }
                )
            elif key == "robustness":
                recs.append(
                    {
                        "title": "Strengthen network robustness",
                        "description": "Improve network redundancy by adding alternative corridors, "
                        "reducing dependency on high-centrality links, and implementing traffic management "
                        f"strategies. Current robustness ({val:.2f}) shows high sensitivity to incident rates.",
                    }
                )
        elif val < 0.8:
            recs.append(
                {
                    "title": f"Monitor {label.lower()} resilience",
                    "description": f"{label} resilience ({val:.2f}) is adequate but could be improved. "
                    "Consider targeted interventions at the identified weak points.",
                }
            )

    # Weak point specific recommendations.
    if score.weak_points:
        top_edges = [wp.edge_id for wp in score.weak_points[:3]]
        recs.append(
            {
                "title": "Address identified weak points",
                "description": f"The most vulnerable edges ({', '.join(top_edges)}) should be prioritised "
                "for infrastructure improvements, incident response pre-positioning, "
                "and traffic management measures.",
            }
        )

    if not recs:
        recs.append(
            {
                "title": "Maintain current performance",
                "description": "The network demonstrates strong resilience across all dimensions. "
                "Continue monitoring and periodic assessment to maintain this level of performance.",
            }
        )

    return recs


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------


def _css() -> str:
    """Return the CSS for the resilience report."""
    return """
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 40px;
    background: #f5f5f5;
    color: #333;
}
.container {
    max-width: 1400px;
    margin: 0 auto;
    background: white;
    padding: 30px 40px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
h1 {
    color: #2c3e50;
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
}
h2 {
    color: #34495e;
    margin-top: 35px;
    border-left: 4px solid #3498db;
    padding-left: 10px;
}
h3 { color: #555; margin-top: 20px; }
.subtitle { color: #7f8c8d; font-size: 0.95em; margin-top: -10px; }

/* Executive summary */
.executive-summary {
    display: flex;
    gap: 40px;
    align-items: center;
    margin: 20px 0;
    padding: 20px;
    background: #f8f9fa;
    border-radius: 8px;
}
.grade-section { text-align: center; }
.grade-badge {
    display: inline-block;
    width: 80px;
    height: 80px;
    line-height: 80px;
    border-radius: 50%;
    color: white;
    font-size: 32px;
    font-weight: bold;
    text-align: center;
}
.score-value { font-size: 1.4em; font-weight: bold; color: #2c3e50; margin-top: 8px; }
.score-interp { font-size: 0.9em; color: #666; max-width: 250px; margin-top: 4px; }
.summary-stats {
    display: flex;
    gap: 30px;
}
.stat {
    text-align: center;
}
.stat-val {
    display: block;
    font-size: 1.8em;
    font-weight: bold;
    color: #3498db;
}
.stat-label {
    display: block;
    font-size: 0.85em;
    color: #7f8c8d;
}

/* Key findings */
.key-findings {
    background: #ecf0f1;
    padding: 15px 20px;
    border-radius: 5px;
    margin: 15px 0;
}
.key-findings h3 { margin-top: 0; }
.key-findings li { margin: 6px 0; }

/* Metadata */
.metadata {
    background: #ecf0f1;
    padding: 15px;
    border-radius: 5px;
    margin: 15px 0;
    font-size: 0.95em;
}
.metadata-row {
    display: grid;
    grid-template-columns: 220px 1fr;
    margin: 6px 0;
}
.metadata-label { font-weight: bold; color: #2c3e50; }
.metadata-value { color: #555; font-family: 'Courier New', monospace; }

/* Tables */
.scenario-table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    font-size: 0.9em;
}
.scenario-table th, .scenario-table td {
    padding: 8px 12px;
    border: 1px solid #ddd;
    text-align: left;
}
.scenario-table th {
    background: #3498db;
    color: white;
}
.scenario-table tr:nth-child(even) { background: #f8f9fa; }

/* Images */
img {
    max-width: 100%;
    height: auto;
    margin: 15px 0;
    border: 1px solid #ddd;
    border-radius: 4px;
}
.image-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
}
@media (max-width: 1024px) {
    .image-grid { grid-template-columns: 1fr; }
    .executive-summary { flex-direction: column; }
}
.missing-img { color: #999; font-style: italic; }

/* Details / collapsible */
details {
    margin: 10px 0;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 5px 15px;
}
summary {
    cursor: pointer;
    font-weight: bold;
    color: #34495e;
    padding: 8px 0;
}
details[open] summary { border-bottom: 1px solid #ddd; margin-bottom: 10px; }

/* Config block */
.config-block {
    background: #f4f4f4;
    padding: 15px;
    border-radius: 5px;
    font-size: 0.85em;
    overflow-x: auto;
    max-height: 400px;
    overflow-y: auto;
}

/* Recommendations */
.recommendations {
    background: #fff3cd;
    padding: 15px 20px;
    border-left: 4px solid #ffc107;
    border-radius: 4px;
    margin: 15px 0;
}
.recommendations ol { padding-left: 20px; }
.recommendations li { margin: 10px 0; }

/* Section notes */
.section-note {
    font-size: 0.88em;
    color: #666;
    font-style: italic;
    margin: -8px 0 12px 0;
}

/* Claude analysis */
.claude-analysis {
    background: #f0f4ff;
    border-left: 4px solid #3f51b5;
    border-radius: 4px;
    padding: 20px 24px;
    margin: 15px 0;
}
.claude-badge {
    font-size: 0.85em;
    color: #3f51b5;
    font-weight: bold;
    border-bottom: 1px solid #c5cae9;
    padding-bottom: 8px;
    margin-bottom: 16px;
}
.claude-body h3 { color: #3f51b5; margin-top: 18px; font-size: 1.05em; }
.claude-body h4 { color: #5c6bc0; margin-top: 14px; font-size: 0.95em; }
.claude-body ul, .claude-body ol { margin: 6px 0; padding-left: 22px; }
.claude-body li { margin: 4px 0; }
.claude-body code {
    background: #e8eaf6;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.9em;
}
.claude-missing {
    background: #fafafa;
    border-left: 4px solid #bdbdbd;
    color: #757575;
}
.claude-missing p { margin: 0; font-size: 0.95em; }

/* Footer */
.footer {
    text-align: center;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #ddd;
    color: #7f8c8d;
    font-size: 0.9em;
}

/* Print */
@media print {
    body { margin: 0; background: white; }
    .container { box-shadow: none; padding: 20px; }
    details { border: none; }
    details[open] > summary { display: none; }
}
"""
