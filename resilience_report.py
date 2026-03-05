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

import base64
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from mfd_analysis import EdgeVulnerability, ResilienceScore
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

    Returns:
        Path to the generated report.html file.
    """
    grade = resilience_score.grade
    grade_color = GRADE_COLORS.get(grade, "#999999")
    score = resilience_score.overall_score
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Count successes / failures.
    n_success = sum(1 for r in all_results if r.get("status") == "success")
    n_fail = sum(1 for r in all_results if r.get("status") != "success")
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
    html.append(f"    <div class='stat'><span class='stat-val'>{n_total}</span><span class='stat-label'>Scenarios</span></div>")
    html.append(f"    <div class='stat'><span class='stat-val'>{n_success}</span><span class='stat-label'>Successful</span></div>")
    html.append(f"    <div class='stat'><span class='stat-val'>{len(demand_levels)}</span><span class='stat-label'>Demand Levels</span></div>")
    html.append(f"    <div class='stat'><span class='stat-val'>{len(incident_types)}</span><span class='stat-label'>Incident Types</span></div>")
    html.append("  </div>")
    html.append("</div>")

    # Key findings.
    html.append("<div class='key-findings'>")
    html.append("  <h3>Key Findings</h3>")
    html.append("  <ul>")
    html.append(f"    <li>Speed resilience: <strong>{resilience_score.speed_resilience:.2f}</strong> — "
                f"{'minimal' if resilience_score.speed_resilience > 0.85 else 'some' if resilience_score.speed_resilience > 0.7 else 'significant'} "
                f"speed degradation under incidents</li>")
    html.append(f"    <li>Throughput resilience: <strong>{resilience_score.throughput_resilience:.2f}</strong> — "
                f"{'flow well maintained' if resilience_score.throughput_resilience > 0.85 else 'moderate flow impact' if resilience_score.throughput_resilience > 0.7 else 'significant flow reduction'}</li>")
    html.append(f"    <li>Recovery: <strong>{resilience_score.recovery_resilience:.2f}</strong> — "
                f"{'excellent' if resilience_score.recovery_resilience > 0.85 else 'good' if resilience_score.recovery_resilience > 0.6 else 'slow'} "
                f"post-incident recovery (AI = {resilience_score.ai_aggregate:.3f})</li>")
    html.append(f"    <li>Robustness: <strong>{resilience_score.robustness:.2f}</strong> — "
                f"{'strong' if resilience_score.robustness > 0.85 else 'moderate' if resilience_score.robustness > 0.6 else 'weak'} "
                f"tolerance to increasing incident rates</li>")
    if resilience_score.weak_points:
        top_wp = resilience_score.weak_points[0]
        html.append(f"    <li>Most vulnerable edge: <strong>{top_wp.edge_id}</strong> "
                    f"({top_wp.accident_count} incidents, vulnerability index {top_wp.vulnerability_index:.3f})</li>")
    html.append("  </ul>")
    html.append("</div>")

    # ── Section 2: Network Overview ──
    html.append("<h2>2. Network Overview</h2>")
    html.append("<div class='metadata'>")
    mfd_p = resilience_score.mfd_parameters
    if "free_flow_speed_kmh" in mfd_p:
        html.append(f"  <div class='metadata-row'><span class='metadata-label'>Free-flow speed:</span>"
                    f"<span class='metadata-value'>{mfd_p['free_flow_speed_kmh']:.1f} km/h</span></div>")
        html.append(f"  <div class='metadata-row'><span class='metadata-label'>Jam density:</span>"
                    f"<span class='metadata-value'>{mfd_p['jam_density_veh_per_km']:.1f} veh/km</span></div>")
        html.append(f"  <div class='metadata-row'><span class='metadata-label'>Capacity:</span>"
                    f"<span class='metadata-value'>{mfd_p['capacity_veh_per_hour']:.0f} veh/h</span></div>")
        html.append(f"  <div class='metadata-row'><span class='metadata-label'>Greenshields R\u00b2:</span>"
                    f"<span class='metadata-value'>{mfd_p['r_squared']:.3f}</span></div>")
    html.append(f"  <div class='metadata-row'><span class='metadata-label'>Demand levels tested:</span>"
                f"<span class='metadata-value'>{', '.join(f'p={p}' for p in demand_levels)}</span></div>")
    html.append(f"  <div class='metadata-row'><span class='metadata-label'>Scenario types:</span>"
                f"<span class='metadata-value'>{', '.join(incident_types)}</span></div>")
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

    # ── Section 3: MFD Analysis ──
    html.append("<h2>3. Macroscopic Fundamental Diagram</h2>")
    html.append("<div class='image-grid'>")
    for key in ["mfd_density_flow", "mfd_density_speed"]:
        if key in figures:
            html.append(_img_tag(figures[key], key.replace("_", " ").title()))
    html.append("</div>")

    # ── Section 4: Resilience Breakdown ──
    html.append("<h2>4. Resilience Score Breakdown</h2>")
    if "resilience_components" in figures:
        html.append(_img_tag(figures["resilience_components"], "Resilience Components"))

    html.append("<div class='image-grid'>")
    for key in ["speed_comparison", "throughput_comparison"]:
        if key in figures:
            html.append(_img_tag(figures[key], key.replace("_", " ").title()))
    html.append("</div>")

    if "ai_distribution" in figures:
        html.append(_img_tag(figures["ai_distribution"], "AI Distribution"))

    # ── Section 5: Weak Point Analysis ──
    html.append("<h2>5. Weak Point Analysis</h2>")
    if "weak_point_map" in figures:
        html.append(_img_tag(figures["weak_point_map"], "Weak Point Map"))

    if resilience_score.weak_points:
        html.append("<h3>Top Vulnerable Edges</h3>")
        html.append("<table class='scenario-table'>")
        html.append("  <tr><th>#</th><th>Edge ID</th><th>Incidents</th>"
                    "<th>Mean Duration (s)</th><th>Mean Affected</th>"
                    "<th>Importance</th><th>Vulnerability</th></tr>")
        for i, wp in enumerate(resilience_score.weak_points, 1):
            html.append(f"  <tr><td>{i}</td><td>{wp.edge_id}</td>"
                       f"<td>{wp.accident_count}</td>"
                       f"<td>{wp.mean_duration_seconds:.0f}</td>"
                       f"<td>{wp.mean_vehicles_affected:.1f}</td>"
                       f"<td>{wp.edge_importance:.3f}</td>"
                       f"<td>{wp.vulnerability_index:.4f}</td></tr>")
        html.append("</table>")
    else:
        html.append("<p>No weak points identified (no accident data available).</p>")

    # ── Section 6: Scenario Results ──
    html.append("<h2>6. Scenario Results</h2>")
    for period in demand_levels:
        period_scenarios = [s for s in scenarios if s.period == period]
        period_results = [
            all_results[i] if i < len(all_results) else {"status": "missing"}
            for i, s in enumerate(scenarios)
            if s.period == period
        ]
        html.append(f"<details>")
        html.append(f"  <summary>Demand Level p={period} ({len(period_scenarios)} runs)</summary>")
        html.append("  <table class='scenario-table'>")
        html.append("    <tr><th>Scenario</th><th>Status</th><th>Accidents</th>"
                    "<th>Mean Speed (km/h)</th><th>AI</th></tr>")
        result_idx = 0
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
            html.append(f"    <tr><td>{s.scenario_id}</td><td>{status_icon}</td>"
                       f"<td>{accidents}</td><td>{speed_str}</td><td>{ai_str}</td></tr>")
        html.append("  </table>")
        html.append("</details>")

    # ── Section 7: Recommendations ──
    html.append("<h2>7. Recommendations</h2>")
    html.append("<div class='recommendations'>")
    recommendations = _generate_recommendations(resilience_score)
    html.append("  <ol>")
    for rec in recommendations:
        html.append(f"    <li><strong>{rec['title']}</strong>: {rec['description']}</li>")
    html.append("  </ol>")
    html.append("</div>")

    # ── Section 8: Appendix ──
    html.append("<details>")
    html.append("  <summary><h2 style='display:inline'>8. Appendix — Configuration</h2></summary>")
    html.append("  <pre class='config-block'>")
    # Sanitize config for display (remove file paths that might be sensitive).
    display_config = {k: v for k, v in config.items() if k != "resilience_assessment"}
    html.append(json.dumps(display_config, indent=2, default=str))
    html.append("  </pre>")
    html.append("</details>")

    # ── Footer ──
    html.append("<div class='footer'>")
    html.append("  Generated by SAS One-Click Resilience Assessment<br>")
    html.append("  <a href='https://github.com/tvlahopanagiotis/sumo-accident-simulation'>"
                "github.com/tvlahopanagiotis/sumo-accident-simulation</a>")
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
                recs.append({
                    "title": "Improve speed maintenance under incidents",
                    "description": "Consider adding alternative routes, dynamic rerouting, "
                    "or variable message signs to distribute traffic away from incident zones. "
                    f"Current speed resilience ({val:.2f}) indicates significant speed drops during incidents.",
                })
            elif key == "throughput":
                recs.append({
                    "title": "Increase throughput resilience",
                    "description": "Review bottleneck junctions and consider signal timing optimisation, "
                    "lane management, or capacity improvements at critical intersections. "
                    f"Current throughput resilience ({val:.2f}) shows substantial flow reduction under stress.",
                })
            elif key == "recovery":
                recs.append({
                    "title": "Improve post-incident recovery",
                    "description": "Reduce incident response times through better emergency coordination, "
                    "dedicated incident management teams, and faster clearance protocols. "
                    f"Current recovery score ({val:.2f}) indicates slow return to normal operations.",
                })
            elif key == "robustness":
                recs.append({
                    "title": "Strengthen network robustness",
                    "description": "Improve network redundancy by adding alternative corridors, "
                    "reducing dependency on high-centrality links, and implementing traffic management "
                    f"strategies. Current robustness ({val:.2f}) shows high sensitivity to incident rates.",
                })
        elif val < 0.8:
            recs.append({
                "title": f"Monitor {label.lower()} resilience",
                "description": f"{label} resilience ({val:.2f}) is adequate but could be improved. "
                "Consider targeted interventions at the identified weak points.",
            })

    # Weak point specific recommendations.
    if score.weak_points:
        top_edges = [wp.edge_id for wp in score.weak_points[:3]]
        recs.append({
            "title": "Address identified weak points",
            "description": f"The most vulnerable edges ({', '.join(top_edges)}) should be prioritised "
            "for infrastructure improvements, incident response pre-positioning, "
            "and traffic management measures.",
        })

    if not recs:
        recs.append({
            "title": "Maintain current performance",
            "description": "The network demonstrates strong resilience across all dimensions. "
            "Continue monitoring and periodic assessment to maintain this level of performance.",
        })

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
