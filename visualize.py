"""
visualize.py
============

Post-simulation visualization module for SAS (SUMO Accident Simulation).
Generates publication-ready plots and HTML reports from simulation results.

Functions:
  - plot_network_metrics() — Time-series plots of vehicle count, speed, accidents
  - plot_severity_distribution() — Pie chart of accident severity tiers
  - plot_before_after_speeds() — Histograms comparing speeds before/after accidents
  - plot_accident_heatmap() — 2D scatter of accident locations colored by severity
  - generate_html_report() — Single HTML file embedding all visualizations
  - visualize_batch_results() — Batch-level summary (per-run AI scores, batch heatmap)
"""

from __future__ import annotations

import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────────

sns.set_style("whitegrid")
sns.set_palette("husl")
FIGURE_DPI = 150
FIGSIZE_WIDE = (14, 5)
FIGSIZE_SQUARE = (10, 8)

SEVERITY_COLORS = {
    "MINOR": "#2ecc71",  # green
    "MODERATE": "#f39c12",  # orange
    "MAJOR": "#e74c3c",  # red
    "CRITICAL": "#8b0000",  # dark red
}

SEVERITY_ORDER = ["MINOR", "MODERATE", "MAJOR", "CRITICAL"]


# ───────────────────────────────────────────────────────────────────────────────
# Network Metrics Timeseries
# ───────────────────────────────────────────────────────────────────────────────


def plot_network_metrics(
    metrics_csv: str,
    output_dir: str,
    run_id: str | None = None,
) -> None:
    """
    Plot network metrics over time: vehicle count, mean speed, active accidents.

    Args:
        metrics_csv: Path to network_metrics.csv
        output_dir: Directory to save PNG
        run_id: Optional run identifier (e.g., seed) for plot title
    """
    if not os.path.exists(metrics_csv):
        logger.warning("Metrics CSV not found: %s", metrics_csv)
        return

    try:
        df = pd.read_csv(metrics_csv)
    except Exception as exc:
        logger.error("Failed to read metrics CSV: %s", exc)
        return

    # Convert timestamp to hours
    df["hours"] = df["timestamp_seconds"] / 3600

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE, sharey=False)

    # Plot 1: Vehicle count
    axes[0].plot(
        df["hours"],
        df["vehicle_count"],
        linewidth=2.5,
        color="#3498db",
        label="Simultaneous vehicles",
    )
    axes[0].set_xlabel("Simulation Time (hours)", fontsize=11)
    axes[0].set_ylabel("Vehicle Count", fontsize=11)
    axes[0].set_title("Network Occupancy", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Mean speed
    axes[1].plot(
        df["hours"],
        df["mean_speed_kmh"],
        linewidth=2.5,
        color="#e74c3c",
        label="Mean speed",
    )
    axes[1].fill_between(
        df["hours"],
        df["mean_speed_kmh"],
        alpha=0.2,
        color="#e74c3c",
    )
    axes[1].set_xlabel("Simulation Time (hours)", fontsize=11)
    axes[1].set_ylabel("Speed (km/h)", fontsize=11)
    axes[1].set_title("Network Congestion", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Active accidents
    axes[2].bar(
        df["hours"],
        df["active_accidents"],
        width=0.02,
        color="#9b59b6",
        alpha=0.7,
        label="Active accidents",
    )
    axes[2].set_xlabel("Simulation Time (hours)", fontsize=11)
    axes[2].set_ylabel("Concurrent Accidents", fontsize=11)
    axes[2].set_title("Accident Activity", fontsize=12, fontweight="bold")
    axes[2].grid(True, alpha=0.3, axis="y")
    axes[2].set_ylim(bottom=0)

    title = "Network Metrics Over Time"
    if run_id is not None:
        title += f" (Run {run_id})"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "metrics_timeseries.png")
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Network metrics plot saved → %s", out_path)


# ───────────────────────────────────────────────────────────────────────────────
# Severity Distribution
# ───────────────────────────────────────────────────────────────────────────────


def plot_severity_distribution(
    accident_reports_json: str,
    output_dir: str,
    run_id: str | None = None,
) -> None:
    """
    Plot pie chart of accident severity distribution.

    Args:
        accident_reports_json: Path to accident_reports.json
        output_dir: Directory to save PNG
        run_id: Optional run identifier for plot title
    """
    if not os.path.exists(accident_reports_json):
        logger.warning("Accident reports JSON not found: %s", accident_reports_json)
        return

    try:
        with open(accident_reports_json) as f:
            accidents = json.load(f)
    except Exception as exc:
        logger.error("Failed to read accident reports: %s", exc)
        return

    if not accidents:
        logger.warning("No accidents in report")
        return

    # Count severities
    severity_counts: dict[str, int] = {}
    for acc in accidents:
        sev = acc.get("severity", "UNKNOWN")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    # Order and filter
    severities = [s for s in SEVERITY_ORDER if s in severity_counts]
    counts = [severity_counts[s] for s in severities]
    colors = [SEVERITY_COLORS.get(s, "#95a5a6") for s in severities]

    # Create pie chart
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    wedges, texts, autotexts = ax.pie(  # type: ignore[misc]
        counts,
        labels=severities,
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        textprops={"fontsize": 11},
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(10)

    title = f"Accident Severity Distribution (n={len(accidents)})"
    if run_id is not None:
        title += f" — Run {run_id}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "severity_distribution.png")
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Severity distribution plot saved → %s", out_path)


# ───────────────────────────────────────────────────────────────────────────────
# Before/After Speed Comparison
# ───────────────────────────────────────────────────────────────────────────────


def plot_before_after_speeds(
    accident_reports_json: str,
    network_metrics_csv: str,
    output_dir: str,
    run_id: str | None = None,
    window_seconds: int = 300,
) -> None:
    """
    Plot speed distributions before and after each accident.

    Args:
        accident_reports_json: Path to accident_reports.json
        network_metrics_csv: Path to network_metrics.csv
        output_dir: Directory to save PNG
        run_id: Optional run identifier for plot title
        window_seconds: Window before/after accident to measure (default 300 s = 5 min)
    """
    if not os.path.exists(accident_reports_json):
        logger.warning("Accident reports JSON not found: %s", accident_reports_json)
        return
    if not os.path.exists(network_metrics_csv):
        logger.warning("Metrics CSV not found: %s", network_metrics_csv)
        return

    try:
        with open(accident_reports_json) as f:
            accidents = json.load(f)
        df_metrics = pd.read_csv(network_metrics_csv)
    except Exception as exc:
        logger.error("Failed to read input files: %s", exc)
        return

    if not accidents:
        logger.warning("No accidents to plot")
        return

    # Create subplots: one per accident, max 4 per row
    n_accidents = len(accidents)
    n_cols = min(4, n_accidents)
    n_rows = (n_accidents + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharey=True)

    if n_accidents == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    axes = axes.flatten()

    for idx, acc in enumerate(accidents):
        ax = axes[idx]
        trigger_step = acc["trigger_step"]

        # Filter metrics by timestamp (trigger_step ≈ timestamp_seconds in this model)
        trigger_time = trigger_step
        before_mask = (df_metrics["timestamp_seconds"] >= trigger_time - window_seconds) & (
            df_metrics["timestamp_seconds"] < trigger_time
        )
        after_mask = (df_metrics["timestamp_seconds"] >= trigger_time) & (
            df_metrics["timestamp_seconds"] < trigger_time + window_seconds
        )

        before_speeds = df_metrics.loc[before_mask, "mean_speed_kmh"].values
        after_speeds = df_metrics.loc[after_mask, "mean_speed_kmh"].values

        # Plot histograms
        if len(before_speeds) > 0:
            ax.hist(
                before_speeds,
                bins=8,
                alpha=0.6,
                color="#2ecc71",
                label="Before",
                edgecolor="black",
            )
        if len(after_speeds) > 0:
            ax.hist(
                after_speeds,
                bins=8,
                alpha=0.6,
                color="#e74c3c",
                label="After",
                edgecolor="black",
            )

        ax.set_xlabel("Speed (km/h)", fontsize=10)
        ax.set_ylabel("Frequency" if idx % n_cols == 0 else "", fontsize=10)
        ax.set_title(f"{acc['accident_id']} (step {trigger_step})", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    # Hide unused subplots
    for idx in range(n_accidents, len(axes)):
        axes[idx].set_visible(False)

    title = "Speed Impact: Before vs After Accidents"
    if run_id is not None:
        title += f" — Run {run_id}"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()

    out_path = os.path.join(output_dir, "before_after_comparison.png")
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Before/after speed plot saved → %s", out_path)


# ───────────────────────────────────────────────────────────────────────────────
# Accident Heatmap
# ───────────────────────────────────────────────────────────────────────────────


def plot_accident_heatmap(
    accident_reports_json: str,
    output_dir: str,
    run_id: str | None = None,
) -> None:
    """
    Plot 2D scatter of accident locations colored by severity.

    Args:
        accident_reports_json: Path to accident_reports.json
        output_dir: Directory to save PNG
        run_id: Optional run identifier for plot title
    """
    if not os.path.exists(accident_reports_json):
        logger.warning("Accident reports JSON not found: %s", accident_reports_json)
        return

    try:
        with open(accident_reports_json) as f:
            accidents = json.load(f)
    except Exception as exc:
        logger.error("Failed to read accident reports: %s", exc)
        return

    if not accidents:
        logger.warning("No accidents to plot")
        return

    # Extract coordinates and severity
    xs = []
    ys = []
    severities = []
    colors_list = []
    sizes = []

    severity_size_map = {"MINOR": 50, "MODERATE": 100, "MAJOR": 150, "CRITICAL": 200}

    for acc in accidents:
        loc = acc.get("location", {})
        x = loc.get("x", 0)
        y = loc.get("y", 0)
        sev = acc.get("severity", "UNKNOWN")

        xs.append(x)
        ys.append(y)
        severities.append(sev)
        colors_list.append(SEVERITY_COLORS.get(sev, "#95a5a6"))
        sizes.append(severity_size_map.get(sev, 75))

    # Plot
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    ax.scatter(xs, ys, c=colors_list, s=sizes, alpha=0.7, edgecolors="black", linewidth=1.5)

    ax.set_xlabel("X coordinate (m)", fontsize=11)
    ax.set_ylabel("Y coordinate (m)", fontsize=11)

    # Create custom legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=SEVERITY_COLORS.get(sev, "#95a5a6"), edgecolor="black", label=sev)
        for sev in SEVERITY_ORDER
        if sev in SEVERITY_COLORS
    ]
    ax.legend(
        handles=legend_elements, loc="best", fontsize=10, title="Severity", title_fontsize=11
    )

    title = f"Accident Location Heatmap (n={len(accidents)})"
    if run_id is not None:
        title += f" — Run {run_id}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "accident_heatmap.png")
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Accident heatmap saved → %s", out_path)


# ───────────────────────────────────────────────────────────────────────────────
# HTML Report Generation
# ───────────────────────────────────────────────────────────────────────────────


def generate_html_report(
    output_dir: str,
    run_id: str | None = None,
    config: dict | None = None,
    metadata: dict | None = None,
) -> None:
    """
    Generate an HTML report embedding all visualizations.

    Args:
        output_dir: Directory containing PNG files and JSON files
        run_id: Optional run identifier (e.g., seed)
        config: Optional config dict for display
        metadata: Optional metadata dict for display
    """
    output_dir = str(output_dir)

    # Collect images
    image_files = [
        "metrics_timeseries.png",
        "severity_distribution.png",
        "before_after_comparison.png",
        "accident_heatmap.png",
    ]

    # Read metadata if available
    metadata_path = os.path.join(output_dir, "metadata.json")
    if metadata is None and os.path.exists(metadata_path):
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except Exception:
            metadata = None

    # Read AI if available
    ai_path = os.path.join(output_dir, "antifragility_index.json")
    ai_data = None
    if os.path.exists(ai_path):
        try:
            with open(ai_path) as f:
                ai_data = json.load(f)
        except Exception:
            pass

    # Build HTML
    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '    <meta charset="UTF-8">',
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        "    <title>SAS Simulation Report</title>",
        "    <style>",
        "        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f5f5f5; }",
        "        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }",
        "        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }",
        "        h2 { color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 10px; }",
        "        .metadata { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 15px 0; font-size: 0.95em; }",
        "        .metadata-row { display: grid; grid-template-columns: 200px 1fr; margin: 8px 0; }",
        "        .metadata-label { font-weight: bold; color: #2c3e50; }",
        "        .metadata-value { color: #555; font-family: 'Courier New', monospace; word-break: break-all; }",
        "        .ai-highlight { background: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 10px 0; border-radius: 4px; }",
        "        .ai-positive { background: #d4edda; border-left-color: #28a745; }",
        "        .ai-negative { background: #f8d7da; border-left-color: #dc3545; }",
        "        img { max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; border-radius: 4px; }",
        "        .image-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }",
        "        @media (max-width: 1024px) { .image-grid { grid-template-columns: 1fr; } }",
        "        .footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 0.9em; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <div class='container'>",
        "        <h1>SUMO Accident Simulation Report</h1>",
    ]

    # Title info
    if metadata:
        run_id_info = metadata.get("run_id", run_id or "Unknown")
        html_parts.append(f"        <p><strong>Run ID:</strong> {run_id_info}</p>")

    # Metadata section
    if metadata:
        html_parts.append("        <h2>Simulation Metadata</h2>")
        html_parts.append("        <div class='metadata'>")

        for key in ["run_id", "timestamp_utc", "sumo_version", "total_steps", "total_accidents"]:
            if key in metadata:
                val = metadata[key]
                if isinstance(val, (int, float)):
                    val = f"{val:,}"
                html_parts.append(
                    f"            <div class='metadata-row'><span class='metadata-label'>{key}:</span><span class='metadata-value'>{val}</span></div>"
                )

        html_parts.append("        </div>")

    # AI section
    if ai_data:
        ai_score = ai_data.get("antifragility_index")
        n_events = ai_data.get("n_events_measured", 0)
        interpretation = ai_data.get("interpretation", "Unknown")

        ai_class = "ai-positive" if ai_score and ai_score > 0 else "ai-negative"
        html_parts.append("        <h2>Antifragility Index</h2>")
        html_parts.append(f"        <div class='ai-highlight {ai_class}'>")
        html_parts.append(
            f"            <strong>AI Score:</strong> {ai_score:.3f} ({interpretation})<br>"
        )
        html_parts.append(f"            <strong>Events Measured:</strong> {n_events}<br>")

        if "ci_95_low" in ai_data and "ci_95_high" in ai_data:
            ci_low = ai_data["ci_95_low"]
            ci_high = ai_data["ci_95_high"]
            html_parts.append(
                f"            <strong>95% Confidence Interval:</strong> [{ci_low:.3f}, {ci_high:.3f}]"
            )

        html_parts.append("        </div>")

    # Visualizations
    html_parts.append("        <h2>Simulation Results</h2>")
    html_parts.append("        <div class='image-grid'>")

    for img_file in image_files:
        img_path = os.path.join(output_dir, img_file)
        if os.path.exists(img_path):
            html_parts.append(f"            <img src='{img_file}' alt='{img_file}'>")

    html_parts.append("        </div>")

    # Footer
    html_parts.append("        <div class='footer'>")
    html_parts.append("            Generated by SAS (SUMO Accident Simulation)<br>")
    html_parts.append(
        "            <a href='https://github.com/tvlahopanagiotis/sumo-accident-simulation'>github.com/tvlahopanagiotis/sumo-accident-simulation</a>"
    )
    html_parts.append("        </div>")

    html_parts.extend(["    </div>", "</body>", "</html>"])

    html_content = "\n".join(html_parts)

    out_path = os.path.join(output_dir, "report.html")
    with open(out_path, "w") as f:
        f.write(html_content)

    logger.info("HTML report saved → %s", out_path)


# ───────────────────────────────────────────────────────────────────────────────
# Batch-level visualization
# ───────────────────────────────────────────────────────────────────────────────


def visualize_batch_results(batch_dir: str, all_summaries: list[dict]) -> None:
    """
    Generate batch-level visualizations: per-run AI scores.

    Args:
        batch_dir: Batch output directory containing all run results
        all_summaries: List of summary dicts from all runs
    """
    if not all_summaries:
        logger.warning("No summaries provided for batch visualization")
        return

    # Extract AI scores
    ai_scores = []
    run_ids = []

    for i, summary in enumerate(all_summaries):
        run_id = summary.get("run_id", f"Run {i}")
        ai_file = os.path.join(batch_dir, f"antifragility_{i + 1}.json")

        if os.path.exists(ai_file):
            try:
                with open(ai_file) as f:
                    ai_data = json.load(f)
                    ai_score = ai_data.get("antifragility_index")
                    if ai_score is not None:
                        ai_scores.append(ai_score)
                        run_ids.append(run_id)
            except Exception as exc:
                logger.warning("Failed to read AI for run %d: %s", i, exc)

    if not ai_scores:
        logger.warning("No AI scores found for batch visualization")
        return

    # Plot bar chart of per-run AI
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#2ecc71" if ai > 0 else "#e74c3c" for ai in ai_scores]
    bars = ax.bar(
        range(len(ai_scores)), ai_scores, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
    )

    # Add horizontal line at AI=0
    ax.axhline(y=0, color="black", linestyle="--", linewidth=2, alpha=0.5)

    ax.set_xlabel("Run", fontsize=12)
    ax.set_ylabel("Antifragility Index (AI)", fontsize=12)
    ax.set_title("Per-Run Antifragility Scores (Batch Summary)", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(ai_scores)))
    ax.set_xticklabels([f"Run {i + 1}" for i in range(len(ai_scores))], fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for _i, (bar, ai) in enumerate(zip(bars, ai_scores, strict=False)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.05 if height > 0 else height - 0.1,
            f"{ai:.2f}",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    out_path = os.path.join(batch_dir, "batch_ai_distribution.png")
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Batch AI distribution plot saved → %s", out_path)
