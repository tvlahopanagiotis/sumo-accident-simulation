"""
mfd_analysis.py
===============

Macroscopic Fundamental Diagram computation, weak-point analysis, and
composite resilience scoring for the one-click resilience assessment.

Operates entirely in post-processing: reads the network_metrics.csv and
accident_reports.json files produced by individual simulation runs,
computes density from vehicle_count / network_lane_km, fits traffic models,
identifies vulnerable edges, and produces a single resilience score.
"""

from __future__ import annotations

import json
import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit

from scenario_generator import Scenario

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plot style (consistent with visualize.py)
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
FIGURE_DPI = 150
FIGSIZE_WIDE = (14, 5)
FIGSIZE_SQUARE = (10, 8)

SCENARIO_COLORS = {
    "baseline": "#3498db",
    "low_incident": "#2ecc71",
    "default_incident": "#f39c12",
    "high_incident": "#e74c3c",
    "extreme_incident": "#8b0000",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EdgeVulnerability:
    """Vulnerability assessment for a single network edge."""

    edge_id: str
    x: float
    y: float
    accident_count: int
    mean_duration_seconds: float
    mean_vehicles_affected: float
    mean_speed_drop_ratio: float
    edge_importance: float  # traffic-volume proxy
    vulnerability_index: float


@dataclass
class ResilienceScore:
    """Composite resilience assessment result."""

    overall_score: float
    grade: str
    interpretation: str
    speed_resilience: float
    throughput_resilience: float
    recovery_resilience: float
    robustness: float
    weights: dict[str, float]
    mfd_parameters: dict
    ai_aggregate: float
    weak_points: list[EdgeVulnerability] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Network geometry helpers
# ---------------------------------------------------------------------------


def compute_network_lane_km(net_xml_path: str) -> float:
    """
    Parse the SUMO .net.xml to sum total lane length in km.

    Excludes internal junction edges (id starts with ":").

    Args:
        net_xml_path: Absolute path to the .net.xml file.

    Returns:
        Total lane-kilometres of the network.
    """
    tree = ET.parse(net_xml_path)
    root = tree.getroot()
    total_length_m = 0.0
    for edge in root.findall("edge"):
        if edge.get("id", "").startswith(":"):
            continue
        for lane in edge.findall("lane"):
            length = float(lane.get("length", "0.0"))
            total_length_m += length
    lane_km = total_length_m / 1000.0
    logger.info("Network total lane length: %.1f km", lane_km)
    return lane_km


# ---------------------------------------------------------------------------
# MFD data extraction
# ---------------------------------------------------------------------------


def extract_mfd_data(
    scenarios: list[Scenario],
    network_lane_km: float,
    warmup_seconds: int = 1200,
) -> pd.DataFrame:
    """
    Read all network_metrics.csv files and compute density for MFD analysis.

    Filters to steady-state snapshots (timestamp > warmup_seconds) to avoid
    the ramp-up phase distorting the MFD.

    Args:
        scenarios:        List of Scenario objects with output_folder paths.
        network_lane_km:  Total lane-km of the network (from compute_network_lane_km).
        warmup_seconds:   Seconds to skip at the start of each run.

    Returns:
        DataFrame with columns: period, scenario_type, seed, timestamp_seconds,
        density_veh_per_km, flow_veh_per_hour, speed_kmh, active_accidents.
    """
    frames: list[pd.DataFrame] = []

    for scenario in scenarios:
        csv_path = os.path.join(scenario.output_folder, "network_metrics.csv")
        if not os.path.exists(csv_path):
            logger.warning("Missing metrics CSV for %s, skipping", scenario.scenario_id)
            continue

        df = pd.read_csv(csv_path)
        df = df[df["timestamp_seconds"] > warmup_seconds].copy()
        if df.empty:
            continue

        df["period"] = scenario.period
        df["scenario_type"] = scenario.scenario_type
        df["seed"] = scenario.seed
        df["density_veh_per_km"] = df["vehicle_count"] / network_lane_km

        frames.append(
            df[
                [
                    "period",
                    "scenario_type",
                    "seed",
                    "timestamp_seconds",
                    "density_veh_per_km",
                    "throughput_per_hour",
                    "mean_speed_kmh",
                    "active_accidents",
                ]
            ].rename(
                columns={
                    "throughput_per_hour": "flow_veh_per_hour",
                    "mean_speed_kmh": "speed_kmh",
                }
            )
        )

    if not frames:
        logger.warning("No MFD data extracted — all CSVs missing or empty")
        return pd.DataFrame()

    mfd = pd.concat(frames, ignore_index=True)
    logger.info("Extracted %d MFD data points from %d scenario runs", len(mfd), len(frames))
    return mfd


# ---------------------------------------------------------------------------
# Greenshields model fitting
# ---------------------------------------------------------------------------


def _greenshields_speed(density: np.ndarray, vf: float, kj: float) -> np.ndarray:
    """Greenshields linear speed–density model: v = vf × (1 - k/kj)."""
    return vf * (1.0 - density / kj)


def _greenshields_flow(density: np.ndarray, vf: float, kj: float) -> np.ndarray:
    """Greenshields flow–density: q = vf × k × (1 - k/kj)."""
    return vf * density * (1.0 - density / kj)


def fit_greenshields_model(mfd_data: pd.DataFrame) -> dict:
    """
    Fit the Greenshields model to baseline data using scipy curve_fit.

    The Greenshields model is linear in speed–density space:
        v(k) = v_f × (1 - k / k_j)

    Parameters:
        v_f  = free-flow speed (intercept, km/h)
        k_j  = jam density (x-intercept, veh/km)
        k_c  = critical density = k_j / 2
        q_max = capacity = v_f × k_j / 4

    Args:
        mfd_data: DataFrame from extract_mfd_data(). Only baseline rows are used.

    Returns:
        Dict with model parameters and R-squared.
    """
    baseline = mfd_data[mfd_data["scenario_type"] == "baseline"]
    if baseline.empty:
        logger.warning("No baseline data for Greenshields fit")
        return {"error": "no_baseline_data"}

    k = baseline["density_veh_per_km"].values
    v = baseline["speed_kmh"].values

    # Remove zero/negative values.
    mask = (k > 0) & (v > 0)
    k, v = k[mask], v[mask]

    if len(k) < 5:
        logger.warning("Too few data points (%d) for Greenshields fit", len(k))
        return {"error": "insufficient_data", "n_points": len(k)}

    try:
        # Initial guesses: vf = max speed, kj = 2 × max density.
        p0 = [float(np.max(v)), float(np.max(k) * 2)]
        bounds = ([0, 0], [200, 500])  # Reasonable upper limits.
        popt, _ = curve_fit(_greenshields_speed, k, v, p0=p0, bounds=bounds, maxfev=5000)
        vf, kj = float(popt[0]), float(popt[1])
    except RuntimeError as exc:
        logger.warning("Greenshields curve_fit failed: %s", exc)
        return {"error": str(exc)}

    # R-squared.
    v_pred = _greenshields_speed(k, vf, kj)
    ss_res = float(np.sum((v - v_pred) ** 2))
    ss_tot = float(np.sum((v - np.mean(v)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    critical_density = kj / 2.0
    capacity = vf * kj / 4.0

    result = {
        "free_flow_speed_kmh": round(vf, 2),
        "jam_density_veh_per_km": round(kj, 2),
        "critical_density_veh_per_km": round(critical_density, 2),
        "capacity_veh_per_hour": round(capacity, 1),
        "r_squared": round(r_squared, 4),
    }
    logger.info(
        "Greenshields fit: vf=%.1f km/h, kj=%.1f veh/km, capacity=%.0f veh/h, R²=%.3f",
        vf,
        kj,
        capacity,
        r_squared,
    )
    return result


# ---------------------------------------------------------------------------
# MFD plotting
# ---------------------------------------------------------------------------


def plot_mfd_density_flow(mfd_data: pd.DataFrame, output_dir: str) -> str:
    """Plot Macroscopic Fundamental Diagram: density vs flow."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    for stype, color in SCENARIO_COLORS.items():
        subset = mfd_data[mfd_data["scenario_type"] == stype]
        if subset.empty:
            continue
        ax.scatter(
            subset["density_veh_per_km"],
            subset["flow_veh_per_hour"],
            alpha=0.4,
            s=10,
            color=color,
            label=stype.replace("_", " ").title(),
        )

    ax.set_xlabel("Network Density (veh/km)", fontsize=12)
    ax.set_ylabel("Network Flow (veh/hour)", fontsize=12)
    ax.set_title("Macroscopic Fundamental Diagram — Density vs Flow", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    path = os.path.join(output_dir, "mfd_density_flow.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved MFD density-flow plot → %s", path)
    return path


def plot_mfd_density_speed(mfd_data: pd.DataFrame, output_dir: str) -> str:
    """Plot density vs space-mean speed."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    for stype, color in SCENARIO_COLORS.items():
        subset = mfd_data[mfd_data["scenario_type"] == stype]
        if subset.empty:
            continue
        ax.scatter(
            subset["density_veh_per_km"],
            subset["speed_kmh"],
            alpha=0.4,
            s=10,
            color=color,
            label=stype.replace("_", " ").title(),
        )

    ax.set_xlabel("Network Density (veh/km)", fontsize=12)
    ax.set_ylabel("Space-Mean Speed (km/h)", fontsize=12)
    ax.set_title("Macroscopic Fundamental Diagram — Density vs Speed", fontsize=14)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    path = os.path.join(output_dir, "mfd_density_speed.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved MFD density-speed plot → %s", path)
    return path


def plot_speed_comparison(mfd_data: pd.DataFrame, output_dir: str) -> str:
    """Bar chart comparing mean speed across demand levels for baseline vs incident."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    summary = mfd_data.groupby(["period", "scenario_type"])["speed_kmh"].mean().reset_index()

    stypes = summary["scenario_type"].unique()
    periods = sorted(summary["period"].unique())
    x = np.arange(len(periods))
    width = 0.8 / max(len(stypes), 1)

    for i, stype in enumerate(stypes):
        sub = summary[summary["scenario_type"] == stype]
        vals = [
            float(sub[sub["period"] == p]["speed_kmh"].values[0])
            if len(sub[sub["period"] == p]) > 0
            else 0
            for p in periods
        ]
        color = SCENARIO_COLORS.get(stype, "#999999")
        ax.bar(x + i * width, vals, width, label=stype.replace("_", " ").title(), color=color)

    ax.set_xlabel("Demand Level (period)", fontsize=12)
    ax.set_ylabel("Mean Speed (km/h)", fontsize=12)
    ax.set_title("Mean Speed by Demand Level and Scenario Type", fontsize=14)
    ax.set_xticks(x + width * (len(stypes) - 1) / 2)
    ax.set_xticklabels([f"p={p}" for p in periods])
    ax.legend(fontsize=8)

    path = os.path.join(output_dir, "speed_comparison.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved speed comparison plot → %s", path)
    return path


def plot_throughput_comparison(mfd_data: pd.DataFrame, output_dir: str) -> str:
    """Bar chart comparing throughput across demand levels."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    summary = (
        mfd_data.groupby(["period", "scenario_type"])["flow_veh_per_hour"].mean().reset_index()
    )

    stypes = summary["scenario_type"].unique()
    periods = sorted(summary["period"].unique())
    x = np.arange(len(periods))
    width = 0.8 / max(len(stypes), 1)

    for i, stype in enumerate(stypes):
        sub = summary[summary["scenario_type"] == stype]
        vals = [
            float(sub[sub["period"] == p]["flow_veh_per_hour"].values[0])
            if len(sub[sub["period"] == p]) > 0
            else 0
            for p in periods
        ]
        color = SCENARIO_COLORS.get(stype, "#999999")
        ax.bar(x + i * width, vals, width, label=stype.replace("_", " ").title(), color=color)

    ax.set_xlabel("Demand Level (period)", fontsize=12)
    ax.set_ylabel("Throughput (veh/hour)", fontsize=12)
    ax.set_title("Throughput by Demand Level and Scenario Type", fontsize=14)
    ax.set_xticks(x + width * (len(stypes) - 1) / 2)
    ax.set_xticklabels([f"p={p}" for p in periods])
    ax.legend(fontsize=8)

    path = os.path.join(output_dir, "throughput_comparison.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved throughput comparison plot → %s", path)
    return path


def plot_ai_distribution(all_results: list[dict], output_dir: str) -> str:
    """Plot distribution of Antifragility Index across all incident runs."""
    ai_values = []
    labels = []
    for r in all_results:
        if r.get("status") != "success":
            continue
        summary = r.get("summary", {})
        ai = summary.get("antifragility_index")
        if ai is not None and summary.get("base_probability", -1) > 0:
            ai_values.append(float(ai))
            labels.append(r.get("scenario_id", ""))

    if not ai_values:
        logger.warning("No AI values to plot")
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.text(
            0.5,
            0.5,
            "No measurable events",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        path = os.path.join(output_dir, "ai_distribution.png")
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        return path

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in ai_values]
    ax.bar(range(len(ai_values)), ai_values, color=colors)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(
        y=0.05,
        color="#2ecc71",
        linewidth=0.8,
        linestyle="--",
        alpha=0.6,
        label="Antifragile threshold",
    )
    ax.axhline(
        y=-0.05,
        color="#f39c12",
        linewidth=0.8,
        linestyle="--",
        alpha=0.6,
        label="Fragile threshold",
    )
    ax.axhline(
        y=-0.20,
        color="#e74c3c",
        linewidth=0.8,
        linestyle="--",
        alpha=0.6,
        label="Brittle threshold",
    )
    ax.set_xlabel("Scenario Run", fontsize=12)
    ax.set_ylabel("Antifragility Index", fontsize=12)
    ax.set_title("Antifragility Index Distribution Across Incident Scenarios", fontsize=14)
    ax.legend(fontsize=9)

    path = os.path.join(output_dir, "ai_distribution.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved AI distribution plot → %s", path)
    return path


def plot_resilience_components(score: ResilienceScore, output_dir: str) -> str:
    """Horizontal bar chart of resilience score components."""
    fig, ax = plt.subplots(figsize=(10, 5))

    components = {
        "Speed Resilience": score.speed_resilience,
        "Throughput Resilience": score.throughput_resilience,
        "Recovery Resilience": score.recovery_resilience,
        "Robustness": score.robustness,
    }
    labels = list(components.keys())
    values = list(components.values())
    colors = ["#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, height=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score (0–1)", fontsize=12)
    ax.set_title(
        f"Resilience Score Components — Overall: {score.overall_score:.2f} ({score.grade})",
        fontsize=14,
    )

    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            fontsize=10,
        )

    ax.axvline(
        x=score.overall_score,
        color="black",
        linewidth=1.5,
        linestyle="--",
        alpha=0.7,
        label=f"Overall: {score.overall_score:.2f}",
    )
    ax.legend(fontsize=9)

    path = os.path.join(output_dir, "resilience_components.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved resilience components plot → %s", path)
    return path


# ---------------------------------------------------------------------------
# Weak point analysis
# ---------------------------------------------------------------------------


def _collect_accident_reports(scenarios: list[Scenario]) -> list[dict]:
    """Load all accident_reports.json from incident scenarios."""
    all_reports: list[dict] = []
    for s in scenarios:
        if s.scenario_type == "baseline":
            continue
        path = os.path.join(s.output_folder, "accident_reports.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            reports = json.load(f)
        for r in reports:
            r["_scenario_id"] = s.scenario_id
            r["_period"] = s.period
        all_reports.extend(reports)
    return all_reports


def _compute_edge_traffic_importance(scenarios: list[Scenario]) -> dict[str, float]:
    """
    Proxy for edge importance based on baseline traffic volume.

    Counts how many accidents occurred per edge across all baseline runs.
    For baseline runs (no accidents), we use the vehicle_count metric as a
    proxy—edges with more traffic are topologically more important.

    Falls back to uniform importance if no baseline accident data is available.
    """
    # Use ALL accident reports to count per-edge occurrences.
    edge_counts: dict[str, int] = {}
    total = 0
    for s in scenarios:
        path = os.path.join(s.output_folder, "accident_reports.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            reports = json.load(f)
        for r in reports:
            eid = r.get("location", {}).get("edge_id", "")
            if eid:
                edge_counts[eid] = edge_counts.get(eid, 0) + 1
                total += 1

    if total == 0:
        return {}

    # Normalize to [0, 1].
    max_count = max(edge_counts.values())
    return {eid: count / max_count for eid, count in edge_counts.items()}


def compute_weak_points(
    scenarios: list[Scenario],
    top_n: int = 10,
) -> list[EdgeVulnerability]:
    """
    Identify the top-N most vulnerable edges in the network.

    Algorithm:
      1. Group accidents by edge_id across all incident scenario runs.
      2. For each edge: count accidents, mean duration, mean vehicles_affected.
      3. Compute speed_drop_ratio placeholder from impact metrics.
      4. Compute edge importance as traffic-volume proxy.
      5. vulnerability_index = (1 - speed_drop_ratio) × frequency × importance.
      6. Return top_n sorted by vulnerability_index descending.

    Args:
        scenarios: All scenarios (baseline + incident).
        top_n:     Number of weak points to return.

    Returns:
        List of EdgeVulnerability in descending vulnerability order.
    """
    all_reports = _collect_accident_reports(scenarios)
    if not all_reports:
        logger.warning("No accident reports found for weak point analysis")
        return []

    edge_importance = _compute_edge_traffic_importance(scenarios)

    # Group accidents by edge_id.
    edge_data: dict[str, list[dict]] = {}
    for r in all_reports:
        eid = r.get("location", {}).get("edge_id", "unknown")
        edge_data.setdefault(eid, []).append(r)

    # Count total incident scenario runs for frequency normalisation.
    n_incident_runs = sum(1 for s in scenarios if s.scenario_type != "baseline")
    n_incident_runs = max(n_incident_runs, 1)

    vulnerabilities: list[EdgeVulnerability] = []
    for eid, reports in edge_data.items():
        count = len(reports)
        durations = [r.get("duration_seconds", 0) for r in reports]
        affected = [r.get("impact", {}).get("vehicles_affected", 0) for r in reports]

        mean_dur = sum(durations) / max(len(durations), 1)
        mean_aff = sum(affected) / max(len(affected), 1)

        # Speed drop ratio: use vehicles_affected as a proxy for disruption severity.
        # Higher affected count → more disruption → lower ratio.
        max_expected_affected = 20.0
        speed_drop_ratio = max(0.0, 1.0 - mean_aff / max_expected_affected)

        importance = edge_importance.get(eid, 0.5)
        frequency = count / n_incident_runs

        vuln_idx = (1.0 - speed_drop_ratio) * frequency * importance

        # Get representative coordinates.
        loc = reports[0].get("location", {})
        x = loc.get("x", 0.0)
        y = loc.get("y", 0.0)

        vulnerabilities.append(
            EdgeVulnerability(
                edge_id=eid,
                x=x,
                y=y,
                accident_count=count,
                mean_duration_seconds=round(mean_dur, 1),
                mean_vehicles_affected=round(mean_aff, 1),
                mean_speed_drop_ratio=round(speed_drop_ratio, 3),
                edge_importance=round(importance, 3),
                vulnerability_index=round(vuln_idx, 4),
            )
        )

    # Sort by vulnerability index descending.
    vulnerabilities.sort(key=lambda v: v.vulnerability_index, reverse=True)

    top = vulnerabilities[:top_n]
    logger.info(
        "Identified %d vulnerable edges (top %d returned from %d total)",
        len(vulnerabilities),
        len(top),
        len(edge_data),
    )
    return top


def plot_weak_point_map(
    weak_points: list[EdgeVulnerability],
    all_accidents: list[dict],
    output_dir: str,
) -> str:
    """Plot a 2D scatter of accident locations with vulnerability overlay."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    # Background: all accident locations.
    if all_accidents:
        xs = [a.get("location", {}).get("x", 0) for a in all_accidents]
        ys = [a.get("location", {}).get("y", 0) for a in all_accidents]
        ax.scatter(xs, ys, s=5, alpha=0.15, color="#bbbbbb", label="All accidents")

    # Overlay: weak points.
    if weak_points:
        wp_x = [wp.x for wp in weak_points]
        wp_y = [wp.y for wp in weak_points]
        wp_sizes = [max(50, wp.vulnerability_index * 2000) for wp in weak_points]
        wp_colors = [wp.vulnerability_index for wp in weak_points]

        scatter = ax.scatter(
            wp_x,
            wp_y,
            s=wp_sizes,
            c=wp_colors,
            cmap="YlOrRd",
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
            label="Weak points",
        )
        plt.colorbar(scatter, ax=ax, label="Vulnerability Index", shrink=0.8)

        # Label top 5.
        for i, wp in enumerate(weak_points[:5]):
            ax.annotate(
                f"#{i + 1}",
                (wp.x, wp.y),
                fontsize=8,
                fontweight="bold",
                xytext=(5, 5),
                textcoords="offset points",
            )

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title("Network Vulnerability Map — Top Weak Points", fontsize=14)
    ax.legend(fontsize=9)

    path = os.path.join(output_dir, "weak_point_map.png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved weak point map → %s", path)
    return path


# ---------------------------------------------------------------------------
# Composite resilience score
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def interpret_resilience_score(r: float) -> tuple[str, str]:
    """
    Return (grade, interpretation) based on the composite resilience score.

    Grading scale:
        A+  (0.90–1.00): Exceptional resilience
        A   (0.80–0.90): Strong resilience
        B   (0.70–0.80): Good resilience
        C   (0.60–0.70): Adequate resilience
        D   (0.50–0.60): Marginal resilience
        F   (< 0.50):    Poor resilience — significant interventions needed
    """
    if r >= 0.90:
        return "A+", "Exceptional resilience — network handles disruptions with minimal impact"
    elif r >= 0.80:
        return "A", "Strong resilience — network recovers well from incidents"
    elif r >= 0.70:
        return "B", "Good resilience — network generally maintains performance"
    elif r >= 0.60:
        return "C", "Adequate resilience — some vulnerability to disruptions"
    elif r >= 0.50:
        return "D", "Marginal resilience — significant performance degradation under stress"
    else:
        return "F", "Poor resilience — network requires significant intervention"


def compute_resilience_score(
    mfd_data: pd.DataFrame,
    all_results: list[dict],
    scenarios: list[Scenario],
    weak_points: list[EdgeVulnerability],
    mfd_params: dict,
    weights: dict[str, float] | None = None,
) -> ResilienceScore:
    """
    Compute the composite resilience score from all simulation results.

    Components:
        R_speed       (0.30): Speed maintenance under incidents vs baseline.
        R_throughput   (0.25): Throughput maintenance under incidents vs baseline.
        R_recovery     (0.25): Post-incident recovery (from Antifragility Index).
        R_robustness   (0.20): Tolerance to increasing incident rates.

    Args:
        mfd_data:     DataFrame from extract_mfd_data().
        all_results:  List of result dicts from parallel execution.
        scenarios:    Scenario objects matching all_results order.
        weak_points:  Output from compute_weak_points().
        mfd_params:   Output from fit_greenshields_model().
        weights:      Optional weight overrides {speed, throughput, recovery, robustness}.

    Returns:
        ResilienceScore with all components and grading.
    """
    if weights is None:
        weights = {"speed": 0.30, "throughput": 0.25, "recovery": 0.25, "robustness": 0.20}

    # Identify the default demand level (period=1.0, or the middle one).
    available_periods = sorted(mfd_data["period"].unique()) if not mfd_data.empty else [1.0]
    default_period = (
        1.0 if 1.0 in available_periods else available_periods[len(available_periods) // 2]
    )

    default_data = (
        mfd_data[mfd_data["period"] == default_period] if not mfd_data.empty else pd.DataFrame()
    )

    # ── R_speed ──
    if not default_data.empty:
        baseline_speed = default_data[default_data["scenario_type"] == "baseline"][
            "speed_kmh"
        ].mean()
        incident_speed = default_data[default_data["scenario_type"] != "baseline"][
            "speed_kmh"
        ].mean()
        r_speed = _clamp01(incident_speed / baseline_speed) if baseline_speed > 0 else 0.5
    else:
        r_speed = 0.5

    # ── R_throughput ──
    if not default_data.empty:
        baseline_tp = default_data[default_data["scenario_type"] == "baseline"][
            "flow_veh_per_hour"
        ].mean()
        incident_tp = default_data[default_data["scenario_type"] != "baseline"][
            "flow_veh_per_hour"
        ].mean()
        r_throughput = _clamp01(incident_tp / baseline_tp) if baseline_tp > 0 else 0.5
    else:
        r_throughput = 0.5

    # ── R_recovery (from AI) ──
    ai_values = []
    for i, r in enumerate(all_results):
        if r.get("status") != "success":
            continue
        summary = r.get("summary", {})
        ai = summary.get("antifragility_index")
        # Only include incident runs (base_probability > 0).
        if i < len(scenarios) and scenarios[i].base_probability > 0 and ai is not None:
            ai_values.append(float(ai))

    ai_aggregate = float(np.mean(ai_values)) if ai_values else 0.0
    # Map AI to [0, 1]: AI=-0.20→0.0, AI=0.0→0.8, AI=0.05→1.0.
    r_recovery = _clamp01((ai_aggregate + 0.20) / 0.25)

    # ── R_robustness ──
    # Fit speed vs base_probability at default demand level.
    if not default_data.empty:
        speed_by_prob: dict[float, float] = {}
        for i, s in enumerate(scenarios):
            if s.period != default_period or i >= len(all_results):
                continue
            r = all_results[i]
            if r.get("status") != "success":
                continue
            prob = s.base_probability
            mean_speed = r.get("summary", {}).get("mean_speed_kmh")
            if mean_speed is None:
                # Fall back to MFD data.
                sub = default_data[
                    (default_data["scenario_type"] == s.scenario_type)
                    & (default_data["seed"] == s.seed)
                ]
                mean_speed = float(sub["speed_kmh"].mean()) if not sub.empty else None
            if mean_speed is not None:
                speed_by_prob.setdefault(prob, []).append(mean_speed) if isinstance(
                    speed_by_prob.get(prob), list
                ) else None
                if prob not in speed_by_prob:
                    speed_by_prob[prob] = [mean_speed]
                elif isinstance(speed_by_prob[prob], list):
                    speed_by_prob[prob].append(mean_speed)

        # Average speeds per probability level.
        prob_vals = []
        speed_vals = []
        for prob, speeds in speed_by_prob.items():
            if isinstance(speeds, list):
                prob_vals.append(prob)
                speed_vals.append(np.mean(speeds))

        if len(prob_vals) >= 2:
            # Linear regression: speed = a + b × probability.
            coeffs = np.polyfit(prob_vals, speed_vals, 1)
            slope = abs(float(coeffs[0]))
            # Max expected slope: a drop of 10 km/h per unit probability.
            max_slope = 10000.0  # km/h per probability unit (very sensitive).
            r_robustness = _clamp01(1.0 - slope / max_slope)
        else:
            r_robustness = 0.5
    else:
        r_robustness = 0.5

    # ── Composite score ──
    overall = (
        weights["speed"] * r_speed
        + weights["throughput"] * r_throughput
        + weights["recovery"] * r_recovery
        + weights["robustness"] * r_robustness
    )
    overall = round(overall, 4)
    grade, interpretation = interpret_resilience_score(overall)

    score = ResilienceScore(
        overall_score=overall,
        grade=grade,
        interpretation=interpretation,
        speed_resilience=round(r_speed, 4),
        throughput_resilience=round(r_throughput, 4),
        recovery_resilience=round(r_recovery, 4),
        robustness=round(r_robustness, 4),
        weights=weights,
        mfd_parameters=mfd_params,
        ai_aggregate=round(ai_aggregate, 4),
        weak_points=weak_points,
    )

    logger.info(
        "Resilience score: %.2f (%s) — speed=%.2f, throughput=%.2f, recovery=%.2f, robustness=%.2f",
        overall,
        grade,
        r_speed,
        r_throughput,
        r_recovery,
        r_robustness,
    )
    return score


def score_to_dict(score: ResilienceScore) -> dict:
    """Serialize a ResilienceScore to a JSON-safe dict."""
    d = asdict(score)
    return d
