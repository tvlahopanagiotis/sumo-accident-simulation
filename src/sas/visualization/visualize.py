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
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
LIVE_PROGRESS_FIGSIZE = (15, 11)


def _supports_interactive_backend() -> bool:
    """Return True when the active Matplotlib backend can show live windows."""
    backend = plt.get_backend().lower()
    noninteractive_markers = ("agg", "cairo", "pdf", "pgf", "ps", "svg", "template", "inline")
    return not any(marker in backend for marker in noninteractive_markers)


class LiveProgressDashboard:
    """
    Live in-run dashboard for headless SUMO executions.

    The dashboard updates from in-memory metrics snapshots and always refreshes
    ``live_progress.png`` inside the run output folder. When the current
    Matplotlib backend supports it, it also opens an interactive window.
    """

    def __init__(
        self,
        output_dir: str,
        total_steps: int,
        refresh_interval_steps: int = 300,
        prefer_window: bool = True,
        net_xml_path: str | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.total_steps = max(int(total_steps), 1)
        self.refresh_interval_steps = max(int(refresh_interval_steps), 1)
        self.output_path = os.path.join(output_dir, "live_progress.png")
        self._last_render_step = -self.refresh_interval_steps
        self._interactive = bool(prefer_window and _supports_interactive_backend())
        self._network_shapes = _load_network_edge_segments(net_xml_path) if net_xml_path else {}
        self._network_colorbar = None

        self._fig = plt.figure(figsize=LIVE_PROGRESS_FIGSIZE)
        grid = self._fig.add_gridspec(
            4,
            2,
            width_ratios=[1.25, 1.0],
            height_ratios=[1.0, 1.0, 1.0, 1.0],
            hspace=0.32,
            wspace=0.18,
        )
        self._ax_network = self._fig.add_subplot(grid[:, 0])
        self._ax_occ = self._fig.add_subplot(grid[0, 1])
        self._ax_speed = self._fig.add_subplot(grid[1, 1])
        self._ax_flow = self._fig.add_subplot(grid[2, 1])
        self._ax_acc = self._fig.add_subplot(grid[3, 1])
        self._ax_network_cbar = inset_axes(
            self._ax_network,
            width="3.5%",
            height="42%",
            loc="lower right",
            borderpad=1.8,
        )
        self._ax_network_cbar.set_visible(False)

        if self._interactive:
            try:
                plt.ion()
                self._fig.canvas.manager.set_window_title("SAS Live Progress")
                self._fig.show()
            except Exception as exc:
                logger.info("Live progress window unavailable; continuing with PNG only: %s", exc)
                self._interactive = False

    def update(
        self,
        snapshots: Sequence[Any],
        current_step: int,
        active_accident_count: int,
        resolved_accidents: int,
        total_accidents: int,
        edge_vehicle_counts: Mapping[str, int] | None = None,
        accident_points: Sequence[Mapping[str, Any]] | None = None,
        resolved_accident_points: Sequence[Mapping[str, Any]] | None = None,
        *,
        force: bool = False,
    ) -> None:
        """Redraw the dashboard from the accumulated network snapshots."""
        if not snapshots and not self._network_shapes:
            return
        if not force and (current_step - self._last_render_step) < self.refresh_interval_steps:
            return

        self._last_render_step = current_step

        times_min = [float(s.timestamp_seconds) / 60.0 for s in snapshots] if snapshots else []
        vehicle_counts = [float(s.vehicle_count) for s in snapshots] if snapshots else []
        mean_speeds = [float(s.mean_speed_kmh) for s in snapshots] if snapshots else []
        throughputs = [float(s.throughput_per_hour) for s in snapshots] if snapshots else []
        active_series = [float(s.active_accidents) for s in snapshots] if snapshots else []

        self._draw_network_load_map(
            edge_vehicle_counts or {},
            accident_points or [],
            resolved_accident_points or [],
        )

        ax_occ, ax_speed, ax_flow, ax_acc = (
            self._ax_occ,
            self._ax_speed,
            self._ax_flow,
            self._ax_acc,
        )
        for ax in (ax_occ, ax_speed, ax_flow, ax_acc):
            ax.clear()
            ax.grid(True, alpha=0.25)
            ax.set_xlim(left=0.0, right=max(times_min[-1], 1.0) if times_min else 1.0)

        self._draw_series_panel(
            ax_occ,
            times_min,
            vehicle_counts,
            "Active Vehicles On Network",
            "Simultaneous vehicles",
            "#1f77b4",
        )
        self._draw_series_panel(
            ax_speed,
            times_min,
            mean_speeds,
            "Mean Network Speed",
            "Average km/h across active vehicles",
            "#d62728",
        )
        self._draw_series_panel(
            ax_flow,
            times_min,
            throughputs,
            "Completed Trips Rate",
            "Vehicles/hour reaching destination",
            "#2ca02c",
        )
        ax_flow.set_xlabel("Simulation Time (min)")

        if active_series:
            ax_acc.step(times_min, active_series, where="post", color="#9467bd", linewidth=2.4)
            ax_acc.fill_between(
                times_min,
                active_series,
                step="post",
                alpha=0.15,
                color="#9467bd",
            )
        else:
            ax_acc.text(
                0.5,
                0.5,
                "Waiting for accident snapshots",
                transform=ax_acc.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#666666",
            )
        ax_acc.set_title("Concurrent Active Accidents", fontweight="bold")
        ax_acc.set_xlabel("Simulation Time (min)")
        ax_acc.set_ylabel("Count")
        ax_acc.set_ylim(bottom=0.0)

        pct_complete = min(100.0, current_step * 100.0 / self.total_steps)
        title = (
            f"SAS Live Progress  |  {pct_complete:5.1f}% complete"
            f"  |  sim {current_step}/{self.total_steps} s"
            f"  |  accidents {total_accidents} total / {active_accident_count} active /"
            f" {resolved_accidents} resolved"
        )
        self._fig.suptitle(title, fontsize=14, fontweight="bold")
        self._fig.savefig(self.output_path, dpi=FIGURE_DPI, bbox_inches="tight")

        if self._interactive:
            try:
                self._fig.canvas.draw_idle()
                self._fig.canvas.flush_events()
                plt.pause(0.001)
            except Exception as exc:
                logger.info("Live progress window update failed; continuing with PNG only: %s", exc)
                self._interactive = False

    def _draw_series_panel(
        self,
        ax: Any,
        x_values: Sequence[float],
        y_values: Sequence[float],
        title: str,
        y_label: str,
        color: str,
    ) -> None:
        """Draw a live time-series panel or a waiting placeholder."""
        if y_values:
            ax.plot(x_values, y_values, color=color, linewidth=2.4)
            ax.fill_between(x_values, y_values, alpha=0.15, color=color)
        else:
            ax.text(
                0.5,
                0.5,
                "Waiting for first metrics snapshot",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#666666",
            )
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(y_label)

    def _draw_network_load_map(
        self,
        edge_vehicle_counts: Mapping[str, int],
        accident_points: Sequence[Mapping[str, Any]],
        resolved_accident_points: Sequence[Mapping[str, Any]],
    ) -> None:
        """Draw the current per-edge vehicle load over the network geometry."""
        ax = self._ax_network
        ax.clear()
        self._ax_network_cbar.cla()
        self._ax_network_cbar.set_visible(False)
        self._network_colorbar = None

        ax.set_title("Live Network Load And Active Accidents", fontweight="bold")
        ax.set_xlabel("X coordinate (m)")
        ax.set_ylabel("Y coordinate (m)")
        ax.grid(False)
        ax.set_aspect("equal", adjustable="box")

        if not self._network_shapes:
            ax.text(
                0.5,
                0.5,
                "Network geometry unavailable for live map",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="#666666",
            )
            return

        background = LineCollection(
            list(self._network_shapes.values()),
            colors="#d0d6de",
            linewidths=0.55,
            alpha=0.45,
            zorder=1,
        )
        ax.add_collection(background)

        loaded_segments = []
        loaded_counts = []
        for edge_id, count in edge_vehicle_counts.items():
            segment = self._network_shapes.get(edge_id)
            if segment is None or count <= 0:
                continue
            loaded_segments.append(segment)
            loaded_counts.append(float(count))

        if loaded_segments:
            vmax = max(1.0, max(loaded_counts))
            widths = [1.1 + 3.2 * (count / vmax) for count in loaded_counts]
            loads = LineCollection(
                loaded_segments,
                cmap=plt.cm.YlOrRd,
                norm=Normalize(vmin=0.0, vmax=vmax),
                linewidths=widths,
                alpha=0.92,
                zorder=3,
            )
            loads.set_array(np.array(loaded_counts))
            ax.add_collection(loads)
            active_edges = len(loaded_segments)
            max_edge_load = int(max(loaded_counts))
            self._network_colorbar = self._fig.colorbar(
                loads,
                cax=self._ax_network_cbar,
            )
            self._ax_network_cbar.set_visible(True)
            self._network_colorbar.set_label("Vehicles currently on edge", fontsize=10)
        else:
            active_edges = 0
            max_edge_load = 0

        severity_sizes = {"MINOR": 60, "MODERATE": 95, "MAJOR": 135, "CRITICAL": 185}
        accident_handles: list[Line2D] = []

        if resolved_accident_points:
            xs_hist = [float(acc.get("x", 0.0)) for acc in resolved_accident_points]
            ys_hist = [float(acc.get("y", 0.0)) for acc in resolved_accident_points]
            ax.scatter(
                xs_hist,
                ys_hist,
                s=42,
                marker="x",
                c="#6b7280",
                linewidths=1.4,
                alpha=0.75,
                zorder=4,
            )

        if accident_points:
            for severity in SEVERITY_ORDER:
                points = [acc for acc in accident_points if str(acc.get("severity")) == severity]
                if not points:
                    continue
                xs = [float(acc.get("x", 0.0)) for acc in points]
                ys = [float(acc.get("y", 0.0)) for acc in points]
                ax.scatter(
                    xs,
                    ys,
                    s=severity_sizes.get(severity, 75),
                    c=SEVERITY_COLORS.get(severity, "#95a5a6"),
                    edgecolors="black",
                    linewidths=1.4,
                    alpha=0.95,
                    zorder=5,
                )
                accident_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="none",
                        label=f"{severity} accident",
                        markerfacecolor=SEVERITY_COLORS.get(severity, "#95a5a6"),
                        markeredgecolor="black",
                        markersize=8,
                    )
                )

        road_handles = [
            Line2D([0], [0], color="#d0d6de", linewidth=2.0, label="No vehicles on edge"),
            Line2D([0], [0], color=plt.cm.YlOrRd(0.82), linewidth=3.0, label="Warmer road = heavier current load"),
            Line2D(
                [0],
                [0],
                marker="x",
                color="#6b7280",
                linewidth=0,
                label="Resolved accident",
                markersize=8,
                markeredgewidth=1.4,
            ),
        ]
        legend_handles = road_handles + accident_handles
        ax.legend(
            handles=legend_handles,
            loc="lower left",
            fontsize=9,
            framealpha=0.92,
            title="Map legend",
            title_fontsize=10,
        )

        ax.autoscale_view()
        ax.text(
            0.01,
            0.99,
            (
                f"loaded edges: {active_edges:,} / {len(self._network_shapes):,}\n"
                f"max vehicles on one edge: {max_edge_load}\n"
                f"active accidents shown: {len(accident_points)}\n"
                f"resolved accidents shown: {len(resolved_accident_points)}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

    def close(self) -> None:
        """Close the Matplotlib figure backing the dashboard."""
        plt.close(self._fig)


def _parse_shape_points(shape: str) -> list[tuple[float, float]]:
    """Convert SUMO shape string 'x1,y1 x2,y2 ...' to point tuples."""
    pts: list[tuple[float, float]] = []
    for raw in shape.split():
        if "," not in raw:
            continue
        try:
            x_str, y_str = raw.split(",", 1)
            pts.append((float(x_str), float(y_str)))
        except ValueError:
            continue
    return pts


def _resolve_net_file_from_metadata(output_dir: str) -> str | None:
    """
    Resolve the SUMO net.xml path for a run using output_dir/metadata.json.
    """
    metadata_path = os.path.join(output_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return None

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        sumo_cfg = metadata.get("config", {}).get("sumo", {}).get("config_file")
        if not sumo_cfg:
            return None
        sumo_cfg = str(sumo_cfg)
        if not os.path.isabs(sumo_cfg):
            sumo_cfg = os.path.normpath(os.path.join(output_dir, sumo_cfg))
        if not os.path.exists(sumo_cfg):
            return None

        tree = ET.parse(sumo_cfg)
        root = tree.getroot()
        input_elem = root.find("input")
        if input_elem is None:
            return None
        net_elem = input_elem.find("net-file")
        if net_elem is None:
            return None
        net_value = net_elem.attrib.get("value")
        if not net_value:
            return None
        if os.path.isabs(net_value):
            net_path = net_value
        else:
            net_path = os.path.normpath(os.path.join(os.path.dirname(sumo_cfg), net_value))
        return net_path if os.path.exists(net_path) else None
    except Exception as exc:
        logger.debug("Could not resolve net file from metadata: %s", exc)
        return None


def _resolve_net_file_from_sumocfg(sumocfg_path: str) -> str | None:
    """
    Resolve the SUMO net.xml path directly from a .sumocfg path.
    """
    if not sumocfg_path:
        return None

    try:
        cfg_path = os.path.abspath(sumocfg_path)
        if not os.path.exists(cfg_path):
            return None
        tree = ET.parse(cfg_path)
        root = tree.getroot()
        input_elem = root.find("input")
        if input_elem is None:
            return None
        net_elem = input_elem.find("net-file")
        if net_elem is None:
            return None
        net_value = net_elem.attrib.get("value")
        if not net_value:
            return None
        if os.path.isabs(net_value):
            net_path = net_value
        else:
            net_path = os.path.normpath(os.path.join(os.path.dirname(cfg_path), net_value))
        return net_path if os.path.exists(net_path) else None
    except Exception as exc:
        logger.debug("Could not resolve net file from sumocfg '%s': %s", sumocfg_path, exc)
        return None


def resolve_net_file(output_dir: str | None = None, sumocfg_path: str | None = None) -> str | None:
    """
    Resolve the network file from either metadata or a direct .sumocfg path.
    """
    if output_dir:
        net_path = _resolve_net_file_from_metadata(output_dir)
        if net_path:
            return net_path
    if sumocfg_path:
        return _resolve_net_file_from_sumocfg(sumocfg_path)
    return None


def _load_network_edge_segments(net_xml_path: str) -> dict[str, list[tuple[float, float]]]:
    """
    Load road geometry from SUMO .net.xml keyed by edge id.
    """
    segments: dict[str, list[tuple[float, float]]] = {}
    try:
        tree = ET.parse(net_xml_path)
        root = tree.getroot()
        for edge in root.findall("edge"):
            edge_id = edge.attrib.get("id")
            if not edge_id or edge.attrib.get("function") == "internal":
                continue
            lane = edge.find("lane")
            if lane is None:
                continue
            shape = lane.attrib.get("shape")
            if not shape:
                continue
            pts = _parse_shape_points(shape)
            if len(pts) >= 2:
                segments[edge_id] = pts
    except Exception as exc:
        logger.debug("Failed to parse network file '%s': %s", net_xml_path, exc)
    return segments


def _load_network_segments(net_xml_path: str) -> list[list[tuple[float, float]]]:
    """
    Load road geometry from SUMO .net.xml as line segments for plotting.
    """
    return list(_load_network_edge_segments(net_xml_path).values())


def _metric_caption_html(image_file: str) -> tuple[str, str]:
    """
    Return a user-facing title and explanation for a generated report figure.
    """
    captions = {
        "metrics_timeseries.png": (
            "Network State Over Time",
            (
                "Top-level run summary. 'Active vehicles on network' means vehicles currently "
                "present in the simulation at each timestamp, not newly inserted vehicles. "
                "'Mean network speed' is the average speed across those active vehicles. "
                "'Completed trips rate' is the destination-completion throughput scaled to vehicles/hour."
            ),
        ),
        "severity_distribution.png": (
            "Accident Severity Mix",
            "Share of simulated accidents by severity tier across the full run.",
        ),
        "before_after_comparison.png": (
            "Network Speed Around Each Accident",
            (
                "For each accident, compares the network-wide mean speed in a short window before "
                "the accident trigger with the window immediately after it. Lower 'after' values "
                "mean the accident degraded overall traffic conditions."
            ),
        ),
        "accident_heatmap.png": (
            "Accident Locations",
            "Spatial distribution of accident events over the road network, colored by severity.",
        ),
    }
    return captions.get(image_file, (image_file, ""))


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

    # Plot 1: active vehicles currently present in the simulation
    axes[0].plot(
        df["hours"],
        df["vehicle_count"],
        linewidth=2.5,
        color="#3498db",
        label="Active vehicles currently on network",
    )
    axes[0].set_xlabel("Simulation Time (hours)", fontsize=11)
    axes[0].set_ylabel("Simultaneous Vehicles", fontsize=11)
    axes[0].set_title("Active Vehicles On Network", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=9)

    # Plot 2: Speed dynamics (smoothed median + p10/p90 envelope)
    speed = df["mean_speed_kmh"].astype(float)
    if len(df) > 1:
        sample_dt_s = float(df["timestamp_seconds"].diff().median())
        if np.isnan(sample_dt_s) or sample_dt_s <= 0:
            sample_dt_s = 60.0
    else:
        sample_dt_s = 60.0

    # Use a broad window to suppress minute-level stop-go oscillations.
    # Default target is ~15 minutes regardless of sampling cadence.
    window_points = max(5, int(round(900.0 / sample_dt_s)))
    if window_points % 2 == 0:
        window_points += 1

    roll = speed.rolling(window=window_points, center=True, min_periods=max(2, window_points // 3))
    speed_p10 = roll.quantile(0.10)
    speed_median = roll.quantile(0.50)
    speed_p90 = roll.quantile(0.90)

    # Secondary smoothing pass for cleaner band edges.
    post_window = max(3, window_points // 3)
    if post_window % 2 == 0:
        post_window += 1
    speed_p10 = speed_p10.rolling(
        window=post_window, center=True, min_periods=max(2, post_window // 2)
    ).mean()
    speed_median = speed_median.rolling(
        window=post_window, center=True, min_periods=max(2, post_window // 2)
    ).mean()
    speed_p90 = speed_p90.rolling(
        window=post_window, center=True, min_periods=max(2, post_window // 2)
    ).mean()

    speed_p10 = speed_p10.fillna(speed)
    speed_median = speed_median.fillna(speed)
    speed_p90 = speed_p90.fillna(speed)

    axes[1].fill_between(
        df["hours"],
        speed_p10,
        speed_p90,
        alpha=0.22,
        color="#e74c3c",
        label="p10-p90 band",
    )
    axes[1].plot(
        df["hours"],
        speed_median,
        linewidth=2.8,
        color="#e74c3c",
        label="Smoothed median speed (~15 min)",
    )
    axes[1].set_xlabel("Simulation Time (hours)", fontsize=11)
    axes[1].set_ylabel("Speed (km/h)", fontsize=11)
    axes[1].set_title("Mean Network Speed (Smoothed)", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=9)

    # Plot 3: Active accidents
    axes[2].bar(
        df["hours"],
        df["active_accidents"],
        width=0.02,
        color="#9b59b6",
        alpha=0.7,
        label="Concurrent active accidents",
    )
    axes[2].set_xlabel("Simulation Time (hours)", fontsize=11)
    axes[2].set_ylabel("Concurrent Accidents", fontsize=11)
    axes[2].set_title("Concurrent Active Accidents", fontsize=12, fontweight="bold")
    axes[2].grid(True, alpha=0.3, axis="y")
    axes[2].set_ylim(bottom=0)

    title = "Network State Over Time"
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
    Plot network-wide speed before and after each accident in a clearer paired view.

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

        groups = []
        labels = []
        colors = []
        if len(before_speeds) > 0:
            groups.append(before_speeds)
            labels.append("Before")
            colors.append("#2ecc71")
        if len(after_speeds) > 0:
            groups.append(after_speeds)
            labels.append("After")
            colors.append("#e74c3c")

        if groups:
            box = ax.boxplot(
                groups,
                patch_artist=True,
                widths=0.55,
                medianprops={"color": "black", "linewidth": 2},
                boxprops={"linewidth": 1.4},
                whiskerprops={"linewidth": 1.2},
                capprops={"linewidth": 1.2},
            )
            for patch, color in zip(box["boxes"], colors, strict=False):
                patch.set_facecolor(color)
                patch.set_alpha(0.65)
                patch.set_edgecolor("black")
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, fontsize=10)

            before_mean = float(np.mean(before_speeds)) if len(before_speeds) > 0 else np.nan
            after_mean = float(np.mean(after_speeds)) if len(after_speeds) > 0 else np.nan
            if not np.isnan(before_mean) and not np.isnan(after_mean):
                delta = after_mean - before_mean
                delta_label = f"{delta:+.1f} km/h"
                delta_color = "#c0392b" if delta < 0 else "#1e8449"
            else:
                delta_label = "insufficient data"
                delta_color = "#666666"
            ax.text(
                0.03,
                0.97,
                f"Mean change: {delta_label}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.5,
                color=delta_color,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No speed samples in the chosen windows",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#666666",
            )

        ax.set_ylabel("Network Mean Speed (km/h)" if idx % n_cols == 0 else "", fontsize=10)
        ax.set_title(
            f"{acc['accident_id']} at t={trigger_step}s",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="y")

    # Hide unused subplots
    for idx in range(n_accidents, len(axes)):
        axes[idx].set_visible(False)

    title = "Network Speed Around Each Accident"
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

    # Draw road network in background when metadata + net file are available.
    net_xml = _resolve_net_file_from_metadata(output_dir)
    if net_xml:
        segments = _load_network_segments(net_xml)
        if segments:
            roads = LineCollection(
                segments,
                colors="#8d99ae",
                linewidths=0.45,
                alpha=0.35,
                zorder=1,
            )
            ax.add_collection(roads)

    ax.scatter(
        xs,
        ys,
        c=colors_list,
        s=sizes,
        alpha=0.8,
        edgecolors="black",
        linewidth=1.5,
        zorder=3,
    )
    ax.set_aspect("equal", adjustable="box")

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
        "        .summary { background: #f8fafc; border: 1px solid #d9e2ec; padding: 18px; border-radius: 6px; margin: 18px 0; }",
        "        .summary p { margin: 8px 0; line-height: 1.5; }",
        "        .figure-card { background: #fbfcfe; border: 1px solid #d9e2ec; border-radius: 6px; padding: 16px; }",
        "        .figure-card h3 { margin-top: 0; color: #243b53; }",
        "        .figure-card p { color: #486581; line-height: 1.45; margin-bottom: 10px; }",
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

        metadata_rows = [
            ("run_id", metadata.get("run_id")),
            ("timestamp_utc", metadata.get("timestamp_utc")),
            ("sumo_version", metadata.get("sumo_version")),
            ("seed", metadata.get("seed")),
            ("config_file", metadata.get("config", {}).get("sumo", {}).get("config_file")),
            ("total_steps", metadata.get("summary", {}).get("steps_run")),
            ("total_accidents", metadata.get("summary", {}).get("total_accidents")),
        ]
        for key, val in metadata_rows:
            if val is None:
                continue
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

    html_parts.append("        <h2>How To Read This Report</h2>")
    html_parts.append("        <div class='summary'>")
    html_parts.append(
        "            <p><strong>Active vehicles on network</strong> counts vehicles currently present in the simulation at each timestamp. It is not the number of newly inserted vehicles.</p>"
    )
    html_parts.append(
        "            <p><strong>Mean network speed</strong> is the average speed across all active vehicles at that timestamp.</p>"
    )
    html_parts.append(
        "            <p><strong>Completed trips rate</strong> is throughput: how many vehicles finish their routes, scaled to vehicles per hour.</p>"
    )
    html_parts.append(
        "            <p><strong>Network speed around each accident</strong> compares the network-wide mean speed shortly before an accident with the window immediately after it, so you can see whether the accident slowed the whole network down.</p>"
    )
    html_parts.append("        </div>")

    # Visualizations
    html_parts.append("        <h2>Simulation Results</h2>")
    html_parts.append("        <div class='image-grid'>")

    for img_file in image_files:
        img_path = os.path.join(output_dir, img_file)
        if os.path.exists(img_path):
            title, description = _metric_caption_html(img_file)
            html_parts.append("            <div class='figure-card'>")
            html_parts.append(f"                <h3>{title}</h3>")
            if description:
                html_parts.append(f"                <p>{description}</p>")
            html_parts.append(f"                <img src='{img_file}' alt='{title}'>")
            html_parts.append("            </div>")

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
