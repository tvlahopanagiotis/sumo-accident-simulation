#!/usr/bin/env python3
"""
visualise.py
============
Simulation Results Dashboard

Reads the output files from results/ (or any folder you point it at) and
produces a single high-resolution PNG dashboard with six panels:

  ┌──────────────────────────────────────────────────────────────────┐
  │  [header] run summary metrics                                    │
  ├──────────────────────────┬───────────────────────────────────────┤
  │  Speed timeline          │  Vehicle count + active accidents     │
  │  (accident bands shaded) │  (dual-axis)                          │
  ├──────────────────────────┼───────────────────────────────────────┤
  │  Accident map            │  Antifragility gauge                  │
  │  (location + impact)     │  (colour bar -1 → +1)                 │
  ├──────────────────────────┴───────────────────────────────────────┤
  │  Throughput timeline (full width)                                │
  └──────────────────────────────────────────────────────────────────┘

Usage:
  python visualise.py                          # reads ./results/, saves dashboard there
  python visualise.py --results-dir ./results  # explicit folder
  python visualise.py --no-show               # save only, don't open the window
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")   # non-interactive; switched to TkAgg below if --show
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as mcm
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Colour palette (dark-themed dashboard)
# ---------------------------------------------------------------------------

BG        = "#0f1117"    # near-black background
PANEL_BG  = "#1a1d27"    # panel background
GRID_C    = "#2a2d3a"    # gridlines
TEXT_C    = "#e8eaf0"    # primary text
DIM_C     = "#8890a4"    # secondary / axis labels
ACCENT    = "#4e9de0"    # blue accent (speed line)
ACCENT2   = "#f0c060"    # amber accent (vehicle count)
ACCENT3   = "#e07070"    # red accent (accidents)
GREEN_C   = "#5ec46a"    # antifragile green
YELLOW_C  = "#f0c060"    # resilient amber
RED_C     = "#e05050"    # fragile/brittle red

ACC_COLOURS = [
    "#e05050", "#e07830", "#d4b020", "#70b850",
    "#30a8b0", "#5060e0", "#9040c0", "#c040a0",
    "#40c080", "#e09080",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: str) -> tuple[pd.DataFrame, list[dict], dict]:
    """
    Load network_metrics.csv, accident_reports.json, antifragility_index.json.
    Returns (metrics_df, accidents, ai_dict).
    """
    metrics_path = os.path.join(results_dir, "network_metrics.csv")
    acc_path     = os.path.join(results_dir, "accident_reports.json")
    ai_path      = os.path.join(results_dir, "antifragility_index.json")

    # --- metrics CSV --------------------------------------------------------
    if not os.path.exists(metrics_path):
        sys.exit(f"❌  {metrics_path} not found.")
    df = pd.read_csv(metrics_path)
    if df.empty:
        sys.exit("❌  network_metrics.csv is empty — did the simulation run correctly?\n"
                 "    Make sure runner.py has the step-increment bug fixed.")

    df["time_min"] = df["timestamp_seconds"] / 60.0

    # --- accident reports ---------------------------------------------------
    accidents = []
    if os.path.exists(acc_path):
        with open(acc_path) as f:
            accidents = json.load(f)

    # --- antifragility index ------------------------------------------------
    ai = {"antifragility_index": None, "note": "No data"}
    if os.path.exists(ai_path):
        with open(ai_path) as f:
            ai = json.load(f)

    return df, accidents, ai


# ---------------------------------------------------------------------------
# Helper: shade accident bands on an Axes
# ---------------------------------------------------------------------------

def shade_accidents(ax, accidents: list[dict], alpha: float = 0.18):
    for i, acc in enumerate(accidents):
        t_start = acc["trigger_step"] / 60.0
        t_end   = acc["resolved_step"] / 60.0
        colour  = ACC_COLOURS[i % len(ACC_COLOURS)]
        ax.axvspan(t_start, t_end, color=colour, alpha=alpha, linewidth=0)


# ---------------------------------------------------------------------------
# Panel 1: Speed timeline
# ---------------------------------------------------------------------------

def plot_speed_timeline(ax, df: pd.DataFrame, accidents: list[dict]):
    ax.set_facecolor(PANEL_BG)

    shade_accidents(ax, accidents)

    # Mean speed in km/h
    ax.plot(df["time_min"], df["mean_speed_kmh"],
            color=ACCENT, linewidth=1.6, zorder=3, label="Mean speed (km/h)")

    # Baseline and post-disruption horizontal lines
    pre_mask  = (df["active_accidents"] == 0) & (df["timestamp_seconds"] <= 1800)
    post_mask = (df["active_accidents"] == 0) & (df.index > 20)
    if pre_mask.any():
        baseline = df.loc[pre_mask, "mean_speed_kmh"].mean()
        ax.axhline(baseline, color=GREEN_C, linewidth=1.0, linestyle="--",
                   alpha=0.7, label=f"Baseline {baseline:.1f} km/h")
    if post_mask.any():
        post_avg = df.loc[post_mask, "mean_speed_kmh"].tail(50).mean()
        ax.axhline(post_avg, color=YELLOW_C, linewidth=1.0, linestyle=":",
                   alpha=0.7, label=f"Post-disruption avg {post_avg:.1f} km/h")

    # Accident labels (vertical tick marks)
    for i, acc in enumerate(accidents):
        t = acc["trigger_step"] / 60.0
        ax.axvline(t, color=ACC_COLOURS[i % len(ACC_COLOURS)],
                   linewidth=0.8, alpha=0.6, linestyle="-")

    ax.set_ylabel("Speed (km/h)", color=TEXT_C, fontsize=9)
    ax.set_xlabel("Simulation time (min)", color=DIM_C, fontsize=8)
    ax.set_title("Network Mean Speed", color=TEXT_C, fontsize=10, pad=6)
    _style_ax(ax)
    ax.legend(fontsize=7, loc="upper right",
              facecolor=PANEL_BG, edgecolor=GRID_C, labelcolor=TEXT_C)


# ---------------------------------------------------------------------------
# Panel 2: Vehicle count + active accidents
# ---------------------------------------------------------------------------

def plot_vehicle_count(ax, df: pd.DataFrame, accidents: list[dict]):
    ax.set_facecolor(PANEL_BG)
    shade_accidents(ax, accidents, alpha=0.12)

    ax.fill_between(df["time_min"], df["vehicle_count"],
                    color=ACCENT2, alpha=0.25)
    ax.plot(df["time_min"], df["vehicle_count"],
            color=ACCENT2, linewidth=1.4, label="Vehicles in network")

    ax2 = ax.twinx()
    ax2.set_facecolor("none")
    ax2.step(df["time_min"], df["active_accidents"],
             where="post", color=ACCENT3, linewidth=1.2,
             alpha=0.85, label="Active accidents")
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel("Active accidents", color=ACCENT3, fontsize=8)
    ax2.tick_params(axis="y", colors=ACCENT3, labelsize=7)
    ax2.spines["right"].set_color(ACCENT3)
    for sp in ["top", "left", "bottom"]:
        ax2.spines[sp].set_visible(False)

    ax.set_ylabel("Vehicles", color=TEXT_C, fontsize=9)
    ax.set_xlabel("Simulation time (min)", color=DIM_C, fontsize=8)
    ax.set_title("Vehicle Count & Active Accidents", color=TEXT_C, fontsize=10, pad=6)
    _style_ax(ax)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left",
              facecolor=PANEL_BG, edgecolor=GRID_C, labelcolor=TEXT_C)


# ---------------------------------------------------------------------------
# Panel 3: Accident map
# ---------------------------------------------------------------------------

def plot_accident_map(ax, accidents: list[dict]):
    ax.set_facecolor(PANEL_BG)
    ax.set_title("Accident Locations & Impact", color=TEXT_C, fontsize=10, pad=6)

    if not accidents:
        ax.text(0.5, 0.5, "No accidents recorded", transform=ax.transAxes,
                color=DIM_C, ha="center", va="center", fontsize=10)
        _style_ax(ax)
        return

    xs = [a["location"]["x"] for a in accidents]
    ys = [a["location"]["y"] for a in accidents]
    sizes    = [max(20, a["impact"]["vehicles_affected"] * 8) for a in accidents]
    durations = [a["duration_seconds"] for a in accidents]

    # Colour by duration (longer = darker red)
    norm  = plt.Normalize(min(durations), max(durations) + 1)
    cmap  = LinearSegmentedColormap.from_list("dur", ["#f0c060", "#e05050", "#800020"])
    cols  = [cmap(norm(d)) for d in durations]

    sc = ax.scatter(xs, ys, s=sizes, c=cols, alpha=0.85,
                    edgecolors=TEXT_C, linewidths=0.5, zorder=3)

    # Label each accident
    for i, acc in enumerate(accidents):
        ax.annotate(
            acc["accident_id"].replace("ACC_", "#"),
            (acc["location"]["x"], acc["location"]["y"]),
            textcoords="offset points", xytext=(6, 4),
            color=TEXT_C, fontsize=6.5, zorder=4,
        )

    # Colour bar for duration
    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label("Duration (s)", color=DIM_C, fontsize=7)
    cbar.ax.tick_params(colors=DIM_C, labelsize=6)
    cbar.outline.set_edgecolor(GRID_C)

    # Size legend
    for veh, lbl in [(5, "5 veh"), (15, "15 veh"), (25, "25 veh")]:
        ax.scatter([], [], s=veh * 8, c="#808080",
                   edgecolors=TEXT_C, linewidths=0.4,
                   label=lbl, alpha=0.8)
    ax.legend(title="Affected vehicles", title_fontsize=6,
              fontsize=6, loc="lower right",
              facecolor=PANEL_BG, edgecolor=GRID_C, labelcolor=TEXT_C)

    ax.set_xlabel("x (m)", color=DIM_C, fontsize=8)
    ax.set_ylabel("y (m)", color=DIM_C, fontsize=8)
    _style_ax(ax)


# ---------------------------------------------------------------------------
# Panel 4: Antifragility gauge
# ---------------------------------------------------------------------------

def plot_antifragility(ax, ai: dict):
    ax.set_facecolor(PANEL_BG)
    ax.set_title("Antifragility Index — per-event & aggregate", color=TEXT_C, fontsize=10, pad=6)

    ai_val     = ai.get("antifragility_index")
    ci_low     = ai.get("ci_95_low")
    ci_high    = ai.get("ci_95_high")
    per_event  = ai.get("per_event", [])

    # ── Colour gradient bar ───────────────────────────────────────────────
    bar_cmap = LinearSegmentedColormap.from_list(
        "ai", [(0.0, RED_C), (0.45, YELLOW_C), (0.55, YELLOW_C), (1.0, GREEN_C)]
    )
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect="auto", cmap=bar_cmap,
              extent=[-1.0, 1.0, -0.1, 0.1], zorder=1)

    # Zone dividers + labels
    for v in [-0.20, -0.05, 0.05]:
        ax.axvline(v, color=PANEL_BG, linewidth=1.0, zorder=2, alpha=0.7)
    for v, lbl in [(-0.60, "BRITTLE"), (-0.125, "FRAGILE"),
                   (0.0, "RESILIENT"), (0.50, "ANTIFRAGILE")]:
        ax.text(v, -0.22, lbl, color=TEXT_C, fontsize=5.5,
                ha="center", va="top", alpha=0.75)

    # Per-event dots above the bar
    for i, ev in enumerate(per_event):
        ev_ai = ev.get("event_ai", 0)
        clamped = max(-1.0, min(1.0, ev_ai))
        y_jitter = 0.14 + (i % 3) * 0.08
        col = ACC_COLOURS[i % len(ACC_COLOURS)]
        ax.scatter([clamped], [y_jitter], color=col, s=28, zorder=4,
                   edgecolors=TEXT_C, linewidths=0.4)
        ax.text(clamped, y_jitter + 0.07,
                ev["accident_id"].replace("ACC_", "#"),
                color=col, fontsize=5.5, ha="center", va="bottom", zorder=5)

    # Aggregate needle + CI bracket
    if ai_val is not None:
        clamped = max(-1.0, min(1.0, ai_val))
        ax.annotate("",
                    xy=(clamped, -0.08), xytext=(clamped, -0.38),
                    arrowprops=dict(arrowstyle="->", color=TEXT_C, lw=2.2),
                    zorder=6)
        ax.text(clamped, -0.55, f"{ai_val:+.4f}",
                color=TEXT_C, fontsize=13, fontweight="bold",
                ha="center", va="top", zorder=6)

        # 95% CI bracket below the needle
        if ci_low is not None and ci_high is not None:
            lo = max(-1.0, min(1.0, ci_low))
            hi = max(-1.0, min(1.0, ci_high))
            ax.annotate("", xy=(lo, -0.70), xytext=(hi, -0.70),
                        arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.5))
            ax.text(clamped, -0.82,
                    f"95% CI  [{ci_low:+.4f},  {ci_high:+.4f}]",
                    color=ACCENT, fontsize=7, ha="center", va="top")

    # Interpretation + counts
    interp  = ai.get("interpretation") or ai.get("note", "")
    n_ev    = ai.get("n_events_measured", "—")
    n_total = ai.get("total_accidents", "—")
    ax.text(0, -1.00, interp, color=DIM_C, fontsize=7.5,
            ha="center", va="top", style="italic",
            transform=ax.get_xaxis_transform())
    ax.text(0, -1.15,
            f"Events measured: {n_ev} / {n_total} accidents",
            color=DIM_C, fontsize=7,
            ha="center", va="top",
            transform=ax.get_xaxis_transform())

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.25, 0.45)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.tick_params(axis="x", colors=DIM_C, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_C)


# ---------------------------------------------------------------------------
# Panel 5: Throughput timeline (full width, bottom)
# ---------------------------------------------------------------------------

def plot_throughput(ax, df: pd.DataFrame, accidents: list[dict]):
    ax.set_facecolor(PANEL_BG)
    shade_accidents(ax, accidents, alpha=0.15)

    # Rolling average to smooth the bursty throughput counter
    window = max(1, len(df) // 30)
    smooth = df["throughput_per_hour"].rolling(window, center=True, min_periods=1).mean()

    ax.fill_between(df["time_min"], smooth,
                    color=GREEN_C, alpha=0.20)
    ax.plot(df["time_min"], smooth,
            color=GREEN_C, linewidth=1.4, label="Throughput (veh/hr, rolling avg)")

    # Speed ratio on secondary axis
    ax2 = ax.twinx()
    ax2.set_facecolor("none")
    ax2.plot(df["time_min"], df["speed_ratio"],
             color=ACCENT, linewidth=1.0, alpha=0.6,
             linestyle="--", label="Speed ratio (actual/free-flow)")
    ax2.axhline(1.0, color=ACCENT, linewidth=0.6, alpha=0.3)
    ax2.set_ylim(0, 1.4)
    ax2.set_ylabel("Speed ratio", color=ACCENT, fontsize=8)
    ax2.tick_params(axis="y", colors=ACCENT, labelsize=7)
    ax2.spines["right"].set_color(ACCENT)
    for sp in ["top", "left", "bottom"]:
        ax2.spines[sp].set_visible(False)

    ax.set_ylabel("Throughput (veh/hr)", color=TEXT_C, fontsize=9)
    ax.set_xlabel("Simulation time (min)", color=DIM_C, fontsize=8)
    ax.set_title("Network Throughput & Speed Ratio", color=TEXT_C, fontsize=10, pad=6)
    _style_ax(ax)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc="upper right",
              facecolor=PANEL_BG, edgecolor=GRID_C, labelcolor=TEXT_C)


# ---------------------------------------------------------------------------
# Header panel (text summary)
# ---------------------------------------------------------------------------

def draw_header(ax, df: pd.DataFrame, accidents: list[dict], ai: dict):
    ax.set_facecolor("#141720")
    ax.axis("off")

    ai_val   = ai.get("antifragility_index")
    interp   = ai.get("interpretation", ai.get("note", "—"))
    ci_low   = ai.get("ci_95_low")
    ci_high  = ai.get("ci_95_high")
    n_events = ai.get("n_events_measured", "—")
    n_acc    = len(accidents)
    sim_min  = df["time_min"].max()
    peak_veh = int(df["vehicle_count"].max())
    avg_spd  = df["mean_speed_kmh"].mean()

    ai_color = (GREEN_C if (ai_val or 0) > 0.05
                else YELLOW_C if (ai_val or 0) > -0.05
                else RED_C)

    ax.text(0.01, 0.85, "SUMO ACCIDENT SIMULATION — RESULTS DASHBOARD",
            transform=ax.transAxes, color=TEXT_C,
            fontsize=12, fontweight="bold", va="center")

    # AI value + CI on its own line
    if ai_val is not None:
        ci_str = (f"  95% CI [{ci_low:+.4f}, {ci_high:+.4f}]  n={n_events} events"
                  if ci_low is not None else f"  n={n_events} events")
        ax.text(0.01, 0.58,
                f"AI = {ai_val:+.4f}{ci_str}   —   {interp}",
                transform=ax.transAxes, color=ai_color,
                fontsize=8.5, fontweight="bold", va="center")
    else:
        ax.text(0.01, 0.58, interp, transform=ax.transAxes,
                color=DIM_C, fontsize=8, va="center", style="italic")

    stats = [
        ("Simulation duration", f"{sim_min:.0f} min"),
        ("Total accidents",      str(n_acc)),
        ("Peak vehicles",        str(peak_veh)),
        ("Mean network speed",   f"{avg_spd:.1f} km/h"),
    ]
    x_positions = [0.01, 0.20, 0.38, 0.56]
    for (label, value), xp in zip(stats, x_positions):
        ax.text(xp, 0.38, label, transform=ax.transAxes,
                color=DIM_C, fontsize=7, va="center")
        ax.text(xp, 0.18, value, transform=ax.transAxes,
                color=TEXT_C, fontsize=8.5, fontweight="bold", va="center")

    # Accident mini-chips
    for i, acc in enumerate(accidents):
        chip_x = 0.01 + i * 0.095
        if chip_x > 0.92:
            break
        dur_s = acc["duration_seconds"]
        aff   = acc["impact"]["vehicles_affected"]
        col   = ACC_COLOURS[i % len(ACC_COLOURS)]
        ax.text(chip_x, -0.05,
                f"#{acc['accident_id'].split('_')[1]}  {dur_s//60}min  {aff}veh",
                transform=ax.transAxes, color=col, fontsize=6.5,
                va="top", bbox=dict(facecolor=PANEL_BG, edgecolor=col,
                                    boxstyle="round,pad=0.3", linewidth=0.7))


# ---------------------------------------------------------------------------
# Shared axis styler
# ---------------------------------------------------------------------------

def _style_ax(ax):
    ax.tick_params(axis="both", colors=DIM_C, labelsize=7)
    ax.xaxis.label.set_color(DIM_C)
    ax.yaxis.label.set_color(TEXT_C)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.grid(True, color=GRID_C, linewidth=0.5, alpha=0.6)
    ax.set_facecolor(PANEL_BG)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_dashboard(results_dir: str, show: bool = False) -> str:
    print(f"\nLoading results from: {results_dir}")
    df, accidents, ai = load_results(results_dir)
    print(f"  Snapshots   : {len(df)}")
    print(f"  Accidents   : {len(accidents)}")
    print(f"  AI index    : {ai.get('antifragility_index')}")

    # ── Figure layout ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12), dpi=120, facecolor=BG)
    gs  = gridspec.GridSpec(
        4, 2,
        figure=fig,
        height_ratios=[0.55, 2.2, 2.2, 2.0],
        hspace=0.48, wspace=0.30,
        left=0.07, right=0.97, top=0.97, bottom=0.05,
    )

    ax_header     = fig.add_subplot(gs[0, :])
    ax_speed      = fig.add_subplot(gs[1, 0])
    ax_vehicles   = fig.add_subplot(gs[1, 1])
    ax_map        = fig.add_subplot(gs[2, 0])
    ax_ai         = fig.add_subplot(gs[2, 1])
    ax_throughput = fig.add_subplot(gs[3, :])

    # ── Draw panels ─────────────────────────────────────────────────────────
    draw_header(ax_header, df, accidents, ai)
    plot_speed_timeline(ax_speed, df, accidents)
    plot_vehicle_count(ax_vehicles, df, accidents)
    plot_accident_map(ax_map, accidents)
    plot_antifragility(ax_ai, ai)
    plot_throughput(ax_throughput, df, accidents)

    # ── Save ────────────────────────────────────────────────────────────────
    out_path = os.path.join(results_dir, "simulation_dashboard.png")
    fig.savefig(out_path, dpi=120, facecolor=BG, bbox_inches="tight")
    print(f"\n✅  Dashboard saved → {out_path}")

    if show:
        matplotlib.use("TkAgg")
        plt.show()

    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate a visual dashboard from SAS simulation results"
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
        help="Folder containing network_metrics.csv etc. (default: ./results/)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the PNG without opening a display window",
    )
    args = parser.parse_args()

    build_dashboard(
        results_dir=os.path.abspath(args.results_dir),
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
