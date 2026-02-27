#!/usr/bin/env python3
"""
visualise_batch.py
==================
Multi-Run Aggregate Dashboard

Reads the results of a --runs N batch (one sub-folder per seed plus an
aggregate/ folder) and produces two outputs:

  1. results/aggregate/batch_dashboard.png
       6-panel summary across all runs

  2. results/aggregate/batch_ai_distribution.png
       Publication-quality AI distribution figure

Layout of batch_dashboard.png
------------------------------
  ┌────────────────────────────────────────────────────┐
  │  Header — aggregate AI, CI, run count              │
  ├──────────────────────┬─────────────────────────────┤
  │  AI per run          │  Speed timelines overlay     │
  │  (bars + CI + mean)  │  (all runs + mean ± σ band)  │
  ├──────────────────────┼─────────────────────────────┤
  │  Accident count dist │  Accident hotspot map        │
  │  (bar per run)       │  (Sioux Falls topology)      │
  ├──────────────────────┴─────────────────────────────┤
  │  Throughput overlay (all runs + mean)              │
  └────────────────────────────────────────────────────┘

Usage
-----
    python visualise_batch.py                        # reads ./results/
    python visualise_batch.py --results-dir ./results
    python visualise_batch.py --no-show
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sioux Falls topology  (for the hotspot map)
# ---------------------------------------------------------------------------

SF_NODES: dict[int, tuple[int, int]] = {
     1: (1000, 5000),   2: (4000, 5000),   3: (7000, 5000),
     4: (2000, 4500),   5: (6000, 4500),
     6: (3000, 4000),   7: (5000, 4000),
     8: (3000, 3500),   9: (5000, 3500),  10: (7000, 3500),
    11: (1500, 3000),  12: (3000, 3000),  13: (5000, 3000),  14: (7000, 3000),
    15: (1500, 2500),  16: (3000, 2500),  17: (5000, 2500),  18: (7000, 2500),
    19: (1500, 2000),  20: (3500, 2000),  21: (5000, 2000),
    22: (1500, 1500),  23: (3500, 1500),  24: (5000, 1500),
}

SF_LINKS: list[tuple[int, int]] = [
    ( 1,  2), ( 1,  3), ( 2,  6), ( 3,  4), ( 3, 12),
    ( 4,  5), ( 4, 11), ( 5,  6), ( 5,  9), ( 6,  7),
    ( 6,  8), ( 7,  8), ( 7,  9), ( 8,  9), ( 8, 11),
    ( 8, 16), ( 9, 10), ( 9, 13), (10, 14), (11, 12),
    (11, 15), (12, 13), (13, 14), (13, 17), (14, 18),
    (15, 16), (15, 19), (16, 17), (16, 20), (17, 18),
    (17, 21), (19, 20), (19, 22), (20, 21), (20, 23),
    (21, 24), (22, 23), (23, 24),
]


# ---------------------------------------------------------------------------
# Colour palette  (matches visualise.py dark theme)
# ---------------------------------------------------------------------------

BG       = "#0f1117"
PANEL_BG = "#1a1d27"
GRID_C   = "#2a2d3a"
TEXT_C   = "#e8eaf0"
DIM_C    = "#8890a4"
ACCENT   = "#4e9de0"
ACCENT2  = "#f0c060"
ACCENT3  = "#e07070"
GREEN_C  = "#5ec46a"
YELLOW_C = "#f0c060"
RED_C    = "#e05050"

RUN_PALETTE = [
    "#4e9de0", "#f0c060", "#e07070", "#5ec46a", "#c07ad0",
    "#e09050", "#50c0d0", "#e0a0d0", "#90d060", "#a070e0",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_run_dirs(results_dir: str) -> list[str]:
    """Return sorted list of run_XXXX sub-directories."""
    dirs = []
    for name in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, name)
        if os.path.isdir(path) and name.startswith("run_"):
            if os.path.exists(os.path.join(path, "network_metrics.csv")):
                dirs.append(path)
    return dirs


def load_run(run_dir: str) -> tuple[pd.DataFrame, list[dict], dict]:
    df  = pd.read_csv(os.path.join(run_dir, "network_metrics.csv"))
    df["time_min"] = df["timestamp_seconds"] / 60.0

    acc_path = os.path.join(run_dir, "accident_reports.json")
    accidents = json.load(open(acc_path)) if os.path.exists(acc_path) else []

    ai_path = os.path.join(run_dir, "antifragility_index.json")
    ai      = json.load(open(ai_path)) if os.path.exists(ai_path) else {}

    return df, accidents, ai


def load_aggregate(results_dir: str) -> dict:
    path = os.path.join(results_dir, "aggregate", "aggregate_summary.json")
    if not os.path.exists(path):
        sys.exit(f"❌  {path} not found — run with --runs N first.")
    return json.load(open(path))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _style_ax(ax):
    ax.tick_params(axis="both", colors=DIM_C, labelsize=7)
    ax.xaxis.label.set_color(DIM_C)
    ax.yaxis.label.set_color(TEXT_C)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.grid(True, color=GRID_C, linewidth=0.5, alpha=0.6)
    ax.set_facecolor(PANEL_BG)


def _ai_color(ai_val):
    if ai_val is None:
        return DIM_C
    if ai_val > 0.05:
        return GREEN_C
    if ai_val > -0.05:
        return YELLOW_C
    if ai_val > -0.20:
        return RED_C
    return "#800020"


# ---------------------------------------------------------------------------
# Panel: Header
# ---------------------------------------------------------------------------

def draw_header(ax, agg: dict, n_runs: int):
    ax.set_facecolor("#141720")
    ax.axis("off")

    ai_val  = agg.get("ai_mean")
    ci_low  = agg.get("ai_ci_95_low")
    ci_high = agg.get("ai_ci_95_high")
    n_acc_m = agg.get("accident_mean", 0)
    n_acc_s = agg.get("accident_std",  0)

    ai_col  = _ai_color(ai_val)

    ax.text(0.01, 0.88,
            "SUMO ACCIDENT SIMULATION — BATCH RESULTS DASHBOARD",
            transform=ax.transAxes, color=TEXT_C,
            fontsize=12, fontweight="bold", va="center")

    if ai_val is not None:
        ci_str = (f"  95% CI [{ci_low:+.4f}, {ci_high:+.4f}]"
                  if ci_low is not None else "")
        interp = ("ANTIFRAGILE" if ai_val > 0.05 else
                  "RESILIENT"   if ai_val > -0.05 else
                  "FRAGILE"     if ai_val > -0.20 else "BRITTLE")
        ax.text(0.01, 0.60,
                f"Aggregate AI = {ai_val:+.4f}{ci_str}   —   {interp}",
                transform=ax.transAxes, color=ai_col,
                fontsize=9, fontweight="bold", va="center")

    stats = [
        ("Runs completed",    str(n_runs)),
        ("Accidents / run",   f"{n_acc_m:.1f} ± {n_acc_s:.2f}"),
        ("AI std dev",        f"{agg.get('ai_std', 0):.4f}" if agg.get('ai_std') else "—"),
        ("Network",           "Sioux Falls  (LeBlanc 1975)"),
    ]
    for i, (label, value) in enumerate(stats):
        xp = 0.01 + i * 0.23
        ax.text(xp, 0.35, label, transform=ax.transAxes,
                color=DIM_C, fontsize=7, va="center")
        ax.text(xp, 0.12, value, transform=ax.transAxes,
                color=TEXT_C, fontsize=8.5, fontweight="bold", va="center")


# ---------------------------------------------------------------------------
# Panel 1: AI per run — horizontal bar chart
# ---------------------------------------------------------------------------

def plot_ai_per_run(ax, run_data: list[tuple]):
    """run_data: list of (seed_label, ai_val, ci_low, ci_high)"""
    ax.set_facecolor(PANEL_BG)
    ax.set_title("Antifragility Index — per run", color=TEXT_C, fontsize=10, pad=6)

    labels    = [r[0] for r in run_data]
    ai_vals   = [r[1] for r in run_data]
    ci_lows   = [r[2] for r in run_data]
    ci_highs  = [r[3] for r in run_data]
    n         = len(labels)
    ys        = list(range(n))

    for i, (label, ai, clo, chi, *_) in enumerate(run_data):
        col  = _ai_color(ai)
        if ai is None:
            ax.barh(i, 0, height=0.55, color=DIM_C, alpha=0.3)
            ax.text(0.01, i, "no data", color=DIM_C, va="center", fontsize=7)
            continue
        ax.barh(i, ai, height=0.55, color=col, alpha=0.7, left=0)
        if clo is not None and chi is not None:
            ax.errorbar(ai, i, xerr=[[ai - clo], [chi - ai]],
                        fmt="none", color=TEXT_C, capsize=3, linewidth=1.2)
        ax.text(max(ai, 0) + 0.002, i, f"{ai:+.4f}",
                color=col, va="center", fontsize=6.5, fontweight="bold")

    # Aggregate mean line
    ai_vals_valid = [v for v in ai_vals if v is not None]
    if ai_vals_valid:
        mean_ai = np.mean(ai_vals_valid)
        ax.axvline(mean_ai, color=TEXT_C, linewidth=1.5, linestyle="--",
                   alpha=0.7, label=f"Mean {mean_ai:+.4f}", zorder=5)

    # Zone shading
    ax.axvspan(-1.0, -0.20, color=RED_C,    alpha=0.05)
    ax.axvspan(-0.20, -0.05, color=ACCENT3, alpha=0.05)
    ax.axvspan(-0.05,  0.05, color=YELLOW_C, alpha=0.05)
    ax.axvspan( 0.05,  1.0,  color=GREEN_C,  alpha=0.05)
    ax.axvline(0, color=GRID_C, linewidth=0.8, linestyle="-")

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7, color=DIM_C)
    ax.set_xlabel("Antifragility Index (AI)", color=DIM_C, fontsize=8)
    ax.set_xlim(-0.35, 0.35)
    ax.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_C, labelcolor=TEXT_C,
              loc="upper right")
    _style_ax(ax)
    ax.grid(axis="x", color=GRID_C, linewidth=0.5, alpha=0.6)
    ax.grid(axis="y", visible=False)


# ---------------------------------------------------------------------------
# Panel 2: Speed timeline overlay
# ---------------------------------------------------------------------------

def plot_speed_overlay(ax, dfs: list[pd.DataFrame]):
    ax.set_facecolor(PANEL_BG)
    ax.set_title("Speed Timeline — all runs (mean ± 1σ band)", color=TEXT_C, fontsize=10, pad=6)

    # Align all runs to a common time axis
    common_time = dfs[0]["time_min"].values
    speed_mat   = []
    for df in dfs:
        s = np.interp(common_time, df["time_min"].values, df["mean_speed_kmh"].values)
        speed_mat.append(s)
        ax.plot(common_time, s, linewidth=0.7, alpha=0.35,
                color=ACCENT, zorder=2)

    speed_arr = np.array(speed_mat)
    mean_spd  = speed_arr.mean(axis=0)
    std_spd   = speed_arr.std(axis=0)

    ax.fill_between(common_time, mean_spd - std_spd, mean_spd + std_spd,
                    color=ACCENT, alpha=0.18, zorder=3, label="Mean ± 1σ")
    ax.plot(common_time, mean_spd, color=ACCENT, linewidth=2.2,
            zorder=4, label=f"Mean  ({mean_spd.mean():.1f} km/h avg)")

    ax.set_xlabel("Simulation time (min)", color=DIM_C, fontsize=8)
    ax.set_ylabel("Network mean speed (km/h)", color=TEXT_C, fontsize=9)
    ax.legend(fontsize=7.5, facecolor=PANEL_BG, edgecolor=GRID_C,
              labelcolor=TEXT_C, loc="upper right")
    _style_ax(ax)


# ---------------------------------------------------------------------------
# Panel 3: Accident count per run
# ---------------------------------------------------------------------------

def plot_accident_counts(ax, run_data: list[tuple], agg: dict):
    ax.set_facecolor(PANEL_BG)
    ax.set_title("Accidents per Run", color=TEXT_C, fontsize=10, pad=6)

    labels = [r[0] for r in run_data]
    counts = [r[4] for r in run_data]       # total_accidents
    cols   = [RUN_PALETTE[i % len(RUN_PALETTE)] for i in range(len(counts))]

    bars = ax.bar(labels, counts, color=cols, alpha=0.75, width=0.6)

    # Mean line
    mean_acc = agg.get("accident_mean", np.mean(counts))
    ax.axhline(mean_acc, color=TEXT_C, linewidth=1.5, linestyle="--",
               alpha=0.7, label=f"Mean {mean_acc:.1f}")

    # Value labels on bars
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                str(val), color=TEXT_C, ha="center", va="bottom", fontsize=8,
                fontweight="bold")

    ax.set_ylabel("Total accidents", color=TEXT_C, fontsize=9)
    ax.set_xlabel("Run (seed)", color=DIM_C, fontsize=8)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_C, labelcolor=TEXT_C)
    _style_ax(ax)
    ax.grid(axis="x", visible=False)


# ---------------------------------------------------------------------------
# Panel 4: Accident hotspot map (Sioux Falls topology)
# ---------------------------------------------------------------------------

def plot_hotspot_map(ax, all_accidents: list[dict]):
    ax.set_facecolor(PANEL_BG)
    ax.set_title("Accident Hotspots — all runs combined", color=TEXT_C, fontsize=10, pad=6)

    # Draw network edges
    for (a, b) in SF_LINKS:
        x1, y1 = SF_NODES[a]
        x2, y2 = SF_NODES[b]
        ax.plot([x1, x2], [y1, y2], color=GRID_C, linewidth=1.0,
                zorder=1, solid_capstyle="round")

    # Draw nodes
    nxs = [SF_NODES[n][0] for n in SF_NODES]
    nys = [SF_NODES[n][1] for n in SF_NODES]
    ax.scatter(nxs, nys, s=18, color=DIM_C, zorder=2, edgecolors=PANEL_BG, linewidths=0.5)
    for nid, (nx, ny) in SF_NODES.items():
        ax.text(nx, ny + 90, str(nid), color=DIM_C, fontsize=4.5,
                ha="center", va="bottom", zorder=3)

    if not all_accidents:
        ax.text(0.5, 0.5, "No accidents recorded", transform=ax.transAxes,
                color=DIM_C, ha="center", va="center", fontsize=10)
        _style_ax(ax)
        return

    # Aggregate accident locations — bin by proximity for a heat map feel
    xs = np.array([a["location"]["x"] for a in all_accidents])
    ys = np.array([a["location"]["y"] for a in all_accidents])
    durations = np.array([a["duration_seconds"] for a in all_accidents])
    sizes     = np.array([max(30, a["impact"]["vehicles_affected"] * 12)
                          for a in all_accidents])

    norm = plt.Normalize(durations.min(), durations.max() + 1)
    cmap = LinearSegmentedColormap.from_list("hot", [YELLOW_C, ACCENT3, "#800020"])

    sc = ax.scatter(xs, ys, s=sizes, c=durations, cmap=cmap, norm=norm,
                    alpha=0.80, edgecolors=TEXT_C, linewidths=0.4, zorder=4)

    cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label("Duration (s)", color=DIM_C, fontsize=7)
    cbar.ax.tick_params(colors=DIM_C, labelsize=6)
    cbar.outline.set_edgecolor(GRID_C)

    # Legend for size
    for aff, lbl in [(5, "5 veh"), (15, "15 veh")]:
        ax.scatter([], [], s=aff * 12, c="#808080", edgecolors=TEXT_C,
                   linewidths=0.4, label=lbl, alpha=0.8)
    ax.legend(title="Affected", title_fontsize=6, fontsize=6, loc="lower right",
              facecolor=PANEL_BG, edgecolor=GRID_C, labelcolor=TEXT_C)

    ax.set_xlabel("x (m)", color=DIM_C, fontsize=8)
    ax.set_ylabel("y (m)", color=DIM_C, fontsize=8)
    _style_ax(ax)


# ---------------------------------------------------------------------------
# Panel 5: Throughput overlay
# ---------------------------------------------------------------------------

def plot_throughput_overlay(ax, dfs: list[pd.DataFrame]):
    ax.set_facecolor(PANEL_BG)
    ax.set_title("Throughput — all runs (mean ± 1σ)", color=TEXT_C, fontsize=10, pad=6)

    common_time = dfs[0]["time_min"].values
    thr_mat = []
    window  = max(1, len(dfs[0]) // 30)

    for df in dfs:
        smooth = df["throughput_per_hour"].rolling(window, center=True,
                                                   min_periods=1).mean().values
        t      = np.interp(common_time, df["time_min"].values, smooth)
        thr_mat.append(t)
        ax.plot(common_time, t, linewidth=0.7, alpha=0.3, color=GREEN_C, zorder=2)

    thr_arr  = np.array(thr_mat)
    mean_thr = thr_arr.mean(axis=0)
    std_thr  = thr_arr.std(axis=0)

    ax.fill_between(common_time, mean_thr - std_thr, mean_thr + std_thr,
                    color=GREEN_C, alpha=0.18, zorder=3)
    ax.plot(common_time, mean_thr, color=GREEN_C, linewidth=2.0,
            zorder=4, label=f"Mean  ({mean_thr.mean():.0f} veh/hr avg)")

    ax.set_xlabel("Simulation time (min)", color=DIM_C, fontsize=8)
    ax.set_ylabel("Throughput (veh/hr)", color=TEXT_C, fontsize=9)
    ax.legend(fontsize=7.5, facecolor=PANEL_BG, edgecolor=GRID_C,
              labelcolor=TEXT_C, loc="upper right")
    _style_ax(ax)


# ---------------------------------------------------------------------------
# Publication figure: AI distribution
# ---------------------------------------------------------------------------

def plot_ai_distribution(agg: dict, run_data: list[tuple], out_path: str):
    """
    Single clean figure suitable for a paper appendix:
    AI values per run with aggregate CI, coloured by interpretation zone.
    """
    ai_valid = [(r[0], r[1]) for r in run_data if r[1] is not None]
    if not ai_valid:
        print("  Skipping AI distribution — no valid events.")
        return

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150, facecolor=BG)
    ax.set_facecolor(PANEL_BG)

    labels = [r[0] for r in ai_valid]
    values = [r[1] for r in ai_valid]
    cols   = [_ai_color(v) for v in values]

    xs = list(range(len(values)))
    ax.bar(xs, values, color=cols, alpha=0.75, width=0.55)

    # Individual CI error bars
    for i, rd in enumerate([r for r in run_data if r[1] is not None]):
        _, ai, clo, chi, _ = rd
        if clo is not None and chi is not None:
            ax.errorbar(i, ai, yerr=[[ai - clo], [chi - ai]],
                        fmt="none", color=TEXT_C, capsize=4, linewidth=1.2)

    # Aggregate mean + CI band
    mean_ai = agg.get("ai_mean")
    ci_lo   = agg.get("ai_ci_95_low")
    ci_hi   = agg.get("ai_ci_95_high")
    if mean_ai is not None:
        ax.axhline(mean_ai, color=TEXT_C, linewidth=2, linestyle="--",
                   label=f"Aggregate AI = {mean_ai:+.4f}", zorder=5)
    if ci_lo is not None and ci_hi is not None:
        ax.axhspan(ci_lo, ci_hi, color=TEXT_C, alpha=0.10, zorder=4,
                   label=f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    # Zone lines
    for v, lbl, col in [(-0.20, "Fragile threshold", ACCENT3),
                         ( 0.05, "Antifragile threshold", GREEN_C)]:
        ax.axhline(v, color=col, linewidth=0.9, linestyle=":", alpha=0.7)
        ax.text(len(values) - 0.4, v + 0.005, lbl, color=col,
                fontsize=6.5, va="bottom", ha="right")

    ax.axhline(0, color=GRID_C, linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, color=DIM_C, fontsize=8)
    ax.set_ylabel("Antifragility Index (AI)", color=TEXT_C, fontsize=9)
    ax.set_title("Per-Run Antifragility Index with 95% CI  —  Sioux Falls Benchmark",
                 color=TEXT_C, fontsize=10, pad=8)
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_C, labelcolor=TEXT_C)
    _style_ax(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  AI distribution → {out_path}")


# ---------------------------------------------------------------------------
# Main dashboard builder
# ---------------------------------------------------------------------------

def build_batch_dashboard(results_dir: str):
    print(f"\nLoading batch results from: {results_dir}")

    agg      = load_aggregate(results_dir)
    run_dirs = find_run_dirs(results_dir)
    n_runs   = len(run_dirs)

    if n_runs == 0:
        sys.exit("❌  No run_XXXX sub-folders found. Run with --runs N first.")

    print(f"  Found {n_runs} run folders")

    # Load all runs
    all_dfs, all_accidents_per_run = [], []
    run_summaries = []   # (label, ai, ci_lo, ci_hi, n_accidents)

    for i, rdir in enumerate(run_dirs):
        seed_label = os.path.basename(rdir).replace("run_", "s")
        df, accidents, ai = load_run(rdir)
        all_dfs.append(df)
        all_accidents_per_run.append(accidents)

        ai_val  = ai.get("antifragility_index")
        ci_lo   = ai.get("ci_95_low")
        ci_hi   = ai.get("ci_95_high")
        n_acc   = agg["runs"][i]["total_accidents"] if i < len(agg["runs"]) else len(accidents)
        run_summaries.append((seed_label, ai_val, ci_lo, ci_hi, n_acc))

    all_accidents = [a for accidents in all_accidents_per_run for a in accidents]
    print(f"  Total accidents across all runs: {len(all_accidents)}")

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 13), dpi=120, facecolor=BG)
    gs  = gridspec.GridSpec(
        4, 2,
        figure=fig,
        height_ratios=[0.50, 2.5, 2.5, 2.0],
        hspace=0.50, wspace=0.28,
        left=0.07, right=0.97, top=0.97, bottom=0.06,
    )

    ax_header    = fig.add_subplot(gs[0, :])
    ax_ai        = fig.add_subplot(gs[1, 0])
    ax_speed     = fig.add_subplot(gs[1, 1])
    ax_counts    = fig.add_subplot(gs[2, 0])
    ax_hotspot   = fig.add_subplot(gs[2, 1])
    ax_thr       = fig.add_subplot(gs[3, :])

    # ── Draw panels ──────────────────────────────────────────────────────────
    draw_header(ax_header, agg["aggregate"], n_runs)
    plot_ai_per_run(ax_ai, run_summaries)
    plot_speed_overlay(ax_speed, all_dfs)
    plot_accident_counts(ax_counts, run_summaries, agg["aggregate"])
    plot_hotspot_map(ax_hotspot, all_accidents)
    plot_throughput_overlay(ax_thr, all_dfs)

    # ── Save main dashboard ──────────────────────────────────────────────────
    agg_dir  = os.path.join(results_dir, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)
    out_main = os.path.join(agg_dir, "batch_dashboard.png")
    fig.savefig(out_main, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✅  Batch dashboard → {out_main}")

    # ── Save AI distribution figure ───────────────────────────────────────────
    out_dist = os.path.join(agg_dir, "batch_ai_distribution.png")
    plot_ai_distribution(agg["aggregate"], run_summaries, out_dist)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Multi-run aggregate dashboard for SAS batch results"
    )
    ap.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
        help="Folder containing run_XXXX sub-folders (default: ./results/)",
    )
    args = ap.parse_args()
    build_batch_dashboard(os.path.abspath(args.results_dir))


if __name__ == "__main__":
    main()
