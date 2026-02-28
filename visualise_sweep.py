"""
visualise_sweep.py
==================
Failure-point visualisation for the SAS parameter-grid sweep.

Reads  :  results/sweep/sweep_results.csv

Writes :  results/sweep/figures/
              fig1_speed_vs_load.png    — MFD-style speed curves per prob level
              fig2_ai_vs_load.png       — AI vs traffic load with ±σ ribbon
              fig3_heatmaps.png         — 2-D speed-ratio & AI heatmaps
              fig4_phase_diagram.png    — resilience-regime phase diagram

Usage
-----
    python visualise_sweep.py
    python visualise_sweep.py --csv results/sweep/sweep_results.csv
    python visualise_sweep.py --out-dir results/sweep/figs
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------

DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
GRID_COL = "#21262d"
TEXT_COL = "#e6edf3"

# One colour per accident-probability level (up to 6)
PROB_COLOURS = [
    "#8b949e",   # grey   — baseline (no accidents)
    "#4fc3f7",   # light blue
    "#f0883e",   # orange
    "#ff7b72",   # red-orange
    "#d2a8ff",   # lavender
    "#ffa657",   # amber
]

_RCPARAMS = {
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_COL,
    "xtick.color":       TEXT_COL,
    "ytick.color":       TEXT_COL,
    "text.color":        TEXT_COL,
    "grid.color":        GRID_COL,
    "grid.linewidth":    0.6,
    "legend.facecolor":  PANEL_BG,
    "legend.edgecolor":  GRID_COL,
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlepad":     10,
}

# AI regime colours and boundaries
AI_ZONES = [
    ( 0.05,  1.00, "#238636", "Antifragile   (AI > +0.05)"),
    (-0.05,  0.05, "#1f6feb", "Resilient     (|AI| ≤ 0.05)"),
    (-0.20, -0.05, "#d29922", "Fragile       (−0.20 < AI ≤ −0.05)"),
    (-1.00, -0.20, "#da3633", "Brittle       (AI ≤ −0.20)"),
]


# ---------------------------------------------------------------------------
# Data loading & aggregation
# ---------------------------------------------------------------------------

def load_sweep(csv_path: str) -> pd.DataFrame:
    """Load sweep_results.csv and coerce numeric columns."""
    if not os.path.exists(csv_path):
        sys.exit(f"ERROR: sweep CSV not found — {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"period", "prob", "seed", "n_accidents"}
    missing  = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: CSV missing columns: {missing}")

    numeric = [
        "period", "prob", "seed",
        "n_accidents", "mean_speed_ms", "mean_speed_kmh",
        "mean_throughput", "mean_speed_ratio",
        "ai", "ci_low", "ci_high", "n_events_measured",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(
        f"  Loaded {len(df)} rows  "
        f"({df['period'].nunique()} periods × "
        f"{df['prob'].nunique()} prob levels × "
        f"{df['seed'].nunique()} seeds)"
    )
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate over seeds for each (period, prob) cell.
    Returns one row per cell with mean / std columns.
    """
    agg = (
        df.groupby(["period", "prob"], sort=False)
        .agg(
            mean_speed_kmh   = ("mean_speed_kmh",  "mean"),
            std_speed_kmh    = ("mean_speed_kmh",  "std"),
            mean_speed_ms    = ("mean_speed_ms",   "mean"),
            mean_speed_ratio = ("mean_speed_ratio", "mean"),
            std_speed_ratio  = ("mean_speed_ratio", "std"),
            mean_throughput  = ("mean_throughput",  "mean"),
            mean_accidents   = ("n_accidents",      "mean"),
            mean_ai          = ("ai",               "mean"),
            std_ai           = ("ai",               "std"),
            n_rows           = ("seed",             "count"),
        )
        .reset_index()
    )

    # Traffic intensity proxy: vehicles inserted per hour
    agg["insertion_rate"] = 3600.0 / agg["period"]

    return agg


# ---------------------------------------------------------------------------
# Helper: assign a display colour and label to each prob value
# ---------------------------------------------------------------------------

def _prob_palette(probs: list[float]) -> dict[float, tuple[str, str]]:
    """
    Return {prob: (colour, label)} for a list of unique prob values.
    Baseline (prob==0) always gets the grey colour.
    """
    palette: dict[float, tuple[str, str]] = {}
    colour_idx = 1   # skip index 0 (grey) for non-zero probs

    for p in sorted(probs):
        if p == 0.0:
            palette[p] = (PROB_COLOURS[0], "Baseline  (no accidents)")
        else:
            palette[p] = (
                PROB_COLOURS[colour_idx % len(PROB_COLOURS)],
                f"P = {p:.1e}",
            )
            colour_idx += 1

    return palette


# ---------------------------------------------------------------------------
# Figure 1 — Speed–flow (MFD-style) curves
# ---------------------------------------------------------------------------

def fig_speed_vs_load(agg: pd.DataFrame, out_path: str):
    """
    Network speed vs traffic insertion rate, one curve per accident-probability level.
    Left panel : absolute speed (km/h)
    Right panel: speed ratio (normalised to free-flow baseline)
    """
    probs   = sorted(agg["prob"].unique())
    palette = _prob_palette(probs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Macroscopic Fundamental Diagram  ·  Sioux Falls Network",
        fontsize=13, fontweight="bold",
    )

    # ── Left: absolute speed ────────────────────────────────────────────────
    ax1.set_title("Mean Network Speed  vs  Traffic Load", fontsize=11)
    ax1.set_xlabel("Vehicle Insertion Rate  (veh / hour)")
    ax1.set_ylabel("Mean Network Speed  (km/h)")
    ax1.grid(True)

    for prob in probs:
        sub       = agg[agg["prob"] == prob].sort_values("insertion_rate")
        colour, lbl = palette[prob]
        ls        = "--" if prob == 0.0 else "-"
        lw        = 2.4 if prob == 0.0 else 1.8

        ax1.plot(
            sub["insertion_rate"], sub["mean_speed_kmh"],
            color=colour, lw=lw, ls=ls, label=lbl, zorder=3,
        )
        if "std_speed_kmh" in sub.columns:
            lo = sub["mean_speed_kmh"] - sub["std_speed_kmh"].fillna(0)
            hi = sub["mean_speed_kmh"] + sub["std_speed_kmh"].fillna(0)
            ax1.fill_between(sub["insertion_rate"], lo, hi,
                             color=colour, alpha=0.13)

    ax1.legend(fontsize=8, loc="lower left")

    # ── Right: speed ratio ─────────────────────────────────────────────────
    ax2.set_title("Speed Ratio  vs  Traffic Load  (normalised)", fontsize=11)
    ax2.set_xlabel("Vehicle Insertion Rate  (veh / hour)")
    ax2.set_ylabel("Speed Ratio  (1.0 = free-flow)")
    ax2.grid(True)

    # Reference threshold lines
    x_max = agg["insertion_rate"].max()
    for threshold, label_str, col in [(0.8, "−20 %", "#f0883e"),
                                       (0.6, "−40 %", "#ff7b72")]:
        ax2.axhline(threshold, color=col, lw=1.0, ls=":", alpha=0.7)
        ax2.text(x_max * 0.97, threshold + 0.01, label_str,
                 color=col, fontsize=8, ha="right", va="bottom")

    ax2.axhline(1.0, color=GRID_COL, lw=0.8, ls=":")

    for prob in probs:
        sub         = agg[agg["prob"] == prob].sort_values("insertion_rate")
        colour, lbl = palette[prob]
        ls          = "--" if prob == 0.0 else "-"
        lw          = 2.4 if prob == 0.0 else 1.8

        ax2.plot(
            sub["insertion_rate"], sub["mean_speed_ratio"],
            color=colour, lw=lw, ls=ls, label=lbl, zorder=3,
        )
        if "std_speed_ratio" in sub.columns:
            lo = sub["mean_speed_ratio"] - sub["std_speed_ratio"].fillna(0)
            hi = sub["mean_speed_ratio"] + sub["std_speed_ratio"].fillna(0)
            ax2.fill_between(sub["insertion_rate"], lo, hi,
                             color=colour, alpha=0.13)

    ax2.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Antifragility Index vs traffic load
# ---------------------------------------------------------------------------

def fig_ai_vs_load(agg: pd.DataFrame, out_path: str):
    """
    AI vs vehicle insertion rate — one line per accident-probability level,
    with ±σ shaded ribbon.  AI regime zones drawn as horizontal bands.
    """
    agg_acc = agg[agg["mean_accidents"] > 0].copy()
    if agg_acc.empty:
        print("  [skip] fig2 — no accident data in sweep")
        return

    probs   = sorted(agg_acc["prob"].unique())
    palette = _prob_palette(probs)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title(
        "Antifragility Index  vs  Traffic Load  ·  Sioux Falls",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Vehicle Insertion Rate  (veh / hour)", fontsize=11)
    ax.set_ylabel("Antifragility Index  (AI)", fontsize=11)
    ax.grid(True)

    # AI zone horizontal bands
    for lo, hi, col, _ in AI_ZONES:
        ax.axhspan(lo, hi, color=col, alpha=0.07, zorder=0)

    ax.axhline(0.0, color=TEXT_COL, lw=0.7, ls=":", zorder=1)

    line_handles = []
    for prob in probs:
        sub         = agg_acc[agg_acc["prob"] == prob].sort_values("insertion_rate")
        if sub.empty:
            continue
        colour, lbl = palette[prob]

        (ln,) = ax.plot(
            sub["insertion_rate"], sub["mean_ai"],
            color=colour, lw=2.2, marker="o", ms=5,
            label=lbl, zorder=3,
        )
        line_handles.append(ln)

        if "std_ai" in sub.columns:
            lo_r = sub["mean_ai"] - sub["std_ai"].fillna(0)
            hi_r = sub["mean_ai"] + sub["std_ai"].fillna(0)
            ax.fill_between(sub["insertion_rate"], lo_r, hi_r,
                            color=colour, alpha=0.18)

    # Combined legend (prob lines + regime patches)
    zone_patches = [
        mpatches.Patch(color=col, alpha=0.5, label=lbl)
        for _, _, col, lbl in AI_ZONES
    ]
    ax.legend(handles=line_handles + zone_patches, fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — 2-D heatmaps
# ---------------------------------------------------------------------------

def _annotate_cells(
    ax,
    mat: np.ndarray,
    fmt: str = ".2f",
    na_str: str = "—",
):
    """Overlay numeric text on a heatmap cell matrix."""
    rows, cols = mat.shape
    vmin = np.nanmin(mat)
    vmax = np.nanmax(mat)
    span = max(vmax - vmin, 1e-9)

    for i in range(rows):
        for j in range(cols):
            val = mat[i, j]
            if np.isnan(val):
                txt   = na_str
                color = "#8b949e"
            else:
                norm  = (val - vmin) / span
                txt   = format(val, fmt)
                # dark text on bright cells, light on dark
                color = "#000000" if norm > 0.60 else TEXT_COL
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7.5, color=color)


def fig_heatmaps(agg: pd.DataFrame, out_path: str):
    """
    Two side-by-side heatmaps over the (insertion period × accident prob) grid:
      Left  — mean speed ratio
      Right — mean Antifragility Index
    Rows   = insertion periods (high-load at bottom)
    Columns = accident probability levels
    """
    periods_asc  = sorted(agg["period"].unique())          # 0.5 … 5.0
    periods_desc = list(reversed(periods_asc))              # displayed top-to-bottom (low period = high load at bottom)
    probs_sorted = sorted(agg["prob"].unique())

    def _build_matrix(col: str) -> np.ndarray:
        mat = np.full((len(periods_desc), len(probs_sorted)), np.nan)
        for i, per in enumerate(periods_desc):
            for j, prob in enumerate(probs_sorted):
                mask = (agg["period"] == per) & (agg["prob"] == prob)
                if mask.any():
                    val = agg.loc[mask, col].values[0]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        mat[i, j] = float(val)
        return mat

    speed_mat = _build_matrix("mean_speed_ratio")
    ai_mat    = _build_matrix("mean_ai")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Performance Heatmaps  ·  Sioux Falls  "
        "(insertion period  ×  accident probability)",
        fontsize=12, fontweight="bold",
    )

    prob_labels   = [
        "Baseline" if p == 0 else f"{p:.1e}"
        for p in probs_sorted
    ]
    period_labels = [f"{p:.2f}s" for p in periods_desc]

    x_label = "Accident Probability"
    y_label = "Insertion Period  [↑ lower load → ↓ higher load]"

    # ── Speed ratio ──────────────────────────────────────────────────────────
    im1 = ax1.imshow(
        speed_mat, aspect="auto", origin="upper",
        cmap="RdYlGn", vmin=0.50, vmax=1.05,
    )
    ax1.set_title("Mean Speed Ratio  (1.0 = free-flow)", fontsize=11)
    ax1.set_xticks(range(len(probs_sorted)))
    ax1.set_xticklabels(prob_labels, rotation=35, ha="right", fontsize=8)
    ax1.set_yticks(range(len(periods_desc)))
    ax1.set_yticklabels(period_labels, fontsize=8)
    ax1.set_xlabel(x_label, fontsize=9)
    ax1.set_ylabel(y_label, fontsize=9)
    _annotate_cells(ax1, speed_mat, fmt=".2f")
    plt.colorbar(im1, ax=ax1, shrink=0.82, label="Speed Ratio")

    # ── Antifragility Index ──────────────────────────────────────────────────
    ai_abs = np.nanmax(np.abs(ai_mat)) if not np.all(np.isnan(ai_mat)) else 0.1
    ai_abs = max(ai_abs, 0.05)

    im2 = ax2.imshow(
        ai_mat, aspect="auto", origin="upper",
        cmap="RdYlGn", vmin=-ai_abs, vmax=ai_abs,
    )
    ax2.set_title("Antifragility Index  (AI)", fontsize=11)
    ax2.set_xticks(range(len(probs_sorted)))
    ax2.set_xticklabels(prob_labels, rotation=35, ha="right", fontsize=8)
    ax2.set_yticks(range(len(periods_desc)))
    ax2.set_yticklabels(period_labels, fontsize=8)
    ax2.set_xlabel(x_label, fontsize=9)
    ax2.set_ylabel(y_label, fontsize=9)
    _annotate_cells(ax2, ai_mat, fmt=".3f", na_str="—")
    plt.colorbar(im2, ax=ax2, shrink=0.82, label="AI")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Figure 4 — Phase diagram
# ---------------------------------------------------------------------------

def _ai_regime_colour(ai) -> str:
    """Map an AI value to its regime colour."""
    if ai is None or (isinstance(ai, float) and np.isnan(ai)):
        return "#8b949e"   # no accident data
    if ai > 0.05:
        return "#238636"   # antifragile
    if ai > -0.05:
        return "#1f6feb"   # resilient
    if ai > -0.20:
        return "#d29922"   # fragile
    return "#da3633"       # brittle


def fig_phase_diagram(agg: pd.DataFrame, out_path: str):
    """
    Phase diagram:
      x-axis — vehicle insertion rate (proxy for traffic demand)
      y-axis — accident base-probability (log scale; baseline row treated specially)

    Each bubble represents one (period, prob) cell.
    Colour = AI resilience regime.
    Bubble size ∝ mean accident count.
    """
    fig, (ax_main, ax_base) = plt.subplots(
        2, 1, figsize=(11, 9),
        gridspec_kw={"height_ratios": [6, 1], "hspace": 0.12},
    )

    fig.suptitle(
        "Network Resilience Phase Diagram  ·  Sioux Falls",
        fontsize=13, fontweight="bold",
    )

    # ── Split data into accident rows and baseline row ─────────────────────
    agg_acc  = agg[agg["prob"] > 0].copy()
    agg_base = agg[agg["prob"] == 0].copy()

    # ── Main panel: accident probability levels on a log y-axis ──────────
    ax_main.set_xlabel("Vehicle Insertion Rate  (veh / hour)", fontsize=11)
    ax_main.set_ylabel("Accident Probability  (base_probability)", fontsize=11)
    ax_main.set_yscale("log")
    ax_main.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0e}")
    )
    ax_main.grid(True, which="both", alpha=0.25)

    for _, row in agg_acc.iterrows():
        ai_val  = row["mean_ai"] if row["mean_accidents"] > 0 else None
        colour  = _ai_regime_colour(ai_val)
        size    = max(60, row["mean_accidents"] * 28)

        ax_main.scatter(
            row["insertion_rate"], row["prob"],
            c=colour, s=size,
            edgecolors=DARK_BG, linewidths=0.8,
            alpha=0.92, zorder=3,
        )
        if ai_val is not None and not np.isnan(ai_val):
            ax_main.annotate(
                f"{ai_val:+.3f}",
                (row["insertion_rate"], row["prob"]),
                textcoords="offset points", xytext=(0, 9),
                fontsize=7, ha="center", color=TEXT_COL,
            )

    # ── Baseline strip (prob == 0) ──────────────────────────────────────
    ax_base.set_xlabel("")
    ax_base.set_ylabel("Baseline\n(no acc.)", fontsize=9, color=TEXT_COL)
    ax_base.set_yticks([])
    ax_base.grid(True, axis="x", alpha=0.25)
    ax_base.set_xlim(ax_main.get_xlim())

    for _, row in agg_base.iterrows():
        # Baseline: colour by speed ratio (green if fast, red if slow)
        ratio  = row["mean_speed_ratio"] or 1.0
        colour = "#238636" if ratio > 0.9 else "#d29922" if ratio > 0.7 else "#da3633"
        ax_base.scatter(
            row["insertion_rate"], 0.5,
            c=colour, s=100,
            edgecolors=DARK_BG, linewidths=0.8,
            alpha=0.9, zorder=3,
        )
        ax_base.annotate(
            f"SR={ratio:.2f}",
            (row["insertion_rate"], 0.5),
            textcoords="offset points", xytext=(0, 8),
            fontsize=7, ha="center", color=TEXT_COL,
        )

    ax_base.set_ylim(0, 1)

    # Share x-axis between panels
    ax_main.sharex(ax_base)

    # ── Legend ──────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color="#238636", label="Antifragile   (AI > +0.05)"),
        mpatches.Patch(color="#1f6feb", label="Resilient     (|AI| ≤ 0.05)"),
        mpatches.Patch(color="#d29922", label="Fragile       (−0.20 < AI ≤ −0.05)"),
        mpatches.Patch(color="#da3633", label="Brittle       (AI ≤ −0.20)"),
        mpatches.Patch(color="#8b949e", label="No AI data"),
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=TEXT_COL, markersize=9,
            label="Bubble size ∝ mean accidents",
        ),
    ]
    ax_main.legend(handles=legend_items, fontsize=8, loc="upper left")

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser(
        description="Visualise SAS failure-point sweep results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--csv",
        default=os.path.join(here, "results", "sweep", "sweep_results.csv"),
        help="Path to sweep_results.csv produced by experiment_sweep.py",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for figures (default: <csv-dir>/figures/)",
    )
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.csv)), "figures"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Apply dark theme globally
    plt.rcParams.update(_RCPARAMS)

    print(f"\n  SAS — Sweep Visualiser")
    print(f"  {'─'*38}")
    print(f"  CSV     : {args.csv}")
    print(f"  Figures : {out_dir}\n")

    df  = load_sweep(args.csv)
    agg = aggregate(df)

    fig_speed_vs_load(agg, os.path.join(out_dir, "fig1_speed_vs_load.png"))
    fig_ai_vs_load   (agg, os.path.join(out_dir, "fig2_ai_vs_load.png"))
    fig_heatmaps     (agg, os.path.join(out_dir, "fig3_heatmaps.png"))
    fig_phase_diagram(agg, os.path.join(out_dir, "fig4_phase_diagram.png"))

    print(f"\n  Done — {out_dir}\n")


if __name__ == "__main__":
    main()
