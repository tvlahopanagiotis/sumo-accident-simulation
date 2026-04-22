"""
visualise_sweep.py
==================
Failure-point visualisation for the SAS parameter-grid sweep.

Reads  :  results/sweep/sweep_results.csv

Writes :  results/sweep/figures/
              fig1_speed_vs_load.png    — MFD-style speed curves per prob level
              fig2_ai_vs_load.png       — AI vs traffic load with error bars
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
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
# IEEE/Nature-compatible: white background, explicit markers, no glow.
# Use the default matplotlib font stack (Helvetica / DejaVu Sans).

RCPARAMS = {
    "figure.facecolor":       "white",
    "axes.facecolor":         "white",
    "axes.edgecolor":         "#333333",
    "axes.labelcolor":        "#111111",
    "axes.linewidth":         0.8,
    "axes.spines.top":        False,
    "axes.spines.right":      False,
    "axes.grid":              True,
    "grid.color":             "#cccccc",
    "grid.linewidth":         0.5,
    "grid.linestyle":         "--",
    "xtick.color":            "#111111",
    "xtick.major.width":      0.8,
    "ytick.color":            "#111111",
    "ytick.major.width":      0.8,
    "text.color":             "#111111",
    "legend.facecolor":       "white",
    "legend.edgecolor":       "#aaaaaa",
    "legend.framealpha":      0.9,
    "font.size":              10,
    "axes.titlesize":         11,
    "axes.labelsize":         10,
    "legend.fontsize":        8.5,
    "xtick.labelsize":        9,
    "ytick.labelsize":        9,
    "lines.linewidth":        1.6,
    "lines.markersize":       5,
    "figure.dpi":             150,
    "savefig.dpi":            200,
    "savefig.bbox":           "tight",
    "savefig.facecolor":      "white",
}

# Colourblind-safe palette (Wong 2011), one entry per prob level (up to 6)
# Ordered: grey → blue → orange → green → red → purple
PROB_COLOURS = [
    "#777777",   # grey   — baseline (no accidents)
    "#0072B2",   # blue
    "#E69F00",   # orange
    "#009E73",   # green
    "#D55E00",   # red
    "#CC79A7",   # pink/purple
]

MARKERS = ["s", "o", "^", "D", "v", "P"]   # one per prob level

# AI regime colours (pastel fills for zone shading)
AI_ZONE_FILLS = [
    ( 0.05,  1.00, "#d4edda", "Antifragile   (AI > +0.05)"),
    (-0.05,  0.05, "#d1ecf1", "Resilient     (|AI| ≤ 0.05)"),
    (-0.20, -0.05, "#fff3cd", "Fragile       (AI > −0.20)"),
    (-1.00, -0.20, "#f8d7da", "Brittle       (AI ≤ −0.20)"),
]


# ---------------------------------------------------------------------------
# Data loading & aggregation
# ---------------------------------------------------------------------------

def load_sweep(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        sys.exit(f"ERROR: sweep CSV not found — {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"period", "prob", "seed", "n_accidents"}
    if missing := required - set(df.columns):
        sys.exit(f"ERROR: CSV missing columns: {missing}")

    numeric = [
        "period", "prob", "seed", "n_accidents",
        "mean_speed_ms", "mean_speed_kmh",
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
    agg = (
        df.groupby(["period", "prob"], sort=False)
        .agg(
            mean_speed_kmh   = ("mean_speed_kmh",   "mean"),
            std_speed_kmh    = ("mean_speed_kmh",   "std"),
            mean_speed_ratio = ("mean_speed_ratio",  "mean"),
            std_speed_ratio  = ("mean_speed_ratio",  "std"),
            mean_throughput  = ("mean_throughput",   "mean"),
            mean_accidents   = ("n_accidents",       "mean"),
            mean_ai          = ("ai",                "mean"),
            std_ai           = ("ai",                "std"),
            n_rows           = ("seed",              "count"),
        )
        .reset_index()
    )
    agg["insertion_rate"] = 3600.0 / agg["period"]
    return agg


def _palette(probs):
    """Return {prob: (colour, marker, label)}."""
    result = {}
    ci = 1
    for p in sorted(probs):
        if p == 0.0:
            result[p] = (PROB_COLOURS[0], MARKERS[0], "Baseline (no accidents)")
        else:
            result[p] = (PROB_COLOURS[ci % len(PROB_COLOURS)],
                         MARKERS[ci % len(MARKERS)],
                         f"$P$ = {p:.0e}")
            ci += 1
    return result


# ---------------------------------------------------------------------------
# Figure 1 — MFD-style speed curves
# ---------------------------------------------------------------------------

def fig_speed_vs_load(agg: pd.DataFrame, out_path: str):
    """Mean speed and speed ratio vs insertion rate — one curve per prob level."""
    probs   = sorted(agg["prob"].unique())
    palette = _palette(probs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    fig.suptitle("Macroscopic Fundamental Diagram — Sioux Falls Network",
                 fontsize=12, fontweight="bold", y=1.01)

    # ── Left: absolute speed ────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_xlabel("Vehicle Insertion Rate  (veh h⁻¹)")
    ax1.set_ylabel("Mean Network Speed  (km h⁻¹)")

    for prob in probs:
        sub          = agg[agg["prob"] == prob].sort_values("insertion_rate")
        col, mrk, lbl = palette[prob]
        ls           = (0, (4, 2)) if prob == 0.0 else "solid"

        ax1.plot(sub["insertion_rate"], sub["mean_speed_kmh"],
                 color=col, ls=ls, marker=mrk, label=lbl, zorder=3)

        if "std_speed_kmh" in sub.columns:
            se = sub["std_speed_kmh"].fillna(0) / np.sqrt(sub["n_rows"].clip(lower=1))
            ax1.fill_between(sub["insertion_rate"],
                             sub["mean_speed_kmh"] - se,
                             sub["mean_speed_kmh"] + se,
                             color=col, alpha=0.18, lw=0)

    ax1.legend(loc="upper right")

    # ── Right: speed ratio ──────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_xlabel("Vehicle Insertion Rate  (veh h⁻¹)")
    ax2.set_ylabel("Speed Ratio  (relative to free-flow)")

    # Reference lines
    x_max = agg["insertion_rate"].max()
    for thr, lbl_str, col_str in [
        (0.80, "−20 %", "#888888"),
        (0.60, "−40 %", "#555555"),
    ]:
        ax2.axhline(thr, color=col_str, lw=0.8, ls=":", zorder=0)
        ax2.text(x_max * 0.99, thr + 0.01, lbl_str,
                 color=col_str, fontsize=8, ha="right", va="bottom")

    ax2.axhline(1.0, color="#333333", lw=0.7, ls=":", zorder=0)

    for prob in probs:
        sub          = agg[agg["prob"] == prob].sort_values("insertion_rate")
        col, mrk, lbl = palette[prob]
        ls           = (0, (4, 2)) if prob == 0.0 else "solid"

        ax2.plot(sub["insertion_rate"], sub["mean_speed_ratio"],
                 color=col, ls=ls, marker=mrk, label=lbl, zorder=3)

        if "std_speed_ratio" in sub.columns:
            se = sub["std_speed_ratio"].fillna(0) / np.sqrt(sub["n_rows"].clip(lower=1))
            ax2.fill_between(sub["insertion_rate"],
                             sub["mean_speed_ratio"] - se,
                             sub["mean_speed_ratio"] + se,
                             color=col, alpha=0.18, lw=0)

    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — AI vs traffic load
# ---------------------------------------------------------------------------

def fig_ai_vs_load(agg: pd.DataFrame, out_path: str):
    """AI vs insertion rate with ±1 SE error bars and regime zone shading."""
    agg_acc = agg[agg["mean_accidents"] > 0].copy()
    if agg_acc.empty:
        print("  [skip] fig2 — no accident data in sweep")
        return

    probs   = sorted(agg_acc["prob"].unique())
    palette = _palette(probs)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_title("Antifragility Index vs Traffic Load — Sioux Falls",
                 fontweight="bold")
    ax.set_xlabel("Vehicle Insertion Rate  (veh h⁻¹)")
    ax.set_ylabel("Antifragility Index  (AI)")

    # Regime zone shading
    for lo, hi, fill_col, _ in AI_ZONE_FILLS:
        ax.axhspan(lo, hi, color=fill_col, alpha=0.55, lw=0, zorder=0)

    ax.axhline(0.0, color="#333333", lw=0.8, ls="--", zorder=1)

    line_handles = []
    for prob in probs:
        sub          = agg_acc[agg_acc["prob"] == prob].sort_values("insertion_rate")
        if sub.empty:
            continue
        col, mrk, lbl = palette[prob]

        if "std_ai" in sub.columns:
            se  = sub["std_ai"].fillna(0) / np.sqrt(sub["n_rows"].clip(lower=1))
            ax.errorbar(
                sub["insertion_rate"], sub["mean_ai"],
                yerr=se,
                color=col, marker=mrk, capsize=3, capthick=0.8,
                lw=1.6, elinewidth=0.9, label=lbl, zorder=3,
            )
        else:
            ax.plot(sub["insertion_rate"], sub["mean_ai"],
                    color=col, marker=mrk, label=lbl, zorder=3)

        (ln,) = ax.plot([], [], color=col, marker=mrk, label=lbl)
        line_handles.append(ln)

    zone_patches = [
        mpatches.Patch(color=fill_col, alpha=0.7, label=lbl)
        for _, _, fill_col, lbl in AI_ZONE_FILLS
    ]
    all_handles = [h for h in ax.get_legend_handles_labels()[0]
                   if not isinstance(h, mpatches.Patch)] + zone_patches
    ax.legend(handles=all_handles, fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — 2-D heatmaps
# ---------------------------------------------------------------------------

def _annotate_cells(ax, mat, fmt=".2f", na_str="—", text_col="#111111"):
    rows, cols = mat.shape
    for i in range(rows):
        for j in range(cols):
            val = mat[i, j]
            txt = na_str if np.isnan(val) else format(val, fmt)
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color=text_col)


def fig_heatmaps(agg: pd.DataFrame, out_path: str):
    periods_desc = sorted(agg["period"].unique(), reverse=True)
    probs_sorted = sorted(agg["prob"].unique())

    def _pivot(col):
        mat = np.full((len(periods_desc), len(probs_sorted)), np.nan)
        for i, per in enumerate(periods_desc):
            for j, prob in enumerate(probs_sorted):
                mask = (agg["period"] == per) & (agg["prob"] == prob)
                if mask.any():
                    v = agg.loc[mask, col].values[0]
                    if v is not None and not np.isnan(float(v)):
                        mat[i, j] = float(v)
        return mat

    speed_mat = _pivot("mean_speed_ratio")
    ai_mat    = _pivot("mean_ai")

    prob_labels   = ["Baseline" if p == 0 else f"{p:.0e}" for p in probs_sorted]
    period_labels = [f"{p:.2f} s" for p in periods_desc]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Performance Heatmaps — Sioux Falls  "
        "(insertion period × accident probability)",
        fontweight="bold", y=1.02,
    )

    # Speed ratio
    im1 = ax1.imshow(speed_mat, aspect="auto", origin="upper",
                     cmap="RdYlGn", vmin=0.50, vmax=1.05)
    ax1.set_title("Mean Speed Ratio  (1.0 = free-flow)")
    ax1.set_xticks(range(len(probs_sorted)))
    ax1.set_xticklabels(prob_labels, rotation=30, ha="right")
    ax1.set_yticks(range(len(periods_desc)))
    ax1.set_yticklabels(period_labels)
    ax1.set_xlabel("Accident Probability")
    ax1.set_ylabel("Insertion Period  [↑ lower load]")
    _annotate_cells(ax1, speed_mat, fmt=".2f")
    plt.colorbar(im1, ax=ax1, shrink=0.82, label="Speed Ratio")

    # AI
    ai_abs = max(np.nanmax(np.abs(ai_mat)) if not np.all(np.isnan(ai_mat)) else 0.1, 0.05)
    im2 = ax2.imshow(ai_mat, aspect="auto", origin="upper",
                     cmap="RdYlGn", vmin=-ai_abs, vmax=ai_abs)
    ax2.set_title("Antifragility Index  (AI)")
    ax2.set_xticks(range(len(probs_sorted)))
    ax2.set_xticklabels(prob_labels, rotation=30, ha="right")
    ax2.set_yticks(range(len(periods_desc)))
    ax2.set_yticklabels(period_labels)
    ax2.set_xlabel("Accident Probability")
    ax2.set_ylabel("Insertion Period  [↑ lower load]")
    _annotate_cells(ax2, ai_mat, fmt=".3f", na_str="—")
    plt.colorbar(im2, ax=ax2, shrink=0.82, label="AI")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Figure 4 — Phase diagram
# ---------------------------------------------------------------------------

def _ai_colour(ai):
    if ai is None or (isinstance(ai, float) and np.isnan(ai)):
        return "#aaaaaa"
    if ai > 0.05:  return "#2ca02c"
    if ai > -0.05: return "#1f77b4"
    if ai > -0.20: return "#ff7f0e"
    return "#d62728"


def fig_phase_diagram(agg: pd.DataFrame, out_path: str):
    """
    Phase diagram: insertion rate (x) × accident probability (y, log scale).
    Bubble colour = AI regime; size ∝ mean accident count.
    Lower strip shows baseline speed-ratio (no accidents).
    """
    agg_acc  = agg[agg["prob"] > 0].copy()
    agg_base = agg[agg["prob"] == 0].copy()

    fig, (ax_main, ax_base) = plt.subplots(
        2, 1, figsize=(9, 7.5),
        gridspec_kw={"height_ratios": [6, 1.2], "hspace": 0.10},
    )
    fig.suptitle("Network Resilience Phase Diagram — Sioux Falls",
                 fontsize=12, fontweight="bold")

    # ── Main panel ─────────────────────────────────────────────────────────
    ax_main.set_ylabel(r"Accident Probability  (base\_probability)")
    ax_main.set_yscale("log")
    ax_main.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0e}")
    )

    for _, row in agg_acc.iterrows():
        ai_val = row["mean_ai"] if row["mean_accidents"] > 0 else None
        col    = _ai_colour(ai_val)
        size   = max(50, row["mean_accidents"] * 22)

        ax_main.scatter(
            row["insertion_rate"], row["prob"],
            c=col, s=size, edgecolors="#333333",
            linewidths=0.6, alpha=0.88, zorder=3,
        )
        if ai_val is not None and not np.isnan(ai_val):
            ax_main.annotate(
                f"{ai_val:+.3f}",
                (row["insertion_rate"], row["prob"]),
                textcoords="offset points", xytext=(0, 8),
                fontsize=7.5, ha="center",
            )

    # ── Baseline strip ─────────────────────────────────────────────────────
    ax_base.set_xlabel("Vehicle Insertion Rate  (veh h⁻¹)")
    ax_base.set_ylabel("Baseline\n(no acc.)", fontsize=8)
    ax_base.set_yticks([])

    for _, row in agg_base.iterrows():
        ratio = row["mean_speed_ratio"] if not np.isnan(row["mean_speed_ratio"]) else 1.0
        col   = "#2ca02c" if ratio > 0.90 else "#ff7f0e" if ratio > 0.70 else "#d62728"
        ax_base.scatter(row["insertion_rate"], 0.5,
                        c=col, s=80, edgecolors="#333333",
                        linewidths=0.6, alpha=0.88, zorder=3)
        ax_base.annotate(
            f"SR={ratio:.2f}",
            (row["insertion_rate"], 0.5),
            textcoords="offset points", xytext=(0, 7),
            fontsize=7.5, ha="center",
        )
    ax_base.set_ylim(0, 1)
    ax_main.sharex(ax_base)

    # ── Legend ─────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color="#2ca02c", label="Antifragile   (AI > +0.05)"),
        mpatches.Patch(color="#1f77b4", label="Resilient     (|AI| ≤ 0.05)"),
        mpatches.Patch(color="#ff7f0e", label="Fragile       (−0.20 < AI ≤ −0.05)"),
        mpatches.Patch(color="#d62728", label="Brittle       (AI ≤ −0.20)"),
        mpatches.Patch(color="#aaaaaa", label="No AI data"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555",
               markersize=8, markeredgecolor="#333333",
               label="Bubble size ∝ mean accidents"),
    ]
    ax_main.legend(handles=legend_items, fontsize=8, loc="upper left",
                   framealpha=0.9)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser(
        description="Visualise SAS failure-point sweep results (publication style)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--csv",
        default=os.path.join(here, "results", "sweep", "sweep_results.csv"),
    )
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.csv)), "figures"
    )
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update(RCPARAMS)

    print(f"\n  SAS — Sweep Visualiser  (academic style)")
    print(f"  {'─'*40}")
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
