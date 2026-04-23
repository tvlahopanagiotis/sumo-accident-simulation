"""
analyse_batch.py
================

Scientific analysis of Thessaloniki batch simulation runs.

Produces five per-batch publication-quality figure sets:

  Figure 1 — resilience_statistics.png
      4-panel: AI distribution · accident count distribution
               AI vs accidents scatter · interpretation categories

  Figure 2 — network_dynamics.png
      3-panel: vehicle count ensemble · speed ensemble (5-min rolling avg)
               accidents ensemble  (per-run traces + mean ± IQR)

  Figure 3 — accident_characteristics.png
      4-panel: severity mix · duration by severity · trigger timing
               impact (vehicles affected) by severity

  Figure 4 — spatial_heatmap.png
      2-panel: 2D KDE heatmap + severity scatter
               (with Thessaloniki road network underlay)

  Figure 5 — per_event_ai.png
      2-panel: per-event AI by severity · pre/post speed

Plus four comparative figure sets (when two batch dirs are supplied):

  comparative_overview.png         — AI distributions, accident counts, categories
  comparative_dynamics.png         — ensemble dynamics side-by-side
  comparative_spatial.png          — KDE heatmaps for both batches on network
  comparative_per_event_ai.png     — per-event AI and speed comparison

Usage:
    # Single batch (with fixes)
    sas-analyse-batch --batch-dir results/Thessaloniki_Batch_2026-03-05_16:53

    # Both batches + comparative
    sas-analyse-batch \\
        --batch-dir results/Thessaloniki_Batch_2026-03-05_16:53 \\
        --compare-dir results/Thessaloniki_Batch_2026-03-05_20:31
"""

from __future__ import annotations

import argparse
import json
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams.update({
    "font.family":        "sans-serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})

DPI = 200

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BATCH_A_DEFAULT = PROJECT_ROOT / "results" / "Thessaloniki_Batch_2026-03-05_16:53"
BATCH_B_DEFAULT = PROJECT_ROOT / "results" / "Thessaloniki_Batch_2026-03-05_20:31"
NET_XML_DEFAULT = PROJECT_ROOT / "data" / "cities" / "thessaloniki" / "network" / "thessaloniki.net.xml"

AI_THRESHOLDS = {
    "BRITTLE":     (-np.inf, -0.20),
    "FRAGILE":     (-0.20,   -0.05),
    "RESILIENT":   (-0.05,    0.05),
    "ANTIFRAGILE": (0.05,    np.inf),
}
AI_COLORS = {
    "BRITTLE":     "#8b0000",
    "FRAGILE":     "#e74c3c",
    "RESILIENT":   "#3498db",
    "ANTIFRAGILE": "#27ae60",
}
SEVERITY_ORDER  = ["MINOR", "MODERATE", "MAJOR", "CRITICAL"]
SEVERITY_COLORS = {
    "MINOR":    "#2ecc71",
    "MODERATE": "#f39c12",
    "MAJOR":    "#e74c3c",
    "CRITICAL": "#8b0000",
}

# Two-batch palette (used in comparative figures)
BATCH_PALETTE = {
    "A": "#1565C0",   # deep blue
    "B": "#BF360C",   # deep orange
}
BATCH_FILL = {
    "A": "#90CAF9",
    "B": "#FFAB91",
}

# Speed rolling-average window (5 minutes = 5 rows at 60-s metric interval)
SPEED_SMOOTH_WINDOW = 5


# ---------------------------------------------------------------------------
# Network loading
# ---------------------------------------------------------------------------

def load_network_shapes(net_xml_path: Path) -> tuple[list[np.ndarray], tuple[float, ...]]:
    """
    Parse a SUMO .net.xml file and return road geometry for background plotting.

    Returns
    -------
    segments : list[np.ndarray]
        Each element is shape (N, 2) — (x, y) coordinate pairs for one lane.
    extent : (xmin, xmax, ymin, ymax)
        Bounding box of the full network.
    """
    tree = ET.parse(net_xml_path)
    root = tree.getroot()

    segments: list[np.ndarray] = []
    all_x: list[float] = []
    all_y: list[float] = []

    for edge in root.findall("edge"):
        # Skip internal junctions (their IDs start with ':')
        if edge.get("id", "").startswith(":"):
            continue
        # Use the rightmost (outermost) lane shape only — avoids overplotting
        lanes = edge.findall("lane")
        if not lanes:
            continue
        lane = lanes[0]
        shape_str = lane.get("shape", "")
        if not shape_str:
            continue
        try:
            coords = np.array(
                [list(map(float, pt.split(","))) for pt in shape_str.split()],
                dtype=float,
            )
        except ValueError:
            continue
        if len(coords) < 2:
            continue
        segments.append(coords)
        all_x.extend(coords[:, 0])
        all_y.extend(coords[:, 1])

    if not all_x:
        return [], (0, 1, 0, 1)

    return segments, (
        float(np.min(all_x)), float(np.max(all_x)),
        float(np.min(all_y)), float(np.max(all_y)),
    )


def draw_network_bg(
    ax: plt.Axes,
    segments: list[np.ndarray],
    extent: tuple[float, ...],
    *,
    color: str = "#90A4AE",
    alpha: float = 0.30,
    lw: float = 0.35,
    zorder: int = 0,
) -> None:
    """Render road network as a lightweight LineCollection background layer."""
    if not segments:
        return
    lc = LineCollection(segments, colors=color, alpha=alpha, linewidths=lw, zorder=zorder)
    ax.add_collection(lc)
    pad_x = (extent[1] - extent[0]) * 0.02
    pad_y = (extent[3] - extent[2]) * 0.02
    ax.set_xlim(extent[0] - pad_x, extent[1] + pad_x)
    ax.set_ylim(extent[2] - pad_y, extent[3] + pad_y)
    ax.set_aspect("equal")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_batch_data(batch_dir: Path) -> dict:
    """
    Load all per-run data from a batch directory.

    Returns a dict with:
        aggregate   – aggregate_summary.json contents
        runs        – list of per-run summaries (from aggregate)
        metrics     – DataFrame with all network_metrics.csv, stacked
        accidents   – DataFrame with all accident_reports.json, stacked
        ai_events   – DataFrame with all per-event AI data
        label       – short human-readable label (derived from dir name)
    """
    agg_path = batch_dir / "aggregate" / "aggregate_summary.json"
    with open(agg_path) as f:
        agg = json.load(f)

    run_dirs = sorted(d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith("run_"))

    metrics_frames: list[pd.DataFrame] = []
    accident_frames: list[pd.DataFrame] = []
    ai_event_frames: list[pd.DataFrame] = []

    for seed_idx, run_dir in enumerate(run_dirs):
        seed = int(run_dir.name.split("_")[1])

        csv_path = run_dir / "network_metrics.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["seed"]    = seed
            df["run_idx"] = seed_idx
            metrics_frames.append(df)

        acc_path = run_dir / "accident_reports.json"
        if acc_path.exists():
            with open(acc_path) as f:
                reports = json.load(f)
            if reports:
                df_acc = pd.json_normalize(reports)
                df_acc["seed"]    = seed
                df_acc["run_idx"] = seed_idx
                accident_frames.append(df_acc)

        ai_path = run_dir / "antifragility_index.json"
        if ai_path.exists():
            with open(ai_path) as f:
                ai_data = json.load(f)
            per_event = ai_data.get("per_event", [])
            if per_event:
                df_ai = pd.DataFrame(per_event)
                df_ai["seed"]    = seed
                df_ai["run_idx"] = seed_idx
                ai_event_frames.append(df_ai)

    metrics   = pd.concat(metrics_frames,  ignore_index=True) if metrics_frames  else pd.DataFrame()
    accidents = pd.concat(accident_frames, ignore_index=True) if accident_frames else pd.DataFrame()
    ai_events = pd.concat(ai_event_frames, ignore_index=True) if ai_event_frames else pd.DataFrame()

    # Derive a short label from the directory name, e.g. "16:53"
    parts = batch_dir.name.split("_")
    label = parts[-1] if len(parts) >= 1 else batch_dir.name

    return {
        "aggregate": agg["aggregate"],
        "runs":      agg["runs"],
        "metrics":   metrics,
        "accidents": accidents,
        "ai_events": ai_events,
        "label":     label,
        "batch_dir": batch_dir,
    }


def classify_ai(ai: float | None) -> str:
    if ai is None:
        return "N/A"
    for cat_label, (lo, hi) in AI_THRESHOLDS.items():
        if lo < ai <= hi:
            return cat_label
    return "BRITTLE" if ai <= -0.20 else "ANTIFRAGILE"


def _annotate_runs_df(data: dict) -> pd.DataFrame:
    """Return a DataFrame of per-run summaries with category column."""
    df = pd.DataFrame(data["runs"])
    df["category"] = df["antifragility_index"].apply(classify_ai)
    return df


# ---------------------------------------------------------------------------
# Shared pivot helper
# ---------------------------------------------------------------------------

def _pivot_and_smooth(
    metrics: pd.DataFrame,
    col: str,
    *,
    smooth_window: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pivot a metric column into shape (timesteps, n_runs).

    If smooth_window is given, applies a rolling mean per run (column).
    Returns (t_minutes, arr).
    """
    pv = metrics.pivot_table(index="timestamp_seconds", columns="run_idx", values=col)
    arr = pv.values.astype(float)
    t   = pv.index.values / 60.0

    if smooth_window and smooth_window > 1:
        smoothed = np.empty_like(arr)
        for j in range(arr.shape[1]):
            s = pd.Series(arr[:, j])
            smoothed[:, j] = s.rolling(window=smooth_window, min_periods=1).mean().values
        arr = smoothed

    return t, arr


# ---------------------------------------------------------------------------
# Figure 1 — Resilience statistics
# ---------------------------------------------------------------------------

def figure_resilience_statistics(data: dict, out_dir: Path) -> Path:
    """4-panel resilience overview."""
    df_runs  = _annotate_runs_df(data)
    agg      = data["aggregate"]
    ai_valid = df_runs["antifragility_index"].dropna()

    n        = int(agg["n_runs"])
    ai_mean  = float(agg["ai_mean"])
    ci_low   = float(agg["ai_ci_95_low"])
    ci_high  = float(agg["ai_ci_95_high"])
    acc_mean = float(agg["accident_mean"])
    acc_std  = float(agg["accident_std"])

    sw_stat, sw_p = stats.shapiro(ai_valid)
    both = df_runs.dropna(subset=["antifragility_index"])
    rho, p_rho = stats.spearmanr(both["total_accidents"], both["antifragility_index"])

    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # ── Panel A: AI distribution ─────────────────────────────────────────────
    ax_ai = fig.add_subplot(gs[0, 0])
    ax_ai.set_title("(a) Antifragility Index Distribution", fontweight="bold", pad=8)

    kde_x = np.linspace(ai_valid.min() - 0.05, ai_valid.max() + 0.05, 400)
    kde_y = stats.gaussian_kde(ai_valid, bw_method="silverman")(kde_x)

    for cat_label, lo, hi in [
        ("BRITTLE",    -1.0, -0.20),
        ("FRAGILE",   -0.20, -0.05),
        ("RESILIENT", -0.05,  0.05),
        ("ANTIFRAGILE", 0.05,  1.0),
    ]:
        mask = (kde_x >= lo) & (kde_x <= hi)
        if mask.any():
            ax_ai.fill_between(kde_x[mask], kde_y[mask], alpha=0.25,
                               color=AI_COLORS[cat_label], label=cat_label)

    ax_ai.plot(kde_x, kde_y, color="#2c3e50", lw=1.8)
    ax_ai.hist(ai_valid, bins=14, density=True, alpha=0.28,
               color="#7f8c8d", edgecolor="white")

    for thresh, ls in [(-0.20, "--"), (-0.05, ":"), (0.05, ":")]:
        ax_ai.axvline(thresh, color="#555", lw=0.8, linestyle=ls, alpha=0.7)

    ax_ai.axvline(ai_mean, color="#2c3e50", lw=2, label=f"Batch mean {ai_mean:.3f}")
    ax_ai.axvspan(ci_low, ci_high, alpha=0.12, color="#2c3e50",
                  label=f"95% CI [{ci_low:.3f}, {ci_high:.3f}]")
    ax_ai.set_xlabel("Antifragility Index (AI)")
    ax_ai.set_ylabel("Density")
    ax_ai.legend(fontsize=8, loc="upper left")
    ax_ai.text(0.98, 0.97,
               f"n={len(ai_valid)}\nShapiro–Wilk: W={sw_stat:.3f}, p={sw_p:.3f}",
               ha="right", va="top", transform=ax_ai.transAxes,
               fontsize=7.5, color="#555")

    # ── Panel B: Accident count distribution ─────────────────────────────────
    ax_acc = fig.add_subplot(gs[0, 1])
    ax_acc.set_title("(b) Accidents per Run", fontweight="bold", pad=8)

    acc_counts = df_runs["total_accidents"].values
    bins = np.arange(acc_counts.min() - 0.5, acc_counts.max() + 1.5, 1)
    ax_acc.hist(acc_counts, bins=bins, color="#3498db", edgecolor="white",
                alpha=0.75, label="Observed")

    k_range = np.arange(int(acc_counts.min()), int(acc_counts.max()) + 1)
    poisson_pmf = stats.poisson.pmf(k_range, mu=acc_mean) * n
    ax_acc.plot(k_range, poisson_pmf, "o--", color="#e74c3c", lw=1.5, ms=5,
                label=f"Poisson(λ={acc_mean:.1f})")
    ax_acc.axvline(acc_mean, color="#2c3e50", lw=2, label=f"Mean {acc_mean:.1f}")
    ax_acc.axvspan(acc_mean - acc_std, acc_mean + acc_std, alpha=0.10,
                   color="#2c3e50", label=f"±1σ ({acc_std:.2f})")
    ax_acc.set_xlabel("Total Accidents per Run")
    ax_acc.set_ylabel("Number of Runs")
    ax_acc.legend(fontsize=8)

    # Chi-square GoF
    obs_counts_per_k = [(acc_counts == k).sum() for k in k_range]
    exp_counts_per_k = [stats.poisson.pmf(k, mu=acc_mean) * n for k in k_range]
    obs_pool, exp_pool, buf_o, buf_e = [], [], 0, 0
    for o, e in zip(obs_counts_per_k, exp_counts_per_k, strict=True):
        buf_o += o
        buf_e += e
        if buf_e >= 5:
            obs_pool.append(buf_o)
            exp_pool.append(buf_e)
            buf_o, buf_e = 0, 0
    if buf_e > 0:
        if obs_pool:
            obs_pool[-1] += buf_o
            exp_pool[-1] += buf_e
        else:
            obs_pool.append(buf_o)
            exp_pool.append(buf_e)
    if len(obs_pool) > 1:
        obs_arr = np.array(obs_pool, dtype=float)
        exp_arr = np.array(exp_pool, dtype=float)
        exp_arr = exp_arr * (obs_arr.sum() / exp_arr.sum())
        chi2, p_chi2 = stats.chisquare(obs_arr, f_exp=exp_arr)
        gof_label = f"χ²={chi2:.2f}, p={p_chi2:.3f}"
    else:
        gof_label = "χ² test: insufficient bins"
    ax_acc.text(0.98, 0.97, f"n={n}\n{gof_label}", ha="right", va="top",
                transform=ax_acc.transAxes, fontsize=7.5, color="#555")

    # ── Panel C: AI vs Accidents scatter ─────────────────────────────────────
    ax_scat = fig.add_subplot(gs[1, 0])
    ax_scat.set_title("(c) AI vs. Total Accidents", fontweight="bold", pad=8)

    colors_scat = [AI_COLORS.get(c, "#aaa") for c in both["category"]]
    ax_scat.scatter(both["total_accidents"], both["antifragility_index"],
                    c=colors_scat, s=55, alpha=0.75, edgecolors="white", lw=0.5, zorder=3)
    m, b, _, _, _ = stats.linregress(both["total_accidents"], both["antifragility_index"])
    x_line = np.linspace(both["total_accidents"].min(), both["total_accidents"].max(), 100)
    ax_scat.plot(x_line, m * x_line + b, color="#2c3e50", lw=1.5, linestyle="--",
                 label=f"OLS  y={m:.4f}x{b:+.3f}")
    for thresh, col in [(0.05, AI_COLORS["ANTIFRAGILE"]), (-0.05, AI_COLORS["FRAGILE"]),
                        (-0.20, AI_COLORS["BRITTLE"])]:
        ax_scat.axhline(thresh, color=col, lw=0.8, linestyle=":", alpha=0.8)
    ax_scat.set_xlabel("Total Accidents per Run")
    ax_scat.set_ylabel("Antifragility Index")
    ax_scat.legend(fontsize=8)
    sig = "***" if p_rho < 0.001 else "**" if p_rho < 0.01 else "*" if p_rho < 0.05 else "n.s."
    ax_scat.text(0.98, 0.97,
                 f"Spearman ρ={rho:.3f} ({sig})\np={p_rho:.3f}  n={len(both)}",
                 ha="right", va="top", transform=ax_scat.transAxes,
                 fontsize=8, color="#555")

    # ── Panel D: Category breakdown (FIX: no invert_yaxis, explicit numeric y) ─
    ax_cat = fig.add_subplot(gs[1, 1])
    ax_cat.set_title("(d) Resilience Category Breakdown", fontweight="bold", pad=8)

    # Order: top → bottom visually means bottom-to-top in data (no invert needed)
    cat_order_display = ["ANTIFRAGILE", "RESILIENT", "FRAGILE", "BRITTLE", "N/A"]
    cat_order_plot    = list(reversed(cat_order_display))   # bottom → top
    all_cats   = df_runs["category"].value_counts()
    cat_counts = [all_cats.get(c, 0) for c in cat_order_plot]
    cat_cols   = [AI_COLORS.get(c, "#aaaaaa") for c in cat_order_plot]
    y_pos      = np.arange(len(cat_order_plot))

    bars = ax_cat.barh(y_pos, cat_counts, color=cat_cols, edgecolor="white",
                       height=0.55, align="center")
    ax_cat.set_yticks(y_pos)
    ax_cat.set_yticklabels(cat_order_plot, fontsize=10)

    max_count = max(cat_counts) if max(cat_counts) > 0 else 1
    for bar, cnt in zip(bars, cat_counts, strict=True):
        pct = 100 * cnt / n
        if cnt > 0:
            ax_cat.text(bar.get_width() + max_count * 0.015,
                        bar.get_y() + bar.get_height() / 2,
                        f"{cnt}  ({pct:.0f}%)", va="center", fontsize=9)

    ax_cat.set_xlabel("Number of Runs")
    ax_cat.set_xlim(0, max_count * 1.40)
    ax_cat.tick_params(axis="y", length=0)

    label = data.get("label", "")
    fig.suptitle(
        f"Thessaloniki Network Resilience — Batch Analysis {label}  (n={n} runs)",
        fontsize=13, fontweight="bold", y=0.99,
    )
    out_path = out_dir / "resilience_statistics.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 2 — Network dynamics ensemble
# ---------------------------------------------------------------------------

def figure_network_dynamics(data: dict, out_dir: Path) -> Path:
    """3-panel time-series ensemble — speed uses 5-min rolling average."""
    metrics = data["metrics"]
    if metrics.empty:
        print("  [WARN] No metrics data — skipping network dynamics figure")
        return out_dir / "network_dynamics.png"

    n_runs = int(metrics["run_idx"].nunique())

    t_count, arr_count = _pivot_and_smooth(metrics, "vehicle_count")
    t_speed, arr_speed = _pivot_and_smooth(metrics, "mean_speed_kmh",
                                           smooth_window=SPEED_SMOOTH_WINDOW)
    t_acc,   arr_acc   = _pivot_and_smooth(metrics, "active_accidents")

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    fig.subplots_adjust(hspace=0.18, top=0.93)

    THIN_ALPHA = 0.07
    THIN_COLOR = "#7f8c8d"
    MEAN_COLOR = "#2c3e50"

    panels = [
        (axes[0], t_count, arr_count,
         "Simultaneous Vehicles on Network", "Vehicles on Network", "#3498db"),
        (axes[1], t_speed, arr_speed,
         "Space-Mean Speed — 5-min Rolling Average (km/h)", "Speed (km/h)", "#e74c3c"),
        (axes[2], t_acc,   arr_acc,
         "Concurrent Active Accidents", "Active Accidents", "#f39c12"),
    ]

    for ax, t, arr, title, ylabel, fill_col in panels:
        # Individual run traces (thin + transparent)
        for j in range(arr.shape[1]):
            ax.plot(t, arr[:, j], color=THIN_COLOR, lw=0.45, alpha=THIN_ALPHA)

        mean_val = np.nanmean(arr, axis=1)
        q25      = np.nanpercentile(arr, 25, axis=1)
        q75      = np.nanpercentile(arr, 75, axis=1)
        p10      = np.nanpercentile(arr, 10, axis=1)
        p90      = np.nanpercentile(arr, 90, axis=1)

        ax.fill_between(t, p10, p90, alpha=0.13, color=fill_col, label="P10–P90 band")
        ax.fill_between(t, q25, q75, alpha=0.28, color=fill_col, label="IQR (P25–P75)")
        ax.plot(t, mean_val, color=MEAN_COLOR, lw=2.2, label="Ensemble mean")

        ax.set_title(title, fontsize=10, fontweight="bold", pad=4, loc="left")
        ax.set_ylabel(ylabel, fontsize=10)
        # Legend in lower-right so it does not overlap the top-left title
        ax.legend(fontsize=8, loc="lower right", ncol=3,
                  framealpha=0.85, edgecolor="none")
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("Simulation Time (min)", fontsize=11)
    axes[-1].set_xlim(t[0], t[-1])

    for ax in axes:
        ax.axvline(60,  color="#555", lw=0.7, linestyle=":", alpha=0.6)
        ax.axvline(120, color="#555", lw=0.7, linestyle=":", alpha=0.6)

    axes[0].text(60.5,  axes[0].get_ylim()[1] * 0.96, "1 h",  fontsize=7.5, color="#555")
    axes[0].text(120.5, axes[0].get_ylim()[1] * 0.96, "2 h",  fontsize=7.5, color="#555")

    label = data.get("label", "")
    fig.suptitle(
        f"Network Temporal Dynamics — Ensemble of {n_runs} Runs  "
        f"[Thessaloniki {label}]",
        fontsize=12, fontweight="bold",
    )

    out_path = out_dir / "network_dynamics.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 3 — Accident characteristics
# ---------------------------------------------------------------------------

def figure_accident_characteristics(data: dict, out_dir: Path) -> Path:
    """4-panel accident deep-dive."""
    accidents = data["accidents"]
    n_runs    = len(data["runs"])

    if accidents.empty:
        print("  [WARN] No accident data — skipping characteristics figure")
        return out_dir / "accident_characteristics.png"

    accidents = accidents.copy()
    if "location.x" in accidents.columns:
        accidents["x"] = accidents["location.x"]
        accidents["y"] = accidents["location.y"]
    accidents["trigger_min"] = accidents["trigger_step"] * 5 / 60

    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # ── Panel A: Severity distribution ───────────────────────────────────────
    ax_sev = fig.add_subplot(gs[0, 0])
    ax_sev.set_title("(a) Severity Distribution (all runs)", fontweight="bold", pad=8)

    sev_counts = accidents["severity"].value_counts().reindex(SEVERITY_ORDER, fill_value=0)
    total_acc  = sev_counts.sum()
    bars = ax_sev.bar(SEVERITY_ORDER, sev_counts.values,
                      color=[SEVERITY_COLORS[s] for s in SEVERITY_ORDER],
                      edgecolor="white", width=0.6)
    for bar, cnt in zip(bars, sev_counts.values, strict=True):
        pct = 100 * cnt / total_acc
        ax_sev.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{cnt}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=9)
    ax_sev.set_ylabel("Number of Accidents")
    ax_sev.set_xlabel("Severity")
    ax_sev.text(0.98, 0.97, f"Total: {total_acc} accidents\nacross {n_runs} runs\n"
                f"Mean: {total_acc/n_runs:.1f}/run",
                ha="right", va="top", transform=ax_sev.transAxes, fontsize=8, color="#555")

    # ── Panel B: Duration by severity ────────────────────────────────────────
    ax_dur = fig.add_subplot(gs[0, 1])
    ax_dur.set_title("(b) Incident Duration by Severity", fontweight="bold", pad=8)

    acc_with_dur = accidents[
        accidents["duration_seconds"].notna() & accidents["severity"].isin(SEVERITY_ORDER)
    ].copy()
    palette     = {s: SEVERITY_COLORS[s] for s in SEVERITY_ORDER}
    present_sev = [s for s in SEVERITY_ORDER if s in acc_with_dur["severity"].values]

    sns.boxplot(data=acc_with_dur, x="severity", y="duration_seconds",
                order=present_sev, palette=palette, ax=ax_dur,
                width=0.5, fliersize=3, linewidth=1.2)
    sns.stripplot(data=acc_with_dur, x="severity", y="duration_seconds",
                  order=present_sev, palette=palette, ax=ax_dur,
                  size=3, alpha=0.5, jitter=True, zorder=3)
    ax_dur.set_ylabel("Duration (s)")
    ax_dur.set_xlabel("Severity")

    for i, sev in enumerate(present_sev):
        med = acc_with_dur[acc_with_dur["severity"] == sev]["duration_seconds"].median()
        ax_dur.text(i, med + 20, f"{med:.0f}s", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    # ── Panel C: Temporal distribution ───────────────────────────────────────
    ax_time = fig.add_subplot(gs[1, 0])
    ax_time.set_title("(c) Accident Trigger Time Distribution", fontweight="bold", pad=8)

    for sev in present_sev:
        sub = accidents[accidents["severity"] == sev]["trigger_min"]
        if len(sub) > 1:
            kde = stats.gaussian_kde(sub, bw_method="silverman")
            x_t = np.linspace(0, 120, 300)
            ax_time.plot(x_t, kde(x_t), color=SEVERITY_COLORS[sev], lw=2, label=sev)
        else:
            ax_time.axvline(sub.values[0], color=SEVERITY_COLORS[sev],
                            lw=1.5, linestyle="--", label=sev)

    ax_time.hist(accidents["trigger_min"], bins=24, density=True,
                 alpha=0.18, color="#7f8c8d", edgecolor="white")

    all_trigger = accidents["trigger_min"].dropna()
    if len(all_trigger) > 2:
        kde_all = stats.gaussian_kde(all_trigger, bw_method="silverman")
        x_all   = np.linspace(0, 120, 300)
        peak_t  = x_all[np.argmax(kde_all(x_all))]
        ax_time.axvline(peak_t, color="#2c3e50", lw=1.2, linestyle="--",
                        label=f"Peak ≈ {peak_t:.0f} min")

    ax_time.set_xlabel("Trigger Time (min)")
    ax_time.set_ylabel("Density")
    ax_time.set_xlim(0, 120)
    ax_time.legend(fontsize=8)

    # ── Panel D: Impact — vehicles affected ───────────────────────────────────
    ax_imp = fig.add_subplot(gs[1, 1])
    ax_imp.set_title("(d) Vehicles Affected per Incident", fontweight="bold", pad=8)

    imp_col = "impact.vehicles_affected" if "impact.vehicles_affected" in accidents.columns \
              else "vehicles_affected"

    if imp_col in accidents.columns:
        acc_imp = accidents[
            accidents[imp_col].notna() & accidents["severity"].isin(SEVERITY_ORDER)
        ].copy()
        sns.violinplot(data=acc_imp, x="severity", y=imp_col,
                       order=present_sev, palette=palette, ax=ax_imp,
                       inner="quartile", linewidth=1.2, density_norm="width")
        ax_imp.set_ylabel("Vehicles Affected")
        ax_imp.set_xlabel("Severity")
        med_all = accidents[imp_col].median()
        max_all = accidents[imp_col].max()
        ax_imp.text(0.98, 0.97, f"Median: {med_all:.0f}  Max: {max_all:.0f}",
                    ha="right", va="top", transform=ax_imp.transAxes,
                    fontsize=8, color="#555")
    else:
        ax_imp.text(0.5, 0.5, "Impact data not available",
                    ha="center", va="center", transform=ax_imp.transAxes)

    label = data.get("label", "")
    fig.suptitle(
        f"Accident Characteristics — {total_acc} accidents across {n_runs} runs  "
        f"[Thessaloniki {label}]",
        fontsize=12, fontweight="bold", y=0.99,
    )
    out_path = out_dir / "accident_characteristics.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 4 — Spatial heatmap (with optional network underlay)
# ---------------------------------------------------------------------------

def figure_spatial_heatmap(
    data: dict,
    out_dir: Path,
    net_segments: list[np.ndarray] | None = None,
    net_extent: tuple[float, ...] | None = None,
) -> Path:
    """2D KDE heatmap of accident locations, with road network underlay."""
    accidents = data["accidents"]

    if accidents.empty:
        print("  [WARN] No accident data — skipping spatial figure")
        return out_dir / "spatial_heatmap.png"

    x_col = "location.x" if "location.x" in accidents.columns else "x"
    y_col = "location.y" if "location.y" in accidents.columns else "y"

    if x_col not in accidents.columns:
        print("  [WARN] No location data — skipping spatial figure")
        return out_dir / "spatial_heatmap.png"

    acc_xy = accidents[[x_col, y_col, "severity"]].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.subplots_adjust(wspace=0.06)

    x_arr = acc_xy[x_col].values
    y_arr = acc_xy[y_col].values

    # Determine axis limits: prefer network extent when available
    if net_extent:
        x_lo, x_hi, y_lo, y_hi = net_extent
    else:
        pad_x = (x_arr.max() - x_arr.min()) * 0.05
        pad_y = (y_arr.max() - y_arr.min()) * 0.05
        x_lo, x_hi = x_arr.min() - pad_x, x_arr.max() + pad_x
        y_lo, y_hi = y_arr.min() - pad_y, y_arr.max() + pad_y

    # ── Left: KDE density with network underlay ───────────────────────────────
    ax_kde = axes[0]
    ax_kde.set_title("(a) Accident Location Density (2D KDE)", fontweight="bold", pad=8)

    if net_segments and net_extent:
        draw_network_bg(ax_kde, net_segments, net_extent, alpha=0.28)
    else:
        ax_kde.set_xlim(x_lo, x_hi)
        ax_kde.set_ylim(y_lo, y_hi)
        ax_kde.set_aspect("equal")

    xi = np.linspace(x_lo, x_hi, 250)
    yi = np.linspace(y_lo, y_hi, 250)
    Xi, Yi = np.meshgrid(xi, yi)
    kernel  = stats.gaussian_kde(np.vstack([x_arr, y_arr]), bw_method=0.15)
    Zi      = kernel(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

    cf = ax_kde.contourf(Xi, Yi, Zi, levels=20, cmap="YlOrRd", alpha=0.80, zorder=2)
    ax_kde.scatter(x_arr, y_arr, s=6, c="white", alpha=0.30, edgecolors="none", zorder=3)
    plt.colorbar(cf, ax=ax_kde, label="KDE Density", shrink=0.85)
    ax_kde.set_xlabel("X Coordinate (m)")
    ax_kde.set_ylabel("Y Coordinate (m)")
    ax_kde.text(0.02, 0.98, f"n = {len(acc_xy)} accidents",
                transform=ax_kde.transAxes, va="top", fontsize=9,
                color="white", fontweight="bold", zorder=4)

    # ── Right: Severity scatter with network underlay ─────────────────────────
    ax_scat = axes[1]
    ax_scat.set_title("(b) Accident Locations by Severity", fontweight="bold", pad=8)

    if net_segments and net_extent:
        draw_network_bg(ax_scat, net_segments, net_extent, alpha=0.28)
    else:
        ax_scat.set_xlim(x_lo, x_hi)
        ax_scat.set_ylim(y_lo, y_hi)
        ax_scat.set_aspect("equal")

    for sev in SEVERITY_ORDER:
        sub = acc_xy[acc_xy["severity"] == sev]
        if sub.empty:
            continue
        ax_scat.scatter(sub[x_col], sub[y_col],
                        s=40 if sev in ("MAJOR", "CRITICAL") else 20,
                        color=SEVERITY_COLORS[sev], alpha=0.70,
                        edgecolors="white", lw=0.4, label=sev, zorder=3)

    ax_scat.set_xlabel("X Coordinate (m)")
    ax_scat.set_yticklabels([])
    ax_scat.legend(title="Severity", fontsize=9, title_fontsize=9,
                   framealpha=0.90, edgecolor="none", loc="upper right")

    label = data.get("label", "")
    fig.suptitle(
        f"Spatial Distribution of Accidents — {len(acc_xy)} incidents  "
        f"across {data['aggregate']['n_runs']} runs  [Thessaloniki {label}]",
        fontsize=12, fontweight="bold",
    )
    out_path = out_dir / "spatial_heatmap.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Figure 5 — Per-event AI by severity
# ---------------------------------------------------------------------------

def figure_per_event_ai(data: dict, out_dir: Path) -> Path:
    """Per-event AI breakdown joined with accident severity."""
    ai_events = data["ai_events"]
    accidents = data["accidents"]

    if ai_events.empty or accidents.empty:
        print("  [WARN] Insufficient event-AI data — skipping per-event figure")
        return out_dir / "per_event_ai.png"

    acc_sev = accidents[["accident_id", "severity"]].drop_duplicates()
    merged  = ai_events.merge(acc_sev, on="accident_id", how="left")
    present_sev  = [s for s in SEVERITY_ORDER if s in merged["severity"].values]
    merged_clean = merged[merged["severity"].isin(present_sev)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.subplots_adjust(wspace=0.28)

    # ── Panel A: per-event AI violin ──────────────────────────────────────────
    ax_v   = axes[0]
    ax_v.set_title("(a) Per-Event Antifragility Index by Severity", fontweight="bold", pad=8)
    palette = {s: SEVERITY_COLORS[s] for s in SEVERITY_ORDER}

    sns.violinplot(data=merged_clean, x="severity", y="event_ai",
                   order=present_sev, palette=palette, ax=ax_v,
                   inner="quartile", linewidth=1.2, density_norm="width")
    sns.stripplot(data=merged_clean, x="severity", y="event_ai",
                  order=present_sev, palette=palette, ax=ax_v,
                  size=3.5, alpha=0.55, jitter=True, zorder=4)

    for thresh, col in [(0.05, AI_COLORS["ANTIFRAGILE"]),
                        (-0.05, AI_COLORS["FRAGILE"]),
                        (-0.20, AI_COLORS["BRITTLE"])]:
        ax_v.axhline(thresh, color=col, lw=0.9, linestyle=":", alpha=0.9)
    ax_v.axhline(0, color="#333", lw=1.0, linestyle="-", alpha=0.5)
    ax_v.set_ylabel("Per-Event Antifragility Index")
    ax_v.set_xlabel("Severity")
    ax_v.text(0.98, 0.97, f"n = {len(merged_clean)} events",
              ha="right", va="top", transform=ax_v.transAxes, fontsize=8, color="#555")

    groups = [
        merged_clean[merged_clean["severity"] == s]["event_ai"].dropna().values
        for s in present_sev
        if len(merged_clean[merged_clean["severity"] == s]) >= 2
    ]
    if len(groups) >= 2:
        h_stat, p_kw = stats.kruskal(*groups)
        ax_v.text(0.02, 0.02, f"Kruskal–Wallis: H={h_stat:.2f}, p={p_kw:.3f}",
                  ha="left", va="bottom", transform=ax_v.transAxes,
                  fontsize=7.5, color="#555")

    # ── Panel B: pre vs post speed ────────────────────────────────────────────
    ax_sp = axes[1]
    ax_sp.set_title("(b) Pre- vs Post-Event Speed by Severity", fontweight="bold", pad=8)

    pre_vals  = merged_clean.groupby("severity")["pre_mean_speed_kmh"].mean().reindex(present_sev)
    post_vals = merged_clean.groupby("severity")["post_mean_speed_kmh"].mean().reindex(present_sev)
    x_pos = np.arange(len(present_sev))
    width = 0.35
    ax_sp.bar(x_pos - width / 2, pre_vals.values,  width, label="Pre-event",
              color="#3498db", alpha=0.8, edgecolor="white")
    ax_sp.bar(x_pos + width / 2, post_vals.values, width, label="Post-event",
              color="#e74c3c", alpha=0.8, edgecolor="white")
    for i, (pre, post) in enumerate(zip(pre_vals.values, post_vals.values, strict=False)):
        if not (np.isnan(pre) or np.isnan(post)):
            arrow_col = "#27ae60" if post >= pre else "#e74c3c"
            ax_sp.annotate("", xy=(i + width / 2, post), xytext=(i - width / 2, pre),
                           arrowprops=dict(arrowstyle="->", color=arrow_col, lw=1.2))
    ax_sp.set_xticks(x_pos)
    ax_sp.set_xticklabels(present_sev)
    ax_sp.set_ylabel("Mean Speed (km/h)")
    ax_sp.set_xlabel("Severity")
    ax_sp.legend(fontsize=9)

    label = data.get("label", "")
    fig.suptitle(
        f"Per-Event Antifragility Analysis by Accident Severity  [Thessaloniki {label}]",
        fontsize=12, fontweight="bold",
    )
    out_path = out_dir / "per_event_ai.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Statistical summary (console)
# ---------------------------------------------------------------------------

def print_statistical_summary(data: dict) -> None:
    df_runs  = _annotate_runs_df(data)
    agg      = data["aggregate"]
    accidents = data["accidents"]
    ai_events = data["ai_events"]
    ai_valid  = df_runs["antifragility_index"].dropna()

    label = data.get("label", "")
    print("\n" + "=" * 65)
    print(f"  THESSALONIKI BATCH [{label}] — STATISTICAL SUMMARY")
    print("=" * 65)

    print("\nRUNS")
    print(f"  Total runs           : {agg['n_runs']}")
    print(f"  Runs with AI data    : {len(ai_valid)}")

    print("\nACCIDENTS")
    print(f"  Total                : {int(df_runs['total_accidents'].sum())}")
    print(f"  Mean ± SD / run      : {agg['accident_mean']:.2f} ± {agg['accident_std']:.2f}")
    acc_range = (f"{df_runs['total_accidents'].min():.0f}–"
                 f"{df_runs['total_accidents'].max():.0f}")
    print(f"  Range                : {acc_range}")

    if not accidents.empty:
        sev_counts = accidents["severity"].value_counts()
        print("  Severity breakdown:")
        for sev in SEVERITY_ORDER:
            cnt = sev_counts.get(sev, 0)
            pct = 100 * cnt / len(accidents)
            print(f"    {sev:12s}: {cnt:4d}  ({pct:.1f}%)")

    print("\nANTIFRAGILITY INDEX")
    print(f"  Batch mean           : {agg['ai_mean']:.4f}")
    print(f"  Batch SD             : {agg['ai_std']:.4f}")
    print(f"  95% CI               : [{agg['ai_ci_95_low']:.4f}, {agg['ai_ci_95_high']:.4f}]")
    sw_stat, sw_p = stats.shapiro(ai_valid)
    print(f"  Shapiro–Wilk         : W={sw_stat:.4f}, p={sw_p:.4f}")
    t_stat, t_p = stats.ttest_1samp(ai_valid, 0)
    print(f"  t-test vs AI=0       : t={t_stat:.3f}, p={t_p:.4f}")

    cat_counts = df_runs["category"].value_counts()
    print("  Category breakdown:")
    for cat in ["ANTIFRAGILE", "RESILIENT", "FRAGILE", "BRITTLE", "N/A"]:
        cnt = cat_counts.get(cat, 0)
        pct = 100 * cnt / agg["n_runs"]
        print(f"    {cat:15s}: {cnt:3d}  ({pct:.0f}%)")

    if not ai_events.empty:
        print("\nPER-EVENT AI")
        print(f"  Total events measured: {len(ai_events)}")
        print(f"  Mean ± SD            : {ai_events['event_ai'].mean():.4f} ± "
              f"{ai_events['event_ai'].std():.4f}")
        pre_m  = ai_events["pre_mean_speed_kmh"].mean()
        post_m = ai_events["post_mean_speed_kmh"].mean()
        print(f"  Mean pre-event speed : {pre_m:.2f} km/h")
        print(f"  Mean post-event speed: {post_m:.2f} km/h")
        t2, p2 = stats.ttest_rel(
            ai_events["post_mean_speed_kmh"].dropna(),
            ai_events["pre_mean_speed_kmh"].dropna(),
        )
        print(f"  Paired t-test (post–pre): t={t2:.3f}, p={p2:.4f}")

    print("\n" + "=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Comparative Figure 1 — Overview (AI, accidents, categories)
# ---------------------------------------------------------------------------

def figure_comparative_overview(
    data_a: dict, data_b: dict, out_dir: Path
) -> Path:
    """4-panel comparative overview: AI distributions, accident counts, categories, stats table."""
    label_a  = data_a["label"]
    label_b  = data_b["label"]
    col_a    = BATCH_PALETTE["A"]
    col_b    = BATCH_PALETTE["B"]
    fill_a   = BATCH_FILL["A"]
    fill_b   = BATCH_FILL["B"]

    df_a = _annotate_runs_df(data_a)
    df_b = _annotate_runs_df(data_b)
    ai_a = df_a["antifragility_index"].dropna()
    ai_b = df_b["antifragility_index"].dropna()
    agg_a = data_a["aggregate"]
    agg_b = data_b["aggregate"]

    fig = plt.figure(figsize=(15, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # ── Panel A: Overlaid AI distributions ───────────────────────────────────
    ax_ai = fig.add_subplot(gs[0, 0])
    ax_ai.set_title("(a) AI Distribution Comparison", fontweight="bold", pad=8)

    for ai_vals, col, fill, label, agg in [
        (ai_a, col_a, fill_a, label_a, agg_a),
        (ai_b, col_b, fill_b, label_b, agg_b),
    ]:
        x_kde = np.linspace(min(ai_a.min(), ai_b.min()) - 0.05,
                            max(ai_a.max(), ai_b.max()) + 0.05, 400)
        y_kde = stats.gaussian_kde(ai_vals, bw_method="silverman")(x_kde)
        ax_ai.fill_between(x_kde, y_kde, alpha=0.18, color=fill)
        ax_ai.plot(x_kde, y_kde, color=col, lw=2,
                   label=f"Batch {label}  (μ={agg['ai_mean']:+.3f})")
        ax_ai.axvline(agg["ai_mean"], color=col, lw=1.5, linestyle="--", alpha=0.8)
        ax_ai.axvspan(agg["ai_ci_95_low"], agg["ai_ci_95_high"],
                      alpha=0.08, color=col)

    for thresh, ls in [(-0.20, "--"), (-0.05, ":"), (0.05, ":")]:
        ax_ai.axvline(thresh, color="#999", lw=0.8, linestyle=ls, alpha=0.7)

    ax_ai.set_xlabel("Antifragility Index")
    ax_ai.set_ylabel("Density")
    ax_ai.legend(fontsize=9)

    # Mann–Whitney U test
    u_stat, p_mw = stats.mannwhitneyu(ai_a, ai_b, alternative="two-sided")
    sig = "***" if p_mw < 0.001 else "**" if p_mw < 0.01 else "*" if p_mw < 0.05 else "n.s."
    ax_ai.text(0.98, 0.97,
               f"Mann–Whitney U={u_stat:.0f}\np={p_mw:.4f} ({sig})",
               ha="right", va="top", transform=ax_ai.transAxes,
               fontsize=8, color="#555")

    # ── Panel B: Accident count distributions ────────────────────────────────
    ax_acc = fig.add_subplot(gs[0, 1])
    ax_acc.set_title("(b) Accident Count Comparison", fontweight="bold", pad=8)

    acc_a = df_a["total_accidents"].values
    acc_b = df_b["total_accidents"].values
    lo_acc = min(acc_a.min(), acc_b.min())
    hi_acc = max(acc_a.max(), acc_b.max())
    bins   = np.arange(lo_acc - 0.5, hi_acc + 1.5, 1)

    ax_acc.hist(acc_a, bins=bins, color=col_a, alpha=0.55, edgecolor="white",
                label=f"Batch {label_a}  (μ={agg_a['accident_mean']:.1f})")
    ax_acc.hist(acc_b, bins=bins, color=col_b, alpha=0.55, edgecolor="white",
                label=f"Batch {label_b}  (μ={agg_b['accident_mean']:.1f})")
    ax_acc.axvline(agg_a["accident_mean"], color=col_a, lw=2, linestyle="--")
    ax_acc.axvline(agg_b["accident_mean"], color=col_b, lw=2, linestyle="--")
    ax_acc.set_xlabel("Total Accidents per Run")
    ax_acc.set_ylabel("Number of Runs")
    ax_acc.legend(fontsize=9)

    u2, p_mw2 = stats.mannwhitneyu(acc_a, acc_b, alternative="two-sided")
    sig2 = "***" if p_mw2 < 0.001 else "**" if p_mw2 < 0.01 else "*" if p_mw2 < 0.05 else "n.s."
    ax_acc.text(0.98, 0.97, f"Mann–Whitney U={u2:.0f}\np={p_mw2:.4f} ({sig2})",
                ha="right", va="top", transform=ax_acc.transAxes,
                fontsize=8, color="#555")

    # ── Panel C: Category breakdown comparison ────────────────────────────────
    ax_cat = fig.add_subplot(gs[1, 0])
    ax_cat.set_title("(c) Resilience Category Comparison", fontweight="bold", pad=8)

    cat_display  = ["ANTIFRAGILE", "RESILIENT", "FRAGILE", "BRITTLE", "N/A"]
    cat_plot     = list(reversed(cat_display))   # bottom → top, no invert_yaxis
    y_pos        = np.arange(len(cat_plot))
    cats_a       = df_a["category"].value_counts()
    cats_b       = df_b["category"].value_counts()
    counts_a     = np.array([cats_a.get(c, 0) for c in cat_plot], dtype=float)
    counts_b     = np.array([cats_b.get(c, 0) for c in cat_plot], dtype=float)
    n_a          = int(agg_a["n_runs"])
    n_b          = int(agg_b["n_runs"])

    height = 0.35
    bars_a = ax_cat.barh(y_pos + height / 2, counts_a, height=height,
                         color=col_a, alpha=0.80, edgecolor="white",
                         label=f"Batch {label_a}")
    bars_b = ax_cat.barh(y_pos - height / 2, counts_b, height=height,
                         color=col_b, alpha=0.80, edgecolor="white",
                         label=f"Batch {label_b}")
    ax_cat.set_yticks(y_pos)
    ax_cat.set_yticklabels(cat_plot, fontsize=10)

    max_c = max(counts_a.max(), counts_b.max()) if max(counts_a.max(), counts_b.max()) > 0 else 1
    for bar, cnt, n in [(bars_a, counts_a, n_a), (bars_b, counts_b, n_b)]:
        for b, c in zip(bar, cnt, strict=True):
            if c > 0:
                ax_cat.text(b.get_width() + max_c * 0.01,
                            b.get_y() + b.get_height() / 2,
                            f"{int(c)} ({100*c/n:.0f}%)", va="center", fontsize=8)

    ax_cat.set_xlabel("Number of Runs")
    ax_cat.set_xlim(0, max_c * 1.50)
    ax_cat.tick_params(axis="y", length=0)
    ax_cat.legend(fontsize=9)

    # ── Panel D: Summary statistics table ─────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[1, 1])
    ax_tbl.set_title("(d) Batch Summary Statistics", fontweight="bold", pad=8)
    ax_tbl.axis("off")

    rows = [
        ("Metric",               f"Batch {label_a}",      f"Batch {label_b}"),
        ("Runs",                 f"{n_a}",                 f"{n_b}"),
        ("AI mean",              f"{agg_a['ai_mean']:+.4f}",  f"{agg_b['ai_mean']:+.4f}"),
        ("AI SD",                f"{agg_a['ai_std']:.4f}",    f"{agg_b['ai_std']:.4f}"),
        ("AI 95% CI low",        f"{agg_a['ai_ci_95_low']:.4f}", f"{agg_b['ai_ci_95_low']:.4f}"),
        ("AI 95% CI high",       f"{agg_a['ai_ci_95_high']:.4f}", f"{agg_b['ai_ci_95_high']:.4f}"),
        ("Accident mean",        f"{agg_a['accident_mean']:.2f}", f"{agg_b['accident_mean']:.2f}"),
        ("Accident SD",          f"{agg_a['accident_std']:.2f}",  f"{agg_b['accident_std']:.2f}"),
        ("Mann–Whitney AI",      f"U={u_stat:.0f}, p={p_mw:.4f}", ""),
        ("Mann–Whitney Acc.",    f"U={u2:.0f}, p={p_mw2:.4f}", ""),
    ]

    tbl = ax_tbl.table(
        cellText=[r[1:] for r in rows[1:]],
        rowLabels=[r[0] for r in rows[1:]],
        colLabels=[rows[0][1], rows[0][2]],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.55)

    # Colour header row
    for j in range(2):
        tbl[0, j].set_facecolor("#E8EAF6")
    for i in range(len(rows) - 1):
        tbl[i + 1, -1].set_facecolor("#FAFAFA")

    fig.suptitle(
        f"Comparative Resilience Overview — Batch {label_a} vs Batch {label_b}  "
        f"(Thessaloniki)",
        fontsize=13, fontweight="bold", y=0.99,
    )
    out_path = out_dir / "comparative_overview.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Comparative Figure 2 — Network dynamics
# ---------------------------------------------------------------------------

def figure_comparative_dynamics(
    data_a: dict, data_b: dict, out_dir: Path
) -> Path:
    """3-panel overlay: ensemble mean ± IQR for both batches."""
    label_a = data_a["label"]
    label_b = data_b["label"]
    col_a   = BATCH_PALETTE["A"]
    col_b   = BATCH_PALETTE["B"]
    fill_a  = BATCH_FILL["A"]
    fill_b  = BATCH_FILL["B"]

    if data_a["metrics"].empty or data_b["metrics"].empty:
        print("  [WARN] Metrics missing for one batch — skipping comparative dynamics")
        return out_dir / "comparative_dynamics.png"

    panels = [
        ("vehicle_count",   "Vehicles on Network",                   None),
        ("mean_speed_kmh",  "Space-Mean Speed — 5-min Avg (km/h)",   SPEED_SMOOTH_WINDOW),
        ("active_accidents","Concurrent Active Accidents",             None),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    fig.subplots_adjust(hspace=0.18, top=0.93)

    for ax, (col, ylabel, smooth) in zip(axes, panels, strict=True):
        t_a, arr_a = _pivot_and_smooth(data_a["metrics"], col, smooth_window=smooth)
        t_b, arr_b = _pivot_and_smooth(data_b["metrics"], col, smooth_window=smooth)

        for t, arr, clr, fill, lab in [
            (t_a, arr_a, col_a, fill_a, label_a),
            (t_b, arr_b, col_b, fill_b, label_b),
        ]:
            mean_val = np.nanmean(arr, axis=1)
            q25      = np.nanpercentile(arr, 25, axis=1)
            q75      = np.nanpercentile(arr, 75, axis=1)
            p10      = np.nanpercentile(arr, 10, axis=1)
            p90      = np.nanpercentile(arr, 90, axis=1)

            ax.fill_between(t, p10, p90, alpha=0.10, color=fill)
            ax.fill_between(t, q25, q75, alpha=0.22, color=fill)
            ax.plot(t, mean_val, color=clr, lw=2.2, label=f"Batch {lab} — mean")

        ax.set_title(ylabel, fontsize=10, fontweight="bold", pad=4, loc="left")
        ax.set_ylabel(ylabel.split("—")[0].strip(), fontsize=10)
        ax.legend(fontsize=9, loc="lower right", framealpha=0.85, edgecolor="none")
        ax.set_ylim(bottom=0)

        for vline_x in [60, 120]:
            ax.axvline(vline_x, color="#999", lw=0.7, linestyle=":", alpha=0.6)

    axes[-1].set_xlabel("Simulation Time (min)", fontsize=11)
    axes[-1].set_xlim(t_a[0], t_a[-1])
    axes[0].text(60.5,  axes[0].get_ylim()[1] * 0.96, "1 h", fontsize=7.5, color="#555")
    axes[0].text(120.5, axes[0].get_ylim()[1] * 0.96, "2 h", fontsize=7.5, color="#555")

    fig.suptitle(
        f"Network Dynamics Comparison — Batch {label_a} vs Batch {label_b}  "
        f"(Thessaloniki, mean ± IQR shading)",
        fontsize=12, fontweight="bold",
    )
    out_path = out_dir / "comparative_dynamics.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Comparative Figure 3 — Spatial heatmaps side-by-side on network
# ---------------------------------------------------------------------------

def figure_comparative_spatial(
    data_a: dict,
    data_b: dict,
    out_dir: Path,
    net_segments: list[np.ndarray] | None = None,
    net_extent: tuple[float, ...] | None = None,
) -> Path:
    """Side-by-side KDE heatmaps for both batches on the road network."""
    label_a = data_a["label"]
    label_b = data_b["label"]

    x_col = "location.x"
    y_col = "location.y"

    def _get_xy(data: dict) -> np.ndarray | None:
        acc = data["accidents"]
        if acc.empty or x_col not in acc.columns:
            return None
        return acc[[x_col, y_col]].dropna().values

    xy_a = _get_xy(data_a)
    xy_b = _get_xy(data_b)

    if xy_a is None or xy_b is None:
        print("  [WARN] Missing location data — skipping comparative spatial figure")
        return out_dir / "comparative_spatial.png"

    # Shared axis limits
    if net_extent:
        x_lo, x_hi, y_lo, y_hi = net_extent
    else:
        all_x = np.concatenate([xy_a[:, 0], xy_b[:, 0]])
        all_y = np.concatenate([xy_a[:, 1], xy_b[:, 1]])
        pad_x = (all_x.max() - all_x.min()) * 0.04
        pad_y = (all_y.max() - all_y.min()) * 0.04
        x_lo, x_hi = all_x.min() - pad_x, all_x.max() + pad_x
        y_lo, y_hi = all_y.min() - pad_y, all_y.max() + pad_y

    # Shared KDE grid
    xi = np.linspace(x_lo, x_hi, 250)
    yi = np.linspace(y_lo, y_hi, 250)
    Xi, Yi = np.meshgrid(xi, yi)
    pos    = np.vstack([Xi.ravel(), Yi.ravel()])

    def _kde(xy: np.ndarray) -> np.ndarray:
        return stats.gaussian_kde(xy.T, bw_method=0.15)(pos).reshape(Xi.shape)

    Za = _kde(xy_a)
    Zb = _kde(xy_b)
    vmax = max(Za.max(), Zb.max())

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    fig.subplots_adjust(wspace=0.04)

    for ax, Zi, xy, _data, label in [
        (axes[0], Za, xy_a, data_a, label_a),
        (axes[1], Zb, xy_b, data_b, label_b),
    ]:
        # Road network underlay
        if net_segments and net_extent:
            draw_network_bg(ax, net_segments, net_extent, alpha=0.28)
        else:
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_aspect("equal")

        cf = ax.contourf(Xi, Yi, Zi, levels=20, cmap="YlOrRd",
                         alpha=0.82, vmin=0, vmax=vmax, zorder=2)
        ax.scatter(xy[:, 0], xy[:, 1], s=5, c="white",
                   alpha=0.30, edgecolors="none", zorder=3)
        ax.set_title(
            f"Batch {label} — {len(xy)} accident locations",
            fontweight="bold", pad=8,
        )
        ax.set_xlabel("X Coordinate (m)")
        if ax is axes[0]:
            ax.set_ylabel("Y Coordinate (m)")
        else:
            ax.set_yticklabels([])

    # Shared colorbar
    cbar = fig.colorbar(cf, ax=axes.tolist(), label="KDE Density",
                        shrink=0.72, pad=0.01)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        f"Spatial Accident Distribution — Batch {label_a} vs Batch {label_b}  "
        f"(Thessaloniki, shared colour scale)",
        fontsize=12, fontweight="bold",
    )
    out_path = out_dir / "comparative_spatial.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Comparative Figure 4 — Per-event AI comparison
# ---------------------------------------------------------------------------

def figure_comparative_per_event_ai(
    data_a: dict, data_b: dict, out_dir: Path
) -> Path:
    """Side-by-side per-event AI distributions and pre/post speed."""
    label_a = data_a["label"]
    label_b = data_b["label"]
    col_a   = BATCH_PALETTE["A"]
    col_b   = BATCH_PALETTE["B"]

    def _merge(data: dict) -> pd.DataFrame | None:
        ai  = data["ai_events"]
        acc = data["accidents"]
        if ai.empty or acc.empty:
            return None
        acc_sev = acc[["accident_id", "severity"]].drop_duplicates()
        merged  = ai.merge(acc_sev, on="accident_id", how="left")
        merged  = merged[merged["severity"].isin(SEVERITY_ORDER)]
        merged["batch"] = data["label"]
        return merged

    merged_a = _merge(data_a)
    merged_b = _merge(data_b)

    if merged_a is None or merged_b is None:
        print("  [WARN] Insufficient event data — skipping comparative per-event AI")
        return out_dir / "comparative_per_event_ai.png"

    combined = pd.concat([merged_a, merged_b], ignore_index=True)
    present_sev = [s for s in SEVERITY_ORDER if s in combined["severity"].values]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.subplots_adjust(wspace=0.30)

    # ── Panel A: violins by batch ─────────────────────────────────────────────
    ax_v = axes[0]
    ax_v.set_title("(a) Per-Event AI — Both Batches by Severity", fontweight="bold", pad=8)

    batch_palette = {label_a: col_a, label_b: col_b}
    sns.violinplot(
        data=combined, x="severity", y="event_ai", hue="batch",
        order=present_sev, palette=batch_palette, ax=ax_v,
        inner="quartile", linewidth=1.1, density_norm="width",
        split=False,
    )
    sns.stripplot(
        data=combined, x="severity", y="event_ai", hue="batch",
        order=present_sev, palette=batch_palette, ax=ax_v,
        size=2.5, alpha=0.40, jitter=True, zorder=4,
        dodge=True, legend=False,
    )
    for thresh, clr in [(0.05, AI_COLORS["ANTIFRAGILE"]),
                        (-0.05, AI_COLORS["FRAGILE"]),
                        (-0.20, AI_COLORS["BRITTLE"])]:
        ax_v.axhline(thresh, color=clr, lw=0.9, linestyle=":", alpha=0.8)
    ax_v.axhline(0, color="#333", lw=1.0, linestyle="-", alpha=0.5)
    ax_v.set_ylabel("Per-Event Antifragility Index")
    ax_v.set_xlabel("Severity")

    # Combined Kruskal–Wallis
    groups = [combined[combined["severity"] == s]["event_ai"].dropna().values
              for s in present_sev if len(combined[combined["severity"] == s]) >= 2]
    if len(groups) >= 2:
        h_stat, p_kw = stats.kruskal(*groups)
        ax_v.text(0.02, 0.02, f"Kruskal–Wallis (all): H={h_stat:.2f}, p={p_kw:.3f}",
                  ha="left", va="bottom", transform=ax_v.transAxes,
                  fontsize=7.5, color="#555")

    # ── Panel B: pre vs post speed comparison ─────────────────────────────────
    ax_sp = axes[1]
    ax_sp.set_title("(b) Pre/Post-Event Speed by Severity & Batch", fontweight="bold", pad=8)

    n_sev = len(present_sev)
    x_base = np.arange(n_sev)
    w = 0.20

    for offset, data, col, lab, fill in [
        (-1.5 * w, merged_a, col_a, label_a, BATCH_FILL["A"]),
        ( 0.5 * w, merged_b, col_b, label_b, BATCH_FILL["B"]),
    ]:
        pre  = data.groupby("severity")["pre_mean_speed_kmh"].mean().reindex(present_sev)
        post = data.groupby("severity")["post_mean_speed_kmh"].mean().reindex(present_sev)
        ax_sp.bar(x_base + offset,       pre.values,  w, color=fill,
                  edgecolor=col, lw=1.2, alpha=0.85,
                  label=f"Batch {lab} — pre")
        ax_sp.bar(x_base + offset + w,   post.values, w, color=col,
                  edgecolor="white", alpha=0.85,
                  label=f"Batch {lab} — post")

    ax_sp.set_xticks(x_base)
    ax_sp.set_xticklabels(present_sev)
    ax_sp.set_ylabel("Mean Speed (km/h)")
    ax_sp.set_xlabel("Severity")
    ax_sp.legend(fontsize=8, ncol=2)

    fig.suptitle(
        f"Per-Event Antifragility — Batch {label_a} vs Batch {label_b}  (Thessaloniki)",
        fontsize=12, fontweight="bold",
    )
    out_path = out_dir / "comparative_per_event_ai.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Comparative Figure 5 — Scatter overlay (AI vs accidents, both batches)
# ---------------------------------------------------------------------------

def figure_comparative_scatter(
    data_a: dict, data_b: dict, out_dir: Path
) -> Path:
    """AI vs accidents scatter with per-batch OLS, category colouring, violin marginals."""
    label_a = data_a["label"]
    label_b = data_b["label"]
    col_a   = BATCH_PALETTE["A"]
    col_b   = BATCH_PALETTE["B"]

    df_a = _annotate_runs_df(data_a).dropna(subset=["antifragility_index"])
    df_b = _annotate_runs_df(data_b).dropna(subset=["antifragility_index"])
    df_a["batch"] = label_a
    df_b["batch"] = label_b
    pd.concat([df_a, df_b], ignore_index=True)

    fig = plt.figure(figsize=(12, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            width_ratios=[4, 1], height_ratios=[1, 4],
                            hspace=0.05, wspace=0.05)

    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    fig.add_subplot(gs[0, 1]).axis("off")

    # Marginal top — accident count KDE per batch
    for df, col, lab in [(df_a, col_a, label_a), (df_b, col_b, label_b)]:
        sns.kdeplot(data=df, x="total_accidents", ax=ax_top,
                    color=col, fill=True, alpha=0.25, lw=1.5, label=f"Batch {lab}")
    ax_top.set_ylabel("Density", fontsize=8)
    ax_top.legend(fontsize=8, loc="upper right")
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # Marginal right — AI KDE per batch
    for df, col in [(df_a, col_a), (df_b, col_b)]:
        sns.kdeplot(data=df, y="antifragility_index", ax=ax_right,
                    color=col, fill=True, alpha=0.25, lw=1.5)
    ax_right.set_xlabel("Density", fontsize=8)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    # Main scatter
    for df, col, marker, lab in [
        (df_a, col_a, "o", label_a),
        (df_b, col_b, "s", label_b),
    ]:
        dot_cols = [AI_COLORS.get(c, "#aaa") for c in df["category"]]
        ax_main.scatter(df["total_accidents"], df["antifragility_index"],
                        c=dot_cols, edgecolors=col, lw=0.8,
                        s=60, alpha=0.75, marker=marker, zorder=3,
                        label=f"Batch {lab}")
        # Per-batch OLS
        m, b, *_ = stats.linregress(df["total_accidents"], df["antifragility_index"])
        x_line = np.linspace(df["total_accidents"].min(), df["total_accidents"].max(), 100)
        ax_main.plot(x_line, m * x_line + b, color=col, lw=1.8, linestyle="--",
                     label=f"OLS {lab}: {m:+.4f}x{b:+.3f}")

    # Threshold lines
    for thresh, clr in [(0.05, AI_COLORS["ANTIFRAGILE"]),
                        (-0.05, AI_COLORS["FRAGILE"]),
                        (-0.20, AI_COLORS["BRITTLE"])]:
        ax_main.axhline(thresh, color=clr, lw=0.8, linestyle=":", alpha=0.7)

    # AI=0 reference
    ax_main.axhline(0, color="#555", lw=0.8, linestyle="-", alpha=0.4)

    ax_main.set_xlabel("Total Accidents per Run", fontsize=11)
    ax_main.set_ylabel("Antifragility Index", fontsize=11)
    ax_main.legend(fontsize=8, loc="upper right", ncol=2)

    # Category legend patches
    cat_patches = [mpatches.Patch(color=AI_COLORS[c], label=c) for c in AI_THRESHOLDS]
    ax_main.legend(handles=cat_patches, fontsize=7.5, loc="lower right",
                   title="AI category", title_fontsize=8)

    fig.suptitle(
        f"AI vs Accidents — Batch {label_a} vs Batch {label_b}  (Thessaloniki)",
        fontsize=12, fontweight="bold",
    )
    out_path = out_dir / "comparative_scatter.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Per-batch pipeline
# ---------------------------------------------------------------------------

def run_single_batch(
    data: dict,
    out_dir: Path,
    net_segments: list[np.ndarray] | None,
    net_extent: tuple[float, ...] | None,
) -> None:
    """Generate all five per-batch figures."""
    print_statistical_summary(data)
    print("  Generating figures...")
    figure_resilience_statistics(data, out_dir)
    figure_network_dynamics(data, out_dir)
    figure_accident_characteristics(data, out_dir)
    figure_spatial_heatmap(data, out_dir, net_segments, net_extent)
    figure_per_event_ai(data, out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse and compare SAS batch runs.")
    parser.add_argument("--batch-dir",    type=Path, default=BATCH_A_DEFAULT,
                        help="Primary batch results directory")
    parser.add_argument("--compare-dir",  type=Path, default=BATCH_B_DEFAULT,
                        help="Second batch for comparative analysis (optional)")
    parser.add_argument("--net-xml",      type=Path, default=NET_XML_DEFAULT,
                        help="Path to SUMO .net.xml for network underlay")
    parser.add_argument("--no-compare",   action="store_true",
                        help="Skip comparative figures even if --compare-dir is set")
    args = parser.parse_args()

    # ── Load network geometry ─────────────────────────────────────────────────
    net_segments: list[np.ndarray] | None = None
    net_extent:   tuple[float, ...] | None = None
    if args.net_xml.exists():
        print(f"\nLoading network from: {args.net_xml}")
        net_segments, net_extent = load_network_shapes(args.net_xml)
        print(f"  {len(net_segments)} road segments loaded")
    else:
        print(f"[WARN] Network XML not found: {args.net_xml}  (spatial overlay disabled)")

    # ── Primary batch ─────────────────────────────────────────────────────────
    if not args.batch_dir.exists():
        print(f"ERROR: batch directory not found: {args.batch_dir}")
        raise SystemExit(1)

    print(f"\nLoading primary batch: {args.batch_dir}")
    data_a   = load_batch_data(args.batch_dir)
    out_dir_a = args.batch_dir / "analysis"
    out_dir_a.mkdir(exist_ok=True)
    run_single_batch(data_a, out_dir_a, net_segments, net_extent)
    print(f"  All figures → {out_dir_a}\n")

    # ── Comparison batch ──────────────────────────────────────────────────────
    if args.no_compare or not args.compare_dir or not args.compare_dir.exists():
        if not args.no_compare and args.compare_dir and not args.compare_dir.exists():
            print(f"[WARN] Compare dir not found: {args.compare_dir}")
        return

    print(f"Loading comparison batch: {args.compare_dir}")
    data_b    = load_batch_data(args.compare_dir)
    out_dir_b = args.compare_dir / "analysis"
    out_dir_b.mkdir(exist_ok=True)
    run_single_batch(data_b, out_dir_b, net_segments, net_extent)
    print(f"  All figures → {out_dir_b}\n")

    # ── Comparative figures ───────────────────────────────────────────────────
    label_a = data_a["label"]
    label_b = data_b["label"]
    comp_dir = args.batch_dir.parent / f"comparative_{label_a}_vs_{label_b}"
    comp_dir.mkdir(exist_ok=True)
    print(f"Generating comparative figures → {comp_dir}")

    figure_comparative_overview(data_a, data_b, comp_dir)
    figure_comparative_dynamics(data_a, data_b, comp_dir)
    figure_comparative_spatial(data_a, data_b, comp_dir, net_segments, net_extent)
    figure_comparative_per_event_ai(data_a, data_b, comp_dir)
    figure_comparative_scatter(data_a, data_b, comp_dir)

    print(f"\nDone.  Comparative figures → {comp_dir}\n")


if __name__ == "__main__":
    main()
