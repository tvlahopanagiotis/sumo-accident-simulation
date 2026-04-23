"""
compare_seattle_real.py
=======================
Compare Seattle simulation accident outputs against real Seattle collision data.

Inputs:
  - Simulation run directory (single run or batch root with run_*/ subfolders)
  - Seattle real collisions CSV (SDOT collisions all years)

Outputs:
  - comparison_dashboard.png
  - spatial_density_comparison.png
  - comparison_summary.json

Example:
  sas-compare-seattle-real \
      --sim-dir results/Seattle_Batch_2026-03-06_11:10 \
      --real-csv data/cities/seattle/bundle/crash_data/sdot_collisions_all_years.csv
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..app.config import PROJECT_ROOT

REPO_ROOT = PROJECT_ROOT

SEVERITY_CATEGORIES = ["PDO", "Injury", "Serious/Fatal", "Unknown"]
SIM_SEVERITY_MAP = {
    "MINOR": "PDO",
    "MODERATE": "Injury",
    "MAJOR": "Serious/Fatal",
    "CRITICAL": "Serious/Fatal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Seattle simulation accidents with Seattle historical collisions "
            "and generate comparative figures."
        )
    )
    parser.add_argument(
        "--sim-dir",
        required=True,
        help="Simulation run directory (single run or batch root containing run_*/)",
    )
    parser.add_argument(
        "--real-csv",
        default=str(
            REPO_ROOT
            / "data"
            / "cities"
            / "seattle"
            / "bundle"
            / "crash_data"
            / "sdot_collisions_all_years.csv"
        ),
        help="Path to Seattle real collisions CSV",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for comparative figures (default: <sim-dir>/real_comparison)",
    )
    parser.add_argument(
        "--start-hour",
        type=float,
        default=8.0,
        help="Clock hour used as simulation start (for matching real window, default: 8.0)",
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=None,
        help=(
            "Comparison window length in hours. "
            "Default: inferred from simulation metadata, fallback to 2.0"
        ),
    )
    parser.add_argument(
        "--year-from",
        type=int,
        default=None,
        help="Lower year bound for real data (inclusive)",
    )
    parser.add_argument(
        "--year-to",
        type=int,
        default=None,
        help="Upper year bound for real data (inclusive)",
    )
    parser.add_argument(
        "--bin-minutes",
        type=int,
        default=15,
        help="Bin width for temporal profile in minutes (default: 15)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=24,
        help="Grid size for spatial concentration analysis (default: 24)",
    )
    return parser.parse_args()


def discover_run_dirs(sim_dir: Path) -> list[Path]:
    if (sim_dir / "accident_reports.json").exists():
        return [sim_dir]
    runs = sorted(
        p for p in sim_dir.glob("run_*") if p.is_dir() and (p / "accident_reports.json").exists()
    )
    return runs


def load_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def infer_window_hours(run_dirs: list[Path]) -> float:
    durations_s: list[float] = []
    for run_dir in run_dirs:
        meta = load_metadata(run_dir)
        cfg = meta.get("config", {})
        steps = cfg.get("sumo", {}).get("total_steps")
        if isinstance(steps, (int, float)) and steps > 0:
            durations_s.append(float(steps))
    if durations_s:
        return float(np.median(durations_s) / 3600.0)
    return 2.0


def load_simulation_accidents(run_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        run_id = run_dir.name
        meta = load_metadata(run_dir)
        run_duration_s = None
        if "config" in meta:
            run_duration_s = meta.get("config", {}).get("sumo", {}).get("total_steps")

        report_path = run_dir / "accident_reports.json"
        with open(report_path, encoding="utf-8") as f:
            accidents = json.load(f)

        run_rows.append(
            {
                "run_id": run_id,
                "accident_count": int(len(accidents)),
                "run_duration_s": float(run_duration_s) if run_duration_s else np.nan,
            }
        )

        for rec in accidents:
            loc = rec.get("location", {}) or {}
            rows.append(
                {
                    "run_id": run_id,
                    "accident_id": rec.get("accident_id"),
                    "severity_sim": rec.get("severity", "UNKNOWN"),
                    "severity_group": SIM_SEVERITY_MAP.get(rec.get("severity", ""), "Unknown"),
                    "trigger_s": float(rec.get("trigger_step", np.nan)),
                    "x": float(loc.get("x", np.nan)),
                    "y": float(loc.get("y", np.nan)),
                }
            )

    sim_acc = pd.DataFrame(rows)
    run_stats = pd.DataFrame(run_rows)
    return sim_acc, run_stats


def map_real_severity(desc: pd.Series) -> pd.Series:
    d = desc.fillna("").str.lower()
    out = pd.Series(index=desc.index, data="Unknown", dtype="object")
    out[d.str.contains("property damage", na=False)] = "PDO"
    out[d.str.contains("serious injury", na=False) | d.str.contains("fatal", na=False)] = (
        "Serious/Fatal"
    )
    out[d.str.contains("injury collision", na=False)] = "Injury"
    return out


def parse_mixed_datetime(series: pd.Series, formats: list[str]) -> pd.Series:
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for fmt in formats:
        parsed = parsed.fillna(pd.to_datetime(series, format=fmt, errors="coerce"))
    # Final generic fallback for any uncommon formats.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = parsed.fillna(pd.to_datetime(series, errors="coerce"))
    return parsed


def load_real_collisions(real_csv: Path, year_from: int | None, year_to: int | None) -> pd.DataFrame:
    usecols = ["INCDTTM", "INCDATE", "SEVERITYDESC", "x", "y"]
    real = pd.read_csv(real_csv, usecols=usecols, low_memory=False)

    dt = parse_mixed_datetime(
        real["INCDTTM"],
        formats=[
            "%m/%d/%Y %I:%M:%S %p",
            "%m/%d/%Y %I:%M %p",
            "%m/%d/%Y",
        ],
    )
    dt_date = parse_mixed_datetime(
        real["INCDATE"],
        formats=[
            "%m/%d/%Y %I:%M:%S %p",
            "%m/%d/%Y",
        ],
    )
    dt = dt.fillna(dt_date)
    real = real.assign(dt=dt).dropna(subset=["dt"]).copy()

    if year_from is not None:
        real = real[real["dt"].dt.year >= year_from]
    if year_to is not None:
        real = real[real["dt"].dt.year <= year_to]

    real["severity_group"] = map_real_severity(real["SEVERITYDESC"])
    real["hour_float"] = (
        real["dt"].dt.hour + real["dt"].dt.minute / 60.0 + real["dt"].dt.second / 3600.0
    )
    real["date"] = real["dt"].dt.date
    return real


def compute_window_offsets(hour_float: pd.Series, start_hour: float) -> pd.Series:
    return (hour_float - start_hour) % 24.0


def category_shares(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    counts = series.value_counts().reindex(SEVERITY_CATEGORIES, fill_value=0)
    total = counts.sum()
    if total > 0:
        shares = counts / total
    else:
        shares = counts.astype(float)
    return counts, shares


def normalize_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return np.array([]), np.array([])

    x_q1, x_q99 = np.quantile(x, [0.01, 0.99])
    y_q1, y_q99 = np.quantile(y, [0.01, 0.99])
    if x_q99 <= x_q1:
        x_q1, x_q99 = float(np.min(x)), float(np.max(x) + 1e-9)
    if y_q99 <= y_q1:
        y_q1, y_q99 = float(np.min(y)), float(np.max(y) + 1e-9)

    x_n = np.clip((x - x_q1) / (x_q99 - x_q1), 0.0, 1.0 - 1e-12)
    y_n = np.clip((y - y_q1) / (y_q99 - y_q1), 0.0, 1.0 - 1e-12)
    return x_n, y_n


def concentration_curve(x: np.ndarray, y: np.ndarray, grid_size: int) -> tuple[np.ndarray, np.ndarray] | None:
    x_n, y_n = normalize_xy(x, y)
    if len(x_n) < max(20, grid_size):
        return None

    xi = (x_n * grid_size).astype(int)
    yi = (y_n * grid_size).astype(int)
    cell = xi + yi * grid_size
    counts = np.bincount(cell, minlength=grid_size * grid_size)
    counts_sorted = np.sort(counts)[::-1]
    cum_acc = np.cumsum(counts_sorted) / counts_sorted.sum()
    frac_cells = np.arange(1, len(counts_sorted) + 1) / len(counts_sorted)
    return frac_cells, cum_acc


def interp_top_share(frac_cells: np.ndarray, cum_acc: np.ndarray, frac: float) -> float:
    if len(frac_cells) == 0:
        return float("nan")
    return float(np.interp(frac, frac_cells, cum_acc))


def json_number(value: float) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)) and not np.isfinite(value):
        return None
    return float(value)


def plot_dashboard(
    out_path: Path,
    run_counts: np.ndarray,
    real_daily_counts: np.ndarray,
    temporal_centers: np.ndarray,
    sim_temporal_share: np.ndarray,
    real_temporal_share: np.ndarray,
    sim_severity_share: pd.Series,
    real_severity_share: pd.Series,
    curve_sim: tuple[np.ndarray, np.ndarray] | None,
    curve_real: tuple[np.ndarray, np.ndarray] | None,
    start_hour: float,
    window_hours: float,
) -> None:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Panel 1: count distributions
    ax1 = fig.add_subplot(gs[0, 0])
    if len(real_daily_counts) > 0:
        ax1.boxplot(
            [real_daily_counts, run_counts],
            tick_labels=[
                f"Real daily\n({len(real_daily_counts)} days)",
                f"Simulation\n({len(run_counts)} runs)",
            ],
            showmeans=True,
        )
        ax1.scatter(
            np.random.default_rng(42).normal(1, 0.03, size=len(real_daily_counts)),
            real_daily_counts,
            s=8,
            alpha=0.18,
            color="#1f77b4",
        )
        ax1.scatter(
            np.random.default_rng(7).normal(2, 0.03, size=len(run_counts)),
            run_counts,
            s=18,
            alpha=0.85,
            color="#d62728",
        )
    ax1.set_title("Accident Count Distribution")
    ax1.set_ylabel("Accidents in window")
    ax1.grid(alpha=0.25)

    # Panel 2: temporal profile
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(temporal_centers, real_temporal_share * 100.0, marker="o", label="Real")
    ax2.plot(temporal_centers, sim_temporal_share * 100.0, marker="o", label="Simulation")
    ax2.set_title("Within-Window Temporal Profile")
    ax2.set_xlabel("Hours since window start")
    ax2.set_ylabel("Share of accidents (%)")
    ax2.grid(alpha=0.25)
    ax2.legend()

    # Panel 3: severity mix
    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(SEVERITY_CATEGORIES))
    width = 0.38
    ax3.bar(
        x - width / 2,
        real_severity_share.reindex(SEVERITY_CATEGORIES).values * 100.0,
        width,
        label="Real",
    )
    ax3.bar(
        x + width / 2,
        sim_severity_share.reindex(SEVERITY_CATEGORIES).values * 100.0,
        width,
        label="Simulation",
    )
    ax3.set_xticks(x)
    ax3.set_xticklabels(SEVERITY_CATEGORIES, rotation=20)
    ax3.set_ylabel("Share (%)")
    ax3.set_title("Severity Mix Comparison")
    ax3.grid(alpha=0.25, axis="y")
    ax3.legend()

    # Panel 4: spatial concentration
    ax4 = fig.add_subplot(gs[1, 1])
    if curve_real is not None:
        ax4.plot(curve_real[0] * 100.0, curve_real[1] * 100.0, label="Real")
    if curve_sim is not None:
        ax4.plot(curve_sim[0] * 100.0, curve_sim[1] * 100.0, label="Simulation")
    ax4.plot([0, 100], [0, 100], linestyle="--", color="gray", linewidth=1, label="Uniform")
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 100)
    ax4.set_xlabel("Top spatial cells (%)")
    ax4.set_ylabel("Accidents captured (%)")
    ax4.set_title("Spatial Concentration Curve")
    ax4.grid(alpha=0.25)
    ax4.legend()

    fig.suptitle(
        (
            "Seattle Simulation vs Real Collisions\n"
            f"Real window: start={start_hour:.2f}h, duration={window_hours:.2f}h"
        ),
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_density(
    out_path: Path,
    sim_xy: tuple[np.ndarray, np.ndarray],
    real_xy: tuple[np.ndarray, np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    datasets = [("Simulation", sim_xy), ("Real", real_xy)]

    for ax, (label, (x, y)) in zip(axes, datasets, strict=False):
        if len(x) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(label)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue
        hb = ax.hexbin(x, y, gridsize=30, cmap="viridis", mincnt=1, bins="log")
        ax.set_title(f"{label} Normalized Density")
        ax.set_xlabel("Normalized X")
        ax.set_ylabel("Normalized Y")
        fig.colorbar(hb, ax=ax, label="log10(count)")

    fig.suptitle("Spatial Accident Density (Each Dataset Normalized Separately)")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    sim_dir = Path(args.sim_dir).resolve()
    real_csv = Path(args.real_csv).resolve()
    if not sim_dir.exists():
        raise FileNotFoundError(f"Simulation directory not found: {sim_dir}")
    if not real_csv.exists():
        raise FileNotFoundError(f"Real CSV not found: {real_csv}")

    run_dirs = discover_run_dirs(sim_dir)
    if not run_dirs:
        raise FileNotFoundError(
            f"No run folders with accident_reports.json found under: {sim_dir}"
        )

    out_dir = Path(args.out_dir).resolve() if args.out_dir else sim_dir / "real_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_acc, run_stats = load_simulation_accidents(run_dirs)
    if sim_acc.empty:
        raise RuntimeError("Simulation data has no accidents; comparison requires accident_reports.json content.")

    window_hours = args.window_hours
    if window_hours is None:
        window_hours = infer_window_hours(run_dirs)
    if window_hours <= 0 or window_hours > 24:
        raise ValueError("window_hours must be in (0, 24].")

    real = load_real_collisions(real_csv, args.year_from, args.year_to)
    real["window_h"] = compute_window_offsets(real["hour_float"], args.start_hour)
    real_win = real[real["window_h"] < window_hours].copy()
    if real_win.empty:
        raise RuntimeError("No real collisions remain after year/window filtering.")

    sim_acc["window_h"] = sim_acc["trigger_s"] / 3600.0
    sim_acc = sim_acc[sim_acc["window_h"] < window_hours].copy()
    if sim_acc.empty:
        raise RuntimeError(
            "No simulated accidents fall inside comparison window. "
            "Try increasing --window-hours."
        )

    run_counts = run_stats["accident_count"].to_numpy(dtype=float)
    real_daily_counts = real_win.groupby("date").size().to_numpy(dtype=float)

    bins = max(1, int(round(window_hours * 60.0 / args.bin_minutes)))
    bin_edges = np.linspace(0.0, window_hours, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    sim_hist, _ = np.histogram(sim_acc["window_h"].to_numpy(), bins=bin_edges)
    real_hist, _ = np.histogram(real_win["window_h"].to_numpy(), bins=bin_edges)
    sim_temporal_share = sim_hist / sim_hist.sum() if sim_hist.sum() > 0 else np.zeros_like(sim_hist, float)
    real_temporal_share = (
        real_hist / real_hist.sum() if real_hist.sum() > 0 else np.zeros_like(real_hist, float)
    )

    sim_sev_counts, sim_sev_share = category_shares(sim_acc["severity_group"])
    real_sev_counts, real_sev_share = category_shares(real_win["severity_group"])

    curve_sim = concentration_curve(
        sim_acc["x"].to_numpy(dtype=float),
        sim_acc["y"].to_numpy(dtype=float),
        grid_size=args.grid_size,
    )
    curve_real = concentration_curve(
        real_win["x"].to_numpy(dtype=float),
        real_win["y"].to_numpy(dtype=float),
        grid_size=args.grid_size,
    )

    dashboard_path = out_dir / "comparison_dashboard.png"
    plot_dashboard(
        out_path=dashboard_path,
        run_counts=run_counts,
        real_daily_counts=real_daily_counts,
        temporal_centers=bin_centers,
        sim_temporal_share=sim_temporal_share,
        real_temporal_share=real_temporal_share,
        sim_severity_share=sim_sev_share,
        real_severity_share=real_sev_share,
        curve_sim=curve_sim,
        curve_real=curve_real,
        start_hour=float(args.start_hour),
        window_hours=float(window_hours),
    )

    sim_xy = normalize_xy(sim_acc["x"].to_numpy(dtype=float), sim_acc["y"].to_numpy(dtype=float))
    real_xy = normalize_xy(real_win["x"].to_numpy(dtype=float), real_win["y"].to_numpy(dtype=float))
    spatial_path = out_dir / "spatial_density_comparison.png"
    plot_spatial_density(spatial_path, sim_xy=sim_xy, real_xy=real_xy)

    top10_sim = float("nan")
    top10_real = float("nan")
    if curve_sim is not None:
        top10_sim = interp_top_share(curve_sim[0], curve_sim[1], 0.10)
    if curve_real is not None:
        top10_real = interp_top_share(curve_real[0], curve_real[1], 0.10)

    summary = {
        "inputs": {
            "sim_dir": str(sim_dir),
            "real_csv": str(real_csv),
            "n_runs": int(len(run_dirs)),
            "window_start_hour": float(args.start_hour),
            "window_hours": float(window_hours),
            "year_from": args.year_from,
            "year_to": args.year_to,
            "bin_minutes": int(args.bin_minutes),
            "grid_size": int(args.grid_size),
        },
        "simulation": {
            "n_accidents_in_window": int(len(sim_acc)),
            "mean_accidents_per_run": json_number(float(np.mean(run_counts))) if len(run_counts) else None,
            "std_accidents_per_run": json_number(float(np.std(run_counts))) if len(run_counts) else None,
            "severity_counts": sim_sev_counts.to_dict(),
            "severity_shares": sim_sev_share.to_dict(),
            "top10pct_cell_capture": json_number(top10_sim),
        },
        "real": {
            "n_accidents_in_window": int(len(real_win)),
            "n_days": int(real_win["date"].nunique()),
            "mean_accidents_per_day_window": json_number(float(np.mean(real_daily_counts)))
            if len(real_daily_counts)
            else None,
            "std_accidents_per_day_window": json_number(float(np.std(real_daily_counts)))
            if len(real_daily_counts)
            else None,
            "severity_counts": real_sev_counts.to_dict(),
            "severity_shares": real_sev_share.to_dict(),
            "top10pct_cell_capture": json_number(top10_real),
        },
        "outputs": {
            "dashboard": str(dashboard_path),
            "spatial_density": str(spatial_path),
        },
    }

    summary_path = out_dir / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 68)
    print("Seattle simulation vs real collisions comparison complete")
    print(f"  Runs loaded         : {len(run_dirs)}")
    print(f"  Sim accidents       : {len(sim_acc)}")
    print(f"  Real accidents      : {len(real_win)}")
    print(f"  Output folder       : {out_dir}")
    print(f"  - {dashboard_path.name}")
    print(f"  - {spatial_path.name}")
    print(f"  - {summary_path.name}")
    print("=" * 68)


if __name__ == "__main__":
    main()
