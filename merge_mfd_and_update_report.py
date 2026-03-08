"""merge_mfd_and_update_report.py
================================

Merges mfd_data.csv from a supplementary run (e.g. p=0.1 and p=0.3) into
the main assessment directory, regenerates the MFD figures, and rebuilds
the full HTML resilience report — no re-simulation required.

Usage
-----
After running the supplementary scenarios into a separate output directory:

    python merge_mfd_and_update_report.py \\
        --main  results/resilience_2026-03-06_1418 \\
        --extra results/resilience_low_demand_0p1_0p3

The script:
  1. Merges mfd_data.csv (extra rows prepended so density axis is ordered)
  2. Saves merged CSV back to the main directory
  3. Regenerates mfd_density_flow.png and mfd_density_speed.png
  4. Reconstructs all required objects from JSON files on disk
  5. Regenerates resilience_report.html with the updated MFD figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _reconstruct_resilience_score(score_path: Path):
    """Rebuild a ResilienceScore dataclass from the saved JSON."""
    from mfd_analysis import EdgeVulnerability, ResilienceScore

    with open(score_path, encoding="utf-8") as f:
        d = json.load(f)

    weak_points = [
        EdgeVulnerability(
            edge_id=wp["edge_id"],
            x=wp["x"],
            y=wp["y"],
            accident_count=wp["accident_count"],
            mean_duration_seconds=wp["mean_duration_seconds"],
            mean_vehicles_affected=wp["mean_vehicles_affected"],
            mean_speed_drop_ratio=wp["mean_speed_drop_ratio"],
            edge_importance=wp["edge_importance"],
            vulnerability_index=wp["vulnerability_index"],
        )
        for wp in d.get("weak_points", [])
    ]

    return ResilienceScore(
        overall_score=d["overall_score"],
        grade=d["grade"],
        interpretation=d["interpretation"],
        speed_resilience=d["speed_resilience"],
        throughput_resilience=d["throughput_resilience"],
        recovery_resilience=d["recovery_resilience"],
        robustness=d["robustness"],
        weights=d["weights"],
        mfd_parameters=d["mfd_parameters"],
        ai_aggregate=d["ai_aggregate"],
        weak_points=weak_points,
    )


def _reconstruct_scenarios(matrix_path: Path) -> list:
    """Rebuild Scenario dataclass objects from the saved scenario_matrix.json."""
    from scenario_generator import Scenario

    with open(matrix_path, encoding="utf-8") as f:
        d = json.load(f)

    return [
        Scenario(
            scenario_id=s["scenario_id"],
            scenario_type=s["scenario_type"],
            period=s["period"],
            seed=s["seed"],
            base_probability=s["base_probability"],
            output_folder=s["output_folder"],
            sumocfg_path=s["sumocfg_path"],
        )
        for s in d.get("scenarios", [])
    ]


def _load_all_results(agg_path: Path) -> list[dict]:
    """Load per-scenario result summaries from aggregate_summary.json."""
    with open(agg_path, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("results", [])


def _build_figures_dict(figures_dir: Path) -> dict[str, str]:
    """Collect all figure PNG paths that exist on disk."""
    known = [
        "mfd_density_flow",
        "mfd_density_speed",
        "mfd_theoretical",
        "resilience_statistics",
        "network_dynamics",
        "accident_characteristics",
        "per_event_ai",
        "spatial_heatmap",
    ]
    out: dict[str, str] = {}
    for name in known:
        p = figures_dir / f"{name}.png"
        if p.exists():
            out[name] = str(p)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge MFD data from a supplementary run into the main assessment "
            "and regenerate MFD figures + HTML report."
        )
    )
    parser.add_argument(
        "--main",
        required=True,
        metavar="DIR",
        help="Main assessment output directory (e.g. results/resilience_2026-03-06_1418)",
    )
    parser.add_argument(
        "--extra",
        required=True,
        metavar="DIR",
        help="Supplementary run directory containing additional mfd_data.csv",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to base config.yaml (default: config.yaml)",
    )
    args = parser.parse_args()

    main_dir = Path(args.main)
    extra_dir = Path(args.extra)

    # ── Validate directories ──────────────────────────────────────────────
    for d, label in [(main_dir, "--main"), (extra_dir, "--extra")]:
        if not d.is_dir():
            print(f"ERROR: {label} directory not found: {d}")
            sys.exit(1)

    main_mfd_path = main_dir / "mfd_data.csv"
    extra_mfd_path = extra_dir / "mfd_data.csv"

    for p, label in [(main_mfd_path, "main mfd_data.csv"), (extra_mfd_path, "extra mfd_data.csv")]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)

    # ── Step 1: Merge mfd_data.csv ────────────────────────────────────────
    print("\n[1/4] Merging mfd_data.csv ...")
    main_mfd = pd.read_csv(main_mfd_path)
    extra_mfd = pd.read_csv(extra_mfd_path)

    main_periods = sorted(main_mfd["period"].unique())
    extra_periods = sorted(extra_mfd["period"].unique())
    print(f"  Main  : {len(main_mfd):,} rows  |  demand levels: {main_periods}")
    print(f"  Extra : {len(extra_mfd):,} rows  |  demand levels: {extra_periods}")

    # Remove any rows from main that cover the same demand levels as extra
    # so the script is safe to run multiple times without double-counting.
    overlap = set(extra_mfd["period"].unique()) & set(main_mfd["period"].unique())
    if overlap:
        print(f"  Removing {len(main_mfd[main_mfd['period'].isin(overlap)]):,} existing rows "
              f"for overlapping period(s) {sorted(overlap)} from main before merging.")
        main_mfd = main_mfd[~main_mfd["period"].isin(overlap)]

    # Concatenate (extra first so the density axis is ordered low → high demand).
    combined = pd.concat([extra_mfd, main_mfd], ignore_index=True)
    combined = combined.sort_values("period").reset_index(drop=True)
    combined.to_csv(main_mfd_path, index=False)
    print(
        f"  Combined: {len(combined):,} rows saved → {main_mfd_path}\n"
        f"  All demand levels: {sorted(combined['period'].unique())}"
    )

    # ── Step 2: Regenerate MFD figures ───────────────────────────────────
    print("\n[2/4] Regenerating MFD figures ...")
    sys.path.insert(0, str(Path(__file__).parent))
    from mfd_analysis import (
        fit_greenshields_per_scenario_type,
        plot_mfd_density_flow,
        plot_mfd_density_speed,
        plot_mfd_theoretical,
    )

    figures_dir = main_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    for fn, label in [
        (lambda: plot_mfd_density_flow(combined, str(figures_dir)), "mfd_density_flow.png"),
        (lambda: plot_mfd_density_speed(combined, str(figures_dir)), "mfd_density_speed.png"),
        (lambda: plot_mfd_theoretical(combined, str(figures_dir)), "mfd_theoretical.png"),
    ]:
        try:
            fn()
            print(f"  ✓  {label}")
        except Exception as exc:
            print(f"  ✗  {label} failed: {exc}")

    # Fit per-type Greenshields and save JSON to main dir.
    per_type_fits: dict | None = None
    try:
        per_type_fits = fit_greenshields_per_scenario_type(combined)
        fits_path = main_dir / "mfd_per_type_fits.json"
        import json as _json
        fits_path.write_text(_json.dumps(per_type_fits, indent=2), encoding="utf-8")
        print(f"  ✓  mfd_per_type_fits.json  ({len(per_type_fits)} scenario types)")
    except Exception as exc:
        print(f"  ✗  per-type Greenshields fit failed: {exc}")

    # ── Step 3: Reconstruct objects from saved JSON ───────────────────────
    print("\n[3/4] Reconstructing assessment objects from saved JSON ...")

    score_path = main_dir / "resilience_score.json"
    matrix_path = main_dir / "scenario_matrix.json"
    agg_path = main_dir / "aggregate" / "aggregate_summary.json"

    for p, label in [
        (score_path, "resilience_score.json"),
        (matrix_path, "scenario_matrix.json"),
        (agg_path, "aggregate/aggregate_summary.json"),
    ]:
        if not p.exists():
            print(f"  ERROR: {label} not found: {p}")
            sys.exit(1)

    resilience = _reconstruct_resilience_score(score_path)
    print(f"  ✓  ResilienceScore: {resilience.grade}  ({resilience.overall_score:.4f})")

    scenarios = _reconstruct_scenarios(matrix_path)
    print(f"  ✓  Scenarios: {len(scenarios)} loaded")

    all_results = _load_all_results(agg_path)
    print(f"  ✓  Results: {len(all_results)} loaded")

    config = _load_config(args.config)
    print(f"  ✓  Config loaded from {args.config}")

    figures = _build_figures_dict(figures_dir)
    print(f"  ✓  Figures found on disk: {sorted(figures.keys())}")

    # Load Claude analysis if present.
    claude_path = main_dir / "assessment_analysis.md"
    claude_analysis: str | None = None
    if claude_path.exists():
        claude_analysis = claude_path.read_text(encoding="utf-8")
        print("  ✓  Claude analysis loaded")

    # ── Step 4: Regenerate HTML report ───────────────────────────────────
    print("\n[4/4] Regenerating HTML resilience report ...")
    from resilience_report import generate_resilience_report

    report_path = generate_resilience_report(
        str(main_dir),
        resilience,
        scenarios,
        all_results,
        config,
        figures,
        claude_analysis=claude_analysis,
        per_type_fits=per_type_fits,
    )
    print(f"  ✓  Report → {report_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  MFD MERGE COMPLETE")
    print(f"  Added demand levels : {extra_periods}")
    print(f"  Total demand levels : {sorted(combined['period'].unique())}")
    print(f"  Total MFD data rows : {len(combined):,}")
    print(f"  Report              : {report_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
