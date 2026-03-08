"""
resilience_assessment.py
========================

One-Click Resilience Assessment — CLI entry point and orchestrator.

Runs a comprehensive scenario matrix (demand levels × incident configs × seeds)
in parallel, computes the Macroscopic Fundamental Diagram, identifies network
weak points, produces a composite resilience score, and generates a self-
contained HTML report.

Figures use the same publication-quality style as analyse_batch.py.
After all results are collected a Claude analysis is requested via the
Anthropic API (requires ANTHROPIC_API_KEY env var; gracefully skipped if absent).

Usage
-----
Full assessment (105 scenarios, ~1–2 hours):
    python resilience_assessment.py

Quick smoke test (12 scenarios, ~15 min):
    python resilience_assessment.py --quick

Custom parameters:
    python resilience_assessment.py --workers 4 --demand-levels 1.0 2.0 5.0 --seeds 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("sas.resilience")

# ---------------------------------------------------------------------------
# Model used for AI-assisted analysis
# (override with env var SAS_CLAUDE_MODEL)
# ---------------------------------------------------------------------------
_DEFAULT_CLAUDE_MODEL = os.environ.get("SAS_CLAUDE_MODEL", "claude-opus-4-5")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resilience_assessment.py",
        description=(
            "SAS One-Click Resilience Assessment\n\n"
            "Runs a comprehensive scenario matrix, computes the MFD,\n"
            "identifies weak points, and generates a resilience report.\n\n"
            "This is a one-command assessment: it generates routes,\n"
            "runs all scenarios in parallel, and produces a final report."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="FILE",
        help="Path to base YAML config (default: config.yaml)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel SUMO instances (default: auto-detect)",
    )
    parser.add_argument(
        "--demand-levels",
        type=float,
        nargs="+",
        default=None,
        metavar="PERIOD",
        help="Custom demand levels as period values (e.g., 0.5 1.0 2.0 5.0)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        metavar="N",
        help="Number of seeds per scenario (default: 3, quick: 2)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 3 demand levels, 2 incident configs, 2 seeds (12 runs)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Output directory (default: results/resilience_YYYY-MM-DD_HH:MM/)",
    )
    parser.add_argument(
        "--skip-routes",
        action="store_true",
        help="Skip route generation (use existing route files)",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=10000,
        help="Base TCP port for TraCI connections (default: 10000)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser


# ---------------------------------------------------------------------------
# Batch data builder
# ---------------------------------------------------------------------------


def _build_assessment_batch_data(
    scenarios: list,
    all_results: list[dict],
    label: str = "Resilience Assessment",
) -> dict:
    """
    Build a data dict compatible with analyse_batch.py figure functions
    from assessment scenario results.

    Only non-baseline scenarios with status='success' are included as 'runs',
    so that AI distributions and accident figures are meaningful.

    Args:
        scenarios:   List of Scenario objects from the matrix.
        all_results: List of result dicts from parallel execution.
        label:       Short human-readable label for figure titles.

    Returns:
        Dict with keys: aggregate, runs, metrics, accidents, ai_events, label.
    """
    from scipy import stats as sp_stats

    metrics_frames: list[pd.DataFrame] = []
    accident_frames: list[pd.DataFrame] = []
    ai_event_frames: list[pd.DataFrame] = []
    runs_list: list[dict] = []

    run_idx = 0
    for s, r in zip(scenarios, all_results, strict=False):
        # Skip failed runs and baseline (no incidents → no AI signal)
        if r.get("status") != "success":
            continue
        if s.scenario_type == "baseline":
            continue

        summary = r.get("summary", {})
        ai = summary.get("antifragility_index")

        runs_list.append(
            {
                "scenario_id": r.get("scenario_id", ""),
                "scenario_type": r.get("scenario_type", ""),
                "period": r.get("period", 0.0),
                "antifragility_index": ai,
                "total_accidents": summary.get("total_accidents", 0),
                "interpretation": summary.get("interpretation", ""),
            }
        )

        # --- network_metrics.csv ---
        csv_path = os.path.join(s.output_folder, "network_metrics.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["seed"] = s.seed
            df["run_idx"] = run_idx
            metrics_frames.append(df)

        # --- accident_reports.json ---
        acc_path = os.path.join(s.output_folder, "accident_reports.json")
        if os.path.exists(acc_path):
            with open(acc_path) as f:
                reports = json.load(f)
            if reports:
                df_acc = pd.json_normalize(reports)
                df_acc["seed"] = s.seed
                df_acc["run_idx"] = run_idx
                accident_frames.append(df_acc)

        # --- antifragility_index.json (per-event) ---
        ai_path = os.path.join(s.output_folder, "antifragility_index.json")
        if os.path.exists(ai_path):
            with open(ai_path) as f:
                ai_data = json.load(f)
            per_event = ai_data.get("per_event", [])
            if per_event:
                df_ai = pd.DataFrame(per_event)
                df_ai["seed"] = s.seed
                df_ai["run_idx"] = run_idx
                ai_event_frames.append(df_ai)

        run_idx += 1

    metrics = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    accidents = (
        pd.concat(accident_frames, ignore_index=True) if accident_frames else pd.DataFrame()
    )
    ai_events = (
        pd.concat(ai_event_frames, ignore_index=True) if ai_event_frames else pd.DataFrame()
    )

    # --- Aggregate statistics ---
    ai_vals = [r["antifragility_index"] for r in runs_list if r["antifragility_index"] is not None]
    acc_vals = [r["total_accidents"] for r in runs_list]
    n = len(runs_list)

    ai_mean = float(np.mean(ai_vals)) if ai_vals else 0.0
    ai_std = float(np.std(ai_vals, ddof=1)) if len(ai_vals) > 1 else 0.0
    se = ai_std / np.sqrt(len(ai_vals)) if len(ai_vals) > 1 else 0.0
    t_crit = sp_stats.t.ppf(0.975, df=max(len(ai_vals) - 1, 1))
    ci_low = float(ai_mean - t_crit * se)
    ci_high = float(ai_mean + t_crit * se)

    acc_mean = float(np.mean(acc_vals)) if acc_vals else 0.0
    acc_std = float(np.std(acc_vals, ddof=1)) if len(acc_vals) > 1 else 0.0

    aggregate = {
        "n_runs": n,
        "ai_mean": ai_mean,
        "ai_ci_95_low": ci_low,
        "ai_ci_95_high": ci_high,
        "accident_mean": acc_mean,
        "accident_std": acc_std,
    }

    logger.info(
        "Batch data: %d non-baseline runs | AI mean=%.3f [%.3f, %.3f] | acc mean=%.1f",
        n,
        ai_mean,
        ci_low,
        ci_high,
        acc_mean,
    )

    return {
        "aggregate": aggregate,
        "runs": runs_list,
        "metrics": metrics,
        "accidents": accidents,
        "ai_events": ai_events,
        "label": label,
        "batch_dir": None,
    }


# ---------------------------------------------------------------------------
# Claude analysis
# ---------------------------------------------------------------------------


def _build_analysis_prompt(
    resilience: object,
    matrix: object,
    all_results: list[dict],
    weak_points: list,
    mfd_params: dict,
    config: dict,
) -> str:
    """Construct a structured prompt for the Claude analysis call."""
    rs = resilience  # ResilienceScore

    # Scenario summary
    n_total = len(all_results)
    n_success = sum(1 for r in all_results if r.get("status") == "success")
    n_baseline = sum(1 for r in all_results if r.get("scenario_type") == "baseline")
    n_incident = n_success - n_baseline

    # AI breakdown across successful non-baseline runs
    ai_vals = [
        r["summary"]["antifragility_index"]
        for r in all_results
        if r.get("status") == "success"
        and r.get("scenario_type") != "baseline"
        and r.get("summary", {}).get("antifragility_index") is not None
    ]

    def _cat(v: float) -> str:
        if v <= -0.20:
            return "BRITTLE"
        if v <= -0.05:
            return "FRAGILE"
        if v <= 0.05:
            return "RESILIENT"
        return "ANTIFRAGILE"

    from collections import Counter

    cat_counts = Counter(_cat(v) for v in ai_vals)
    cat_str = ", ".join(f"{k}: {v}" for k, v in sorted(cat_counts.items()))

    # MFD params
    mfd_lines = []
    if "free_flow_speed_kmh" in mfd_params:
        mfd_lines.append(f"- Free-flow speed: {mfd_params['free_flow_speed_kmh']:.1f} km/h")
        mfd_lines.append(f"- Jam density: {mfd_params['jam_density_veh_per_km']:.1f} veh/km")
        mfd_lines.append(
            f"- Network capacity: {mfd_params['capacity_veh_per_hour']:.0f} veh/h"
        )
        mfd_lines.append(f"- Greenshields R²: {mfd_params['r_squared']:.3f}")
    else:
        mfd_lines.append("- MFD fit not available (insufficient data)")

    # Weak points
    wp_lines = []
    if weak_points:
        for i, wp in enumerate(weak_points[:5], 1):
            wp_lines.append(
                f"  {i}. Edge {wp.edge_id}: {wp.accident_count} incidents, "
                f"vulnerability index {wp.vulnerability_index:.4f}, "
                f"mean speed-drop ratio {wp.mean_speed_drop_ratio:.2f}, "
                f"importance {wp.edge_importance:.3f}"
            )
    else:
        wp_lines.append("  No weak points identified.")

    # Demand levels tested
    dm = sorted(set(s.period for s in matrix.scenarios))
    dm_str = ", ".join(f"p={d}" for d in dm)

    prompt = f"""You are a senior traffic engineering expert and resilience scientist with deep expertise
in urban network analysis, SUMO traffic simulation, and the Antifragility Index (AI) framework.

I have just completed a One-Click Resilience Assessment of the Thessaloniki, Greece road network
using a SUMO-based accident simulator. Please provide a rigorous expert analysis of the results.

---

## Assessment Configuration

- Simulation framework: SUMO + TraCI (Thessaloniki network, ~13,000 road segments)
- Demand levels tested: {dm_str}
- Incident configs: {', '.join(sorted(set(s.scenario_type for s in matrix.scenarios)))}
- Seeds per scenario: {len(matrix.seeds)}
- Total scenarios: {n_total}  |  Successful: {n_success}  |  Baseline (no incidents): {n_baseline}
- Incident scenarios completed: {n_incident}

---

## Resilience Score

- **Overall Grade**: {rs.grade}
- **Overall Score**: {rs.overall_score:.3f} / 1.00
- **Interpretation**: {rs.interpretation}

### Component Scores (weights: speed 30%, throughput 25%, recovery 25%, robustness 20%)
- Speed resilience:      {rs.speed_resilience:.3f}   — speed maintenance under incidents vs baseline
- Throughput resilience: {rs.throughput_resilience:.3f}   — flow maintenance under incidents
- Recovery resilience:   {rs.recovery_resilience:.3f}   — post-incident recovery (based on AI)
- Robustness:            {rs.robustness:.3f}   — tolerance to increasing incident rates

---

## Macroscopic Fundamental Diagram (Greenshields Fit)

{chr(10).join(mfd_lines)}

The MFD was extracted from network-wide telemetry, filtered to steady-state (first 1200 s
warm-up excluded). Density = vehicles / network lane-km.

---

## Antifragility Index (AI) Summary

- AI formula: (V_post / V_pre) − 1  where V = space-mean speed measured around each incident
- AI aggregate: {rs.ai_aggregate:.4f}
- Category distribution across all incident scenarios: {cat_str}
- Interpretation: AI < −0.20 = BRITTLE, −0.20 to −0.05 = FRAGILE,
  −0.05 to +0.05 = RESILIENT, > +0.05 = ANTIFRAGILE

---

## Top Network Weak Points (by Vulnerability Index)

Vulnerability Index = (1 − speed_drop_ratio) × accident_frequency × edge_importance

{chr(10).join(wp_lines)}

---

## Requested Analysis

Please provide a structured expert analysis covering the following sections:

### 1. Methodology Assessment
Evaluate the soundness of the simulation-based resilience assessment approach. What are the
key strengths? What are the limitations of using SUMO simulation rather than real traffic data?
How reliable is the AI metric for measuring post-incident recovery?

### 2. Key Findings
What are the most important results from this assessment? Highlight unexpected or noteworthy
patterns. What does the score distribution (AI categories, component scores) tell us about
Thessaloniki's network behaviour?

### 3. Resilience Interpretation
Interpret the overall grade and component scores in the context of urban traffic networks.
Is {rs.grade} ({rs.overall_score:.2f}) a good result for a Mediterranean city of this size?
Which components are most concerning and why?

### 4. Weak Point Prioritisation
Based on the vulnerability indices and network topology, which edges should be addressed
first? Are there any patterns in the geographic distribution of weak points that suggest
systemic vulnerabilities (e.g., corridor dependencies, junction bottlenecks)?

### 5. Recommended Interventions
Provide specific, actionable recommendations for improving network resilience. Distinguish
between: (a) short-term operational measures (signal timing, incident response protocols),
(b) medium-term infrastructure improvements, and (c) long-term strategic changes.

### 6. Caveats and Limitations
What should urban planners and transport authorities be aware of when using these simulation
results for decision-making? What additional data or analyses would strengthen the conclusions?

Please be specific, technically rigorous, and practical. Avoid generic statements — ground
every recommendation in the specific numerical results above.
"""
    return prompt


def generate_claude_analysis(
    resilience: object,
    matrix: object,
    all_results: list[dict],
    weak_points: list,
    mfd_params: dict,
    config: dict,
    output_dir: str,
) -> str | None:
    """
    Call the Claude API for an expert analysis of the resilience assessment results.

    Requires ANTHROPIC_API_KEY environment variable. Gracefully skips and returns
    None if the package is not installed or the key is not set.

    Args:
        resilience:  ResilienceScore object.
        matrix:      ScenarioMatrix object.
        all_results: List of result dicts from parallel execution.
        weak_points: List of EdgeVulnerability objects.
        mfd_params:  Dict of Greenshields parameters.
        config:      Base YAML config dict.
        output_dir:  Output directory where the markdown file will be saved.

    Returns:
        The analysis text, or None if skipped.
    """
    try:
        import anthropic  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "anthropic package not installed — skipping Claude analysis. "
            "Install with: pip install anthropic"
        )
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning(
            "ANTHROPIC_API_KEY not set — skipping Claude analysis. "
            "Set the environment variable to enable AI-assisted reporting."
        )
        return None

    print("\n  Requesting Claude AI analysis (this may take ~30 seconds)...")

    prompt = _build_analysis_prompt(resilience, matrix, all_results, weak_points, mfd_params, config)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=_DEFAULT_CLAUDE_MODEL,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )
        analysis_text: str = message.content[0].text
    except Exception as exc:
        logger.error("Claude API call failed: %s", exc)
        return None

    # Save to markdown file
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    md_header = (
        f"# Resilience Assessment — AI-Assisted Analysis\n\n"
        f"*Generated by {_DEFAULT_CLAUDE_MODEL} on {ts}*\n\n"
        f"---\n\n"
    )
    analysis_path = os.path.join(output_dir, "assessment_analysis.md")
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(md_header + analysis_text)

    logger.info("Claude analysis saved → %s", analysis_path)
    print(f"  Claude analysis saved → {analysis_path}")

    return analysis_text


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_assessment(args: argparse.Namespace) -> None:
    """Main orchestration: generate -> execute -> analyze -> report."""
    from analyse_batch import (  # noqa: PLC0415
        figure_accident_characteristics,
        figure_network_dynamics,
        figure_per_event_ai,
        figure_resilience_statistics,
        figure_spatial_heatmap,
        load_network_shapes,
    )
    from mfd_analysis import (  # noqa: PLC0415
        compute_network_lane_km,
        compute_resilience_score,
        compute_weak_points,
        extract_mfd_data,
        fit_greenshields_model,
        fit_greenshields_per_scenario_type,
        plot_mfd_density_flow,
        plot_mfd_density_speed,
        plot_mfd_theoretical,
        score_to_dict,
    )
    from parallel_runner import ParallelExecutor  # noqa: PLC0415
    from resilience_report import generate_resilience_report  # noqa: PLC0415
    from runner import load_config, validate_config  # noqa: PLC0415
    from scenario_generator import (  # noqa: PLC0415
        assign_route_files,
        build_scenario_config,
        generate_scenario_matrix,
        matrix_to_dict,
        prepare_route_files,
    )

    # ── 1. Load and validate config ──
    config = load_config(args.config)
    validate_config(config)
    ra_cfg = config.get("resilience_assessment", {})

    # ── 2. Set up output directory ──
    if args.output_dir:
        output_dir = args.output_dir
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M")
        base_out = config.get("output", {}).get("output_folder", "results")
        output_dir = os.path.join(base_out, f"resilience_{ts}")

    os.makedirs(output_dir, exist_ok=True)
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(exist_ok=True)
    routes_dir = os.path.join(output_dir, "routes")

    # Set up logging.
    log_path = os.path.join(output_dir, "assessment.log")
    log_level = getattr(logging, args.log_level, logging.INFO)
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="w"),
    ]
    logging.basicConfig(level=log_level, format=fmt, handlers=handlers, force=True)

    logger.info("=" * 60)
    logger.info("  SAS One-Click Resilience Assessment")
    logger.info("=" * 60)
    logger.info("Output directory: %s", output_dir)

    # ── 3. Resolve parameters ──
    seeds_n = args.seeds
    if seeds_n is not None:
        base_seed = ra_cfg.get("base_seed", 42)
        seeds_list = list(range(base_seed, base_seed + seeds_n))
    else:
        seeds_list = None  # Let scenario_generator use defaults.

    # ── Phase 1/4: Generate scenario matrix ──
    print("\nPhase 1/4: Generating scenario matrix...")
    matrix = generate_scenario_matrix(
        config,
        output_dir,
        demand_levels=args.demand_levels,
        seeds=seeds_list,
        quick=args.quick,
    )
    logger.info(
        "Matrix: %d scenarios (%d demand levels × %d incident configs × %d seeds)",
        len(matrix.scenarios),
        len(matrix.demand_levels),
        len(matrix.incident_configs),
        len(matrix.seeds),
    )

    # ── Phase 1.5: Route files ──
    sumocfg_path = config["sumo"]["config_file"]
    net_xml_path = sumocfg_path.replace(".sumocfg", ".net.xml")

    if not args.skip_routes:
        print(f"  Generating route files for {len(matrix.demand_levels)} demand levels...")
        route_map = prepare_route_files(net_xml_path, routes_dir, matrix.demand_levels)
    else:
        # Assume routes already exist in routes_dir.
        route_map = {}
        for period in matrix.demand_levels:
            tag = f"{period:.2f}".replace(".", "p")
            cfg_path = os.path.join(routes_dir, f"thessaloniki_{tag}.sumocfg")
            rou_path = os.path.join(routes_dir, f"thessaloniki_{tag}.rou.xml")
            if os.path.exists(cfg_path):
                route_map[period] = (cfg_path, rou_path)
            else:
                logger.error(
                    "Route file missing for period=%s: %s (run without --skip-routes)",
                    period,
                    cfg_path,
                )
                sys.exit(1)

    assign_route_files(matrix, route_map)

    # Save scenario matrix.
    matrix_path = os.path.join(output_dir, "scenario_matrix.json")
    with open(matrix_path, "w") as f:
        json.dump(matrix_to_dict(matrix), f, indent=2)
    logger.info("Scenario matrix saved → %s", matrix_path)

    # ── Phase 2/4: Execute scenarios in parallel ──
    max_workers = args.workers or ra_cfg.get("max_workers") or None
    base_port = args.base_port or ra_cfg.get("base_port", 10000)

    print(f"\nPhase 2/4: Running {len(matrix.scenarios)} scenarios...")
    executor = ParallelExecutor(max_workers=max_workers, base_port=base_port)
    logger.info("Workers: %d  |  Base port: %d", executor.max_workers, executor.base_port)

    scenario_tuples = [
        (build_scenario_config(config, s), s.seed, s.output_folder) for s in matrix.scenarios
    ]
    all_results = executor.execute_scenarios(scenario_tuples, output_dir=output_dir)

    # Attach scenario metadata to results for easier downstream processing.
    for i, r in enumerate(all_results):
        if i < len(matrix.scenarios):
            r["scenario_id"] = matrix.scenarios[i].scenario_id
            r["period"] = matrix.scenarios[i].period
            r["scenario_type"] = matrix.scenarios[i].scenario_type
            r["base_probability"] = matrix.scenarios[i].base_probability

    n_success = sum(1 for r in all_results if r.get("status") == "success")
    n_fail = len(all_results) - n_success
    logger.info("Execution complete: %d success, %d failed", n_success, n_fail)

    if n_success == 0:
        logger.error("All scenarios failed — cannot proceed with analysis")
        print("\nERROR: All scenarios failed. Check assessment.log for details.")
        sys.exit(1)

    # ── Phase 3/4: Analysis ──
    print("\nPhase 3/4: Computing MFD and resilience metrics...")

    # Compute network lane-km (for MFD density).
    network_lane_km = compute_network_lane_km(net_xml_path)

    # Extract MFD data.
    mfd_data = extract_mfd_data(matrix.scenarios, network_lane_km)
    if not mfd_data.empty:
        mfd_csv_path = os.path.join(output_dir, "mfd_data.csv")
        mfd_data.to_csv(mfd_csv_path, index=False)
        logger.info("MFD data saved → %s  (%d rows)", mfd_csv_path, len(mfd_data))

    # Fit Greenshields model (overall, baseline-only).
    mfd_params = fit_greenshields_model(mfd_data)

    # Fit Greenshields per scenario type and save.
    if not mfd_data.empty:
        per_type_fits = fit_greenshields_per_scenario_type(mfd_data)
        if per_type_fits:
            per_type_path = os.path.join(output_dir, "mfd_per_type_fits.json")
            with open(per_type_path, "w") as f:
                json.dump(per_type_fits, f, indent=2)
            logger.info("Per-type MFD fits saved → %s", per_type_path)

    # Weak point analysis.
    top_n = ra_cfg.get("top_weak_points", 10)
    weak_points = compute_weak_points(matrix.scenarios, top_n=top_n)
    if weak_points:
        wp_path = os.path.join(output_dir, "weak_points.json")
        with open(wp_path, "w") as f:
            json.dump(
                [
                    {
                        "edge_id": wp.edge_id,
                        "x": wp.x,
                        "y": wp.y,
                        "accident_count": wp.accident_count,
                        "mean_duration_seconds": wp.mean_duration_seconds,
                        "mean_vehicles_affected": wp.mean_vehicles_affected,
                        "mean_speed_drop_ratio": wp.mean_speed_drop_ratio,
                        "edge_importance": wp.edge_importance,
                        "vulnerability_index": wp.vulnerability_index,
                    }
                    for wp in weak_points
                ],
                f,
                indent=2,
            )
        logger.info("Weak points saved → %s  (%d edges)", wp_path, len(weak_points))

    # Resilience score.
    weights_cfg = ra_cfg.get("resilience_weights")
    weights = {k: float(v) for k, v in weights_cfg.items()} if weights_cfg else None
    resilience = compute_resilience_score(
        mfd_data,
        all_results,
        matrix.scenarios,
        weak_points,
        mfd_params,
        weights,
    )

    # Save resilience score.
    score_path = os.path.join(output_dir, "resilience_score.json")
    with open(score_path, "w") as f:
        json.dump(score_to_dict(resilience), f, indent=2, default=str)
    logger.info("Resilience score saved → %s", score_path)

    # ── Generate figures (publication-quality, same style as analyse_batch.py) ──
    print("\n  Generating publication-quality figures...")

    # Load network shapes for spatial heatmap underlay.
    net_xml = Path(net_xml_path)
    if net_xml.exists():
        logger.info("Loading network shapes from %s", net_xml)
        net_segments, net_extent = load_network_shapes(net_xml)
        logger.info(
            "Network: %d road segments, extent (%d,%d)–(%d,%d)",
            len(net_segments),
            net_extent[0],
            net_extent[1],
            net_extent[2],
            net_extent[3],
        )
    else:
        logger.warning("Network file not found: %s — spatial heatmap will have no underlay", net_xml)
        net_segments, net_extent = [], (0, 1, 0, 1)

    # Build batch-compatible data dict from all non-baseline scenario results.
    ts_label = Path(output_dir).name  # e.g. "resilience_2026-03-06_14:30"
    batch_data = _build_assessment_batch_data(matrix.scenarios, all_results, label=ts_label)

    figures: dict[str, str] = {}

    # MFD figures (density-flow and density-speed, coloured by scenario type)
    if not mfd_data.empty:
        try:
            figures["mfd_density_flow"] = plot_mfd_density_flow(mfd_data, str(figures_dir))
            figures["mfd_density_speed"] = plot_mfd_density_speed(mfd_data, str(figures_dir))
        except Exception as exc:
            logger.warning("MFD figures failed: %s", exc)

        # Theoretical Greenshields curves per incident level.
        try:
            p = plot_mfd_theoretical(mfd_data, str(figures_dir))
            if p:
                figures["mfd_theoretical"] = p
        except Exception as exc:
            logger.warning("Theoretical MFD figure failed: %s", exc)

    # Figure 1: Resilience statistics (AI distribution, accident counts, scatter, categories)
    if len(batch_data["runs"]) >= 3:
        try:
            p = figure_resilience_statistics(batch_data, figures_dir)
            figures["resilience_statistics"] = str(p)
        except Exception as exc:
            logger.warning("figure_resilience_statistics failed: %s", exc)
    else:
        logger.warning("Too few runs (%d) for resilience_statistics figure", len(batch_data["runs"]))

    # Figure 2: Network dynamics time-series ensemble
    if not batch_data["metrics"].empty:
        try:
            p = figure_network_dynamics(batch_data, figures_dir)
            figures["network_dynamics"] = str(p)
        except Exception as exc:
            logger.warning("figure_network_dynamics failed: %s", exc)

    # Figure 3: Accident characteristics (severity, duration, trigger timing, impact)
    if not batch_data["accidents"].empty:
        try:
            p = figure_accident_characteristics(batch_data, figures_dir)
            figures["accident_characteristics"] = str(p)
        except Exception as exc:
            logger.warning("figure_accident_characteristics failed: %s", exc)

    # Figure 4: Spatial heatmap with network underlay
    if not batch_data["accidents"].empty:
        try:
            p = figure_spatial_heatmap(batch_data, figures_dir, net_segments, net_extent)
            figures["spatial_heatmap"] = str(p)
        except Exception as exc:
            logger.warning("figure_spatial_heatmap failed: %s", exc)

    # Figure 5: Per-event AI analysis
    if not batch_data["ai_events"].empty:
        try:
            p = figure_per_event_ai(batch_data, figures_dir)
            figures["per_event_ai"] = str(p)
        except Exception as exc:
            logger.warning("figure_per_event_ai failed: %s", exc)

    logger.info("Figures generated: %s", list(figures.keys()))

    # ── Claude AI-assisted analysis ──
    print("\nPhase 3b/4: Requesting Claude AI analysis...")
    claude_analysis = generate_claude_analysis(
        resilience,
        matrix,
        all_results,
        weak_points,
        mfd_params,
        config,
        output_dir,
    )

    # ── Phase 4/4: Generate report ──
    print("\nPhase 4/4: Generating resilience report...")
    report_path = generate_resilience_report(
        output_dir,
        resilience,
        matrix.scenarios,
        all_results,
        config,
        figures,
        claude_analysis=claude_analysis,
    )

    # ── Save aggregate summary ──
    agg_dir = os.path.join(output_dir, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)
    agg_path = os.path.join(agg_dir, "aggregate_summary.json")
    with open(agg_path, "w") as f:
        json.dump(
            {
                "assessment_timestamp": datetime.now(timezone.utc).isoformat(),
                "n_scenarios": len(matrix.scenarios),
                "n_success": n_success,
                "n_failed": n_fail,
                "resilience_grade": resilience.grade,
                "resilience_score": resilience.overall_score,
                "claude_analysis_generated": claude_analysis is not None,
                "results": [
                    {
                        "scenario_id": r.get("scenario_id", ""),
                        "status": r.get("status", ""),
                        "summary": r.get("summary", {}),
                    }
                    for r in all_results
                ],
            },
            f,
            indent=2,
            default=str,
        )

    # ── Final summary ──
    print(f"\n{'=' * 60}")
    print("  RESILIENCE ASSESSMENT COMPLETE")
    print(f"  Grade:  {resilience.grade}  (Score: {resilience.overall_score:.2f})")
    print(f"  Report: {report_path}")
    print(f"  Output: {output_dir}")
    if claude_analysis:
        print(f"  AI Analysis: {os.path.join(output_dir, 'assessment_analysis.md')}")
    print(f"{'=' * 60}")

    logger.info(
        "Assessment complete — Grade: %s (%.2f)", resilience.grade, resilience.overall_score
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run_assessment(args)


if __name__ == "__main__":
    main()
