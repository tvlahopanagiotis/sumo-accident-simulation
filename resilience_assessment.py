"""
resilience_assessment.py
========================

One-Click Resilience Assessment — CLI entry point and orchestrator.

Runs a comprehensive scenario matrix (demand levels × incident configs × seeds)
in parallel, computes the Macroscopic Fundamental Diagram, identifies network
weak points, produces a composite resilience score, and generates a self-
contained HTML report.

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

logger = logging.getLogger("sas.resilience")


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
# Orchestrator
# ---------------------------------------------------------------------------


def run_assessment(args: argparse.Namespace) -> None:
    """Main orchestration: generate -> execute -> analyze -> report."""
    from mfd_analysis import (
        compute_network_lane_km,
        compute_resilience_score,
        compute_weak_points,
        extract_mfd_data,
        fit_greenshields_model,
        plot_ai_distribution,
        plot_mfd_density_flow,
        plot_mfd_density_speed,
        plot_resilience_components,
        plot_speed_comparison,
        plot_throughput_comparison,
        plot_weak_point_map,
        score_to_dict,
    )
    from parallel_runner import ParallelExecutor
    from resilience_report import generate_resilience_report
    from runner import load_config, validate_config
    from scenario_generator import (
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
        ts = datetime.now().strftime("%Y-%m-%d_%H:%M")
        base_out = config.get("output", {}).get("output_folder", "results")
        output_dir = os.path.join(base_out, f"resilience_{ts}")

    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
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
    all_results = executor.execute_scenarios(scenario_tuples)

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

    # Compute network lane-km.
    network_lane_km = compute_network_lane_km(net_xml_path)

    # Extract MFD data.
    mfd_data = extract_mfd_data(matrix.scenarios, network_lane_km)
    if not mfd_data.empty:
        mfd_csv_path = os.path.join(output_dir, "mfd_data.csv")
        mfd_data.to_csv(mfd_csv_path, index=False)
        logger.info("MFD data saved → %s  (%d rows)", mfd_csv_path, len(mfd_data))

    # Fit Greenshields model.
    mfd_params = fit_greenshields_model(mfd_data)

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

    # ── Generate figures ──
    figures: dict[str, str] = {}
    if not mfd_data.empty:
        figures["mfd_density_flow"] = plot_mfd_density_flow(mfd_data, figures_dir)
        figures["mfd_density_speed"] = plot_mfd_density_speed(mfd_data, figures_dir)
        figures["speed_comparison"] = plot_speed_comparison(mfd_data, figures_dir)
        figures["throughput_comparison"] = plot_throughput_comparison(mfd_data, figures_dir)

    figures["resilience_components"] = plot_resilience_components(resilience, figures_dir)
    figures["ai_distribution"] = plot_ai_distribution(all_results, figures_dir)

    # Weak point map.
    all_accident_reports = []
    for s in matrix.scenarios:
        if s.scenario_type == "baseline":
            continue
        path = os.path.join(s.output_folder, "accident_reports.json")
        if os.path.exists(path):
            with open(path) as f:
                all_accident_reports.extend(json.load(f))
    figures["weak_point_map"] = plot_weak_point_map(weak_points, all_accident_reports, figures_dir)

    # ── Phase 4/4: Generate report ──
    print("\nPhase 4/4: Generating resilience report...")
    report_path = generate_resilience_report(
        output_dir,
        resilience,
        matrix.scenarios,
        all_results,
        config,
        figures,
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
