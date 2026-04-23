"""
scenario_generator.py
=====================

Generates the scenario matrix for one-click resilience assessment.

Produces all combinations of demand levels × incident configurations × seeds,
along with the route files and SUMO configs needed for each demand level.

Reuses generate_thessaloniki.generate_routes() and write_sumocfg() so that
route generation is consistent with the existing network workflow.
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default benchmark parameters
# ---------------------------------------------------------------------------

# Period (seconds) for randomTrips.py — lower = more vehicles.
# Approximate Thessaloniki simultaneous vehicle counts per period:
#   0.5  → ~1400-1600  (heavily congested)
#   0.75 → ~1200-1400  (congested)
#   1.0  → ~1100-1200  (near-capacity, current default)
#   1.5  → ~700-900    (moderate)
#   2.0  → ~500-600    (light)
#   3.0  → ~300-400    (free-flow)
#   5.0  → ~150-200    (very light)
DEFAULT_DEMAND_LEVELS: list[float] = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

# Quick mode: fewer scenarios for fast iteration.
QUICK_DEMAND_LEVELS: list[float] = [1.0, 2.0, 5.0]

# Incident configurations — each overrides risk.base_probability.
DEFAULT_INCIDENT_CONFIGS: list[dict] = [
    {"name": "baseline", "base_probability": 0.0},
    {"name": "low_incident", "base_probability": 5.0e-05},
    {"name": "default_incident", "base_probability": 1.5e-04},
    {"name": "high_incident", "base_probability": 5.0e-04},
    {"name": "extreme_incident", "base_probability": 1.0e-03},
]

QUICK_INCIDENT_CONFIGS: list[dict] = [
    {"name": "baseline", "base_probability": 0.0},
    {"name": "default_incident", "base_probability": 1.5e-04},
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """A single simulation scenario."""

    scenario_id: str
    scenario_type: str  # "baseline" | incident config name
    period: float
    seed: int
    base_probability: float
    output_folder: str
    sumocfg_path: str  # Path to the .sumocfg for this demand level


@dataclass
class ScenarioMatrix:
    """Complete set of scenarios for resilience assessment."""

    scenarios: list[Scenario] = field(default_factory=list)
    demand_levels: list[float] = field(default_factory=list)
    incident_configs: list[dict] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    network_config_file: str = ""


# ---------------------------------------------------------------------------
# Route file preparation
# ---------------------------------------------------------------------------


def prepare_route_files(
    net_path: str,
    routes_dir: str,
    demand_levels: list[float],
) -> dict[float, tuple[str, str]]:
    """
    Generate route files and .sumocfg files for each demand level.

    Calls generate_thessaloniki.generate_routes() and write_sumocfg() for each
    period value. Route files are stored in *routes_dir* for isolation.

    Args:
        net_path:      Absolute path to the .net.xml network file.
        routes_dir:    Directory to write route files and configs into.
        demand_levels: Period values to generate routes for.

    Returns:
        Mapping {period: (sumocfg_path, rou_path)}.
    """
    from ..generators.generate_thessaloniki import generate_routes, write_sumocfg

    os.makedirs(routes_dir, exist_ok=True)
    result: dict[float, tuple[str, str]] = {}

    for period in demand_levels:
        tag = f"{period:.2f}".replace(".", "p")
        rou_path = os.path.join(routes_dir, f"thessaloniki_{tag}.rou.xml")
        cfg_path = os.path.join(routes_dir, f"thessaloniki_{tag}.sumocfg")

        if os.path.exists(rou_path) and os.path.exists(cfg_path):
            logger.info("Reusing existing routes for period=%s: %s", period, rou_path)
        else:
            logger.info("Generating routes for period=%s → %s", period, rou_path)
            generate_routes(net_path, rou_path, period)
            # Write .sumocfg referencing absolute paths so SUMO can find them.
            write_sumocfg(cfg_path, os.path.abspath(net_path), os.path.abspath(rou_path))

        result[period] = (cfg_path, rou_path)

    return result


# ---------------------------------------------------------------------------
# Scenario matrix generation
# ---------------------------------------------------------------------------


def generate_scenario_matrix(
    config: dict,
    output_base: str,
    demand_levels: list[float] | None = None,
    incident_configs: list[dict] | None = None,
    seeds: list[int] | None = None,
    quick: bool = False,
) -> ScenarioMatrix:
    """
    Generate the full scenario matrix for resilience assessment.

    Args:
        config:          Base YAML config dict.
        output_base:     Root output directory for the assessment.
        demand_levels:   Period values (None → use defaults).
        incident_configs: Incident config dicts (None → use defaults).
        seeds:           Random seeds (None → [42, 43, 44]).
        quick:           If True, use reduced quick-mode defaults.

    Returns:
        ScenarioMatrix with all scenarios populated.
    """
    # Resolve config-level overrides first, then argument overrides.
    ra_cfg = config.get("resilience_assessment", {})

    if demand_levels is None:
        if quick:
            demand_levels = QUICK_DEMAND_LEVELS
        else:
            demand_levels = ra_cfg.get("demand_levels", DEFAULT_DEMAND_LEVELS)

    if incident_configs is None:
        if quick:
            incident_configs = QUICK_INCIDENT_CONFIGS
        else:
            cfg_incidents = ra_cfg.get("incident_configs")
            if (
                cfg_incidents
                and isinstance(cfg_incidents, list)
                and isinstance(cfg_incidents[0], dict)
            ):
                incident_configs = cfg_incidents
            else:
                incident_configs = DEFAULT_INCIDENT_CONFIGS

    if seeds is None:
        base_seed = ra_cfg.get("base_seed", 42)
        n_seeds = 2 if quick else ra_cfg.get("seeds_per_scenario", 3)
        seeds = list(range(base_seed, base_seed + n_seeds))

    runs_dir = os.path.join(output_base, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    matrix = ScenarioMatrix(
        demand_levels=demand_levels,
        incident_configs=incident_configs,
        seeds=seeds,
        network_config_file=config["sumo"]["config_file"],
    )

    for period in demand_levels:
        tag = f"{period:.2f}".replace(".", "p")
        for ic in incident_configs:
            for seed in seeds:
                scenario_id = f"{ic['name']}_p{tag}_s{seed}"
                output_folder = os.path.join(runs_dir, scenario_id)

                scenario = Scenario(
                    scenario_id=scenario_id,
                    scenario_type=ic["name"],
                    period=period,
                    seed=seed,
                    base_probability=ic["base_probability"],
                    output_folder=output_folder,
                    sumocfg_path="",  # Filled after route preparation.
                )
                matrix.scenarios.append(scenario)

    logger.info(
        "Scenario matrix: %d demand levels × %d incident configs × %d seeds = %d scenarios",
        len(demand_levels),
        len(incident_configs),
        len(seeds),
        len(matrix.scenarios),
    )
    return matrix


def assign_route_files(
    matrix: ScenarioMatrix,
    route_map: dict[float, tuple[str, str]],
) -> None:
    """
    Assign .sumocfg paths to each scenario based on its demand level.

    Modifies scenarios in-place.
    """
    for scenario in matrix.scenarios:
        cfg_path, _ = route_map[scenario.period]
        scenario.sumocfg_path = cfg_path


# ---------------------------------------------------------------------------
# Config overrides per scenario
# ---------------------------------------------------------------------------


def build_scenario_config(base_config: dict, scenario: Scenario) -> dict:
    """
    Create a per-scenario config by overriding relevant fields.

    Overrides:
        - sumo.config_file → scenario-specific .sumocfg
        - sumo.seed        → scenario seed
        - risk.base_probability → scenario base_probability
        - output.output_folder  → scenario output folder
    """
    cfg = copy.deepcopy(base_config)
    cfg["sumo"]["config_file"] = scenario.sumocfg_path
    cfg["sumo"]["seed"] = scenario.seed
    cfg["risk"]["base_probability"] = scenario.base_probability
    cfg["output"]["output_folder"] = scenario.output_folder
    return cfg


def matrix_to_dict(matrix: ScenarioMatrix) -> dict:
    """Serialize a ScenarioMatrix to a JSON-safe dict."""
    return {
        "demand_levels": matrix.demand_levels,
        "incident_configs": matrix.incident_configs,
        "seeds": matrix.seeds,
        "network_config_file": matrix.network_config_file,
        "total_scenarios": len(matrix.scenarios),
        "scenarios": [
            {
                "scenario_id": s.scenario_id,
                "scenario_type": s.scenario_type,
                "period": s.period,
                "seed": s.seed,
                "base_probability": s.base_probability,
                "output_folder": s.output_folder,
                "sumocfg_path": s.sumocfg_path,
            }
            for s in matrix.scenarios
        ],
    }
