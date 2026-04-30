from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

from ..app.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from ..integrations.download_osm_place import DEFAULT_NOMINATIM_URL, DEFAULT_OVERPASS_URL, DEFAULT_USER_AGENT
from ..simulation.runner import _generate_output_folder_name


@dataclass(slots=True)
class WorkflowField:
    name: str
    label: str
    type: str = "text"
    required: bool = False
    default: Any = None
    help: str = ""
    placeholder: str | None = None
    options: list[str] | None = None


@dataclass(slots=True)
class WorkflowSpec:
    id: str
    category: str
    title: str
    description: str
    module: str
    progress_mode: str = "indeterminate"
    fields: list[WorkflowField] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["command_preview"] = [sys.executable, "-m", self.module]
        return payload


def _value(payload: dict[str, Any], name: str, default: Any = None) -> Any:
    value = payload.get(name, default)
    if value == "":
        return default
    return value


def _append_arg(args: list[str], flag: str, value: Any) -> None:
    if value in (None, "", False):
        return
    if value is True:
        args.append(flag)
        return
    if isinstance(value, list):
        if not value:
            return
        args.append(flag)
        args.extend(str(item) for item in value)
        return
    args.extend([flag, str(value)])


def _predict_simulation_output(payload: dict[str, Any]) -> str | None:
    config_path = _value(payload, "config", DEFAULT_CONFIG_PATH.as_posix())
    config = load_config(config_path)
    base_output_path = config["output"]["output_folder"]
    return _generate_output_folder_name(
        base_output_path,
        config["sumo"]["config_file"],
        run_number=1,
        is_batch=int(_value(payload, "runs", 1)) > 1,
    )


def _predict_assessment_output(payload: dict[str, Any]) -> str | None:
    explicit = _value(payload, "output_dir")
    if explicit:
        return str((PROJECT_ROOT / explicit).resolve()) if not Path(explicit).is_absolute() else explicit

    config_path = _value(payload, "config", DEFAULT_CONFIG_PATH.as_posix())
    config = load_config(config_path)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    base_out = config.get("output", {}).get("output_folder", "results")
    return str((Path(base_out) / f"resilience_{ts}").resolve())


def _predict_direct_output(payload: dict[str, Any], field_name: str) -> str | None:
    out = _value(payload, field_name)
    if not out:
        return None
    path = Path(str(out)).expanduser()
    return str((PROJECT_ROOT / path).resolve()) if not path.is_absolute() else str(path)


def build_command(workflow_id: str, payload: dict[str, Any]) -> list[str]:
    command = [sys.executable, "-m", WORKFLOW_SPECS[workflow_id].module]

    if workflow_id == "simulation.run":
        _append_arg(command, "--config", _value(payload, "config", DEFAULT_CONFIG_PATH.as_posix()))
        _append_arg(command, "--runs", _value(payload, "runs", 1))
        _append_arg(command, "--log-level", _value(payload, "log_level", "INFO"))
        if _value(payload, "live_progress", False):
            command.append("--live-progress")
        return command

    if workflow_id == "assessment.run":
        _append_arg(command, "--config", _value(payload, "config", DEFAULT_CONFIG_PATH.as_posix()))
        _append_arg(command, "--workers", _value(payload, "workers"))
        _append_arg(command, "--demand-levels", _value(payload, "demand_levels", []))
        _append_arg(command, "--seeds", _value(payload, "seeds"))
        if _value(payload, "quick", False):
            command.append("--quick")
        _append_arg(command, "--output-dir", _value(payload, "output_dir"))
        if _value(payload, "skip_routes", False):
            command.append("--skip-routes")
        _append_arg(command, "--base-port", _value(payload, "base_port", 10000))
        _append_arg(command, "--log-level", _value(payload, "log_level", "INFO"))
        return command

    if workflow_id == "generator.thessaloniki":
        _append_arg(command, "--out-dir", _value(payload, "out_dir"))
        _append_arg(command, "--period", _value(payload, "period"))
        if _value(payload, "update_config", True):
            command.append("--update-config")
        _append_arg(command, "--config", _value(payload, "config", DEFAULT_CONFIG_PATH.as_posix()))
        if _value(payload, "gui", False):
            command.append("--gui")
        if _value(payload, "skip_download", False):
            command.append("--skip-download")
        return command

    if workflow_id == "generator.seattle":
        _append_arg(command, "--osm-file", _value(payload, "osm_file"))
        _append_arg(command, "--out-dir", _value(payload, "out_dir"))
        _append_arg(command, "--period", _value(payload, "period"))
        _append_arg(command, "--demand-source", _value(payload, "demand_source", "od"))
        _append_arg(command, "--od-file", _value(payload, "od_file"))
        _append_arg(command, "--node-file", _value(payload, "node_file"))
        _append_arg(command, "--od-scale", _value(payload, "od_scale"))
        _append_arg(command, "--edges-per-zone", _value(payload, "edges_per_zone"))
        _append_arg(command, "--seed", _value(payload, "seed"))
        _append_arg(command, "--end", _value(payload, "end"))
        if _value(payload, "update_config", True):
            command.append("--update-config")
        _append_arg(command, "--config", _value(payload, "config", "configs/seattle/default.yaml"))
        if _value(payload, "gui", False):
            command.append("--gui")
        if _value(payload, "skip_network", False):
            command.append("--skip-network")
        return command

    if workflow_id == "generator.city":
        _append_arg(command, "--city-slug", _value(payload, "city_slug"))
        _append_arg(command, "--osm-file", _value(payload, "osm_file"))
        _append_arg(command, "--out-dir", _value(payload, "out_dir"))
        _append_arg(command, "--period", _value(payload, "period"))
        _append_arg(command, "--demand-source", _value(payload, "demand_source", "random"))
        _append_arg(command, "--od-file", _value(payload, "od_file"))
        _append_arg(command, "--node-file", _value(payload, "node_file"))
        _append_arg(command, "--od-scale", _value(payload, "od_scale"))
        _append_arg(command, "--edges-per-zone", _value(payload, "edges_per_zone"))
        _append_arg(command, "--seed", _value(payload, "seed"))
        _append_arg(command, "--end", _value(payload, "end"))
        if _value(payload, "update_config", True):
            command.append("--update-config")
        _append_arg(command, "--config", _value(payload, "config"))
        if _value(payload, "gui", False):
            command.append("--gui")
        if _value(payload, "skip_network", False):
            command.append("--skip-network")
        return command

    if workflow_id == "generator.sioux_falls":
        _append_arg(command, "--out-dir", _value(payload, "out_dir"))
        _append_arg(command, "--period", _value(payload, "period"))
        if _value(payload, "update_config", False):
            command.append("--update-config")
        _append_arg(command, "--config", _value(payload, "config", DEFAULT_CONFIG_PATH.as_posix()))
        return command

    if workflow_id == "generator.riverside":
        _append_arg(command, "--out-dir", _value(payload, "out_dir"))
        if _value(payload, "update_config", False):
            command.append("--update-config")
        return command

    if workflow_id == "integration.fetch_osm":
        _append_arg(command, "--place", _value(payload, "place"))
        _append_arg(command, "--out", _value(payload, "out"))
        _append_arg(command, "--city-slug", _value(payload, "city_slug"))
        _append_arg(command, "--pad-km", _value(payload, "pad_km"))
        _append_arg(command, "--south", _value(payload, "south"))
        _append_arg(command, "--west", _value(payload, "west"))
        _append_arg(command, "--north", _value(payload, "north"))
        _append_arg(command, "--east", _value(payload, "east"))
        _append_arg(command, "--road-types", _value(payload, "road_types", []))
        if _value(payload, "all_features", False):
            command.append("--all-features")
        if _value(payload, "bootstrap_layout", True) is False:
            command.append("--no-bootstrap-layout")
        if _value(payload, "bootstrap_config", True) is False:
            command.append("--no-bootstrap-config")
        _append_arg(command, "--config-out", _value(payload, "config_out"))
        _append_arg(command, "--config-template", _value(payload, "config_template"))
        _append_arg(command, "--nominatim-url", _value(payload, "nominatim_url"))
        _append_arg(command, "--overpass-url", _value(payload, "overpass_url"))
        _append_arg(command, "--user-agent", _value(payload, "user_agent"))
        _append_arg(command, "--email", _value(payload, "email"))
        return command

    if workflow_id == "integration.govgr_download":
        _append_arg(command, "--dataset", _value(payload, "dataset", "all"))
        _append_arg(command, "--source", _value(payload, "source", "realtime"))
        _append_arg(command, "--limit", _value(payload, "limit"))
        _append_arg(command, "--offset-start", _value(payload, "offset_start"))
        _append_arg(command, "--max-pages", _value(payload, "max_pages"))
        _append_arg(command, "--timeout", _value(payload, "timeout"))
        _append_arg(command, "--retries", _value(payload, "retries"))
        _append_arg(command, "--backoff", _value(payload, "backoff"))
        _append_arg(command, "--historical-max-files", _value(payload, "historical_max_files"))
        _append_arg(command, "--historical-pattern", _value(payload, "historical_pattern"))
        if _value(payload, "no_extract_historical", False):
            command.append("--no-extract-historical")
        _append_arg(command, "--output-dir", _value(payload, "output_dir"))
        if _value(payload, "dry_run", False):
            command.append("--dry-run")
        if _value(payload, "skip_parquet", False):
            command.append("--skip-parquet")
        _append_arg(command, "--log-level", _value(payload, "log_level", "INFO"))
        return command

    if workflow_id == "integration.govgr_targets":
        _append_arg(command, "--downloads-root", _value(payload, "downloads_root"))
        _append_arg(command, "--calibration-year", _value(payload, "calibration_year"))
        _append_arg(command, "--validation-year", _value(payload, "validation_year"))
        _append_arg(command, "--output-dir", _value(payload, "output_dir"))
        _append_arg(command, "--chunksize", _value(payload, "chunksize"))
        return command

    if workflow_id == "analysis.batch":
        _append_arg(command, "--batch-dir", _value(payload, "batch_dir"))
        _append_arg(command, "--compare-dir", _value(payload, "compare_dir"))
        _append_arg(command, "--net-xml", _value(payload, "net_xml"))
        if _value(payload, "no_compare", False):
            command.append("--no-compare")
        return command

    if workflow_id == "analysis.sweep":
        _append_arg(command, "--config", _value(payload, "config", DEFAULT_CONFIG_PATH.as_posix()))
        _append_arg(command, "--runner", _value(payload, "runner"))
        _append_arg(command, "--out-dir", _value(payload, "out_dir"))
        _append_arg(command, "--periods", _value(payload, "periods", []))
        _append_arg(command, "--probs", _value(payload, "probs", []))
        _append_arg(command, "--seeds", _value(payload, "seeds"))
        return command

    if workflow_id == "analysis.visualise_sweep":
        _append_arg(command, "--csv", _value(payload, "csv"))
        _append_arg(command, "--out-dir", _value(payload, "out_dir"))
        return command

    if workflow_id == "analysis.merge_report":
        _append_arg(command, "--main", _value(payload, "main"))
        _append_arg(command, "--extra", _value(payload, "extra"))
        _append_arg(command, "--config", _value(payload, "config", DEFAULT_CONFIG_PATH.as_posix()))
        return command

    if workflow_id == "analysis.compare_seattle":
        _append_arg(command, "--sim-dir", _value(payload, "sim_dir"))
        _append_arg(command, "--real-csv", _value(payload, "real_csv"))
        _append_arg(command, "--out-dir", _value(payload, "out_dir"))
        _append_arg(command, "--start-hour", _value(payload, "start_hour"))
        _append_arg(command, "--window-hours", _value(payload, "window_hours"))
        _append_arg(command, "--year-from", _value(payload, "year_from"))
        _append_arg(command, "--year-to", _value(payload, "year_to"))
        _append_arg(command, "--bin-minutes", _value(payload, "bin_minutes"))
        _append_arg(command, "--grid-size", _value(payload, "grid_size"))
        return command

    raise KeyError(f"Unknown workflow: {workflow_id}")


def predict_output_dir(workflow_id: str, payload: dict[str, Any]) -> str | None:
    if workflow_id == "simulation.run":
        return _predict_simulation_output(payload)
    if workflow_id == "assessment.run":
        return _predict_assessment_output(payload)
    if workflow_id == "generator.city":
        explicit = _value(payload, "out_dir")
        if explicit:
            return _predict_direct_output(payload, "out_dir")
        city_slug = _value(payload, "city_slug")
        if not city_slug:
            return None
        return str((PROJECT_ROOT / "data" / "cities" / str(city_slug) / "network").resolve())
    if workflow_id in {
        "generator.thessaloniki",
        "generator.seattle",
        "generator.sioux_falls",
        "generator.riverside",
    }:
        return _predict_direct_output(payload, "out_dir")
    if workflow_id in {"integration.fetch_osm"}:
        out = _value(payload, "out")
        if not out:
            return None
        target = Path(str(out)).expanduser()
        return str(target.parent.resolve()) if target.is_absolute() else str((PROJECT_ROOT / target).resolve().parent)
    if workflow_id in {"integration.govgr_download", "integration.govgr_targets"}:
        return _predict_direct_output(payload, "output_dir")
    if workflow_id == "analysis.batch":
        return _predict_direct_output(payload, "batch_dir")
    if workflow_id in {"analysis.sweep", "analysis.visualise_sweep", "analysis.compare_seattle"}:
        return _predict_direct_output(payload, "out_dir")
    if workflow_id == "analysis.merge_report":
        return _predict_direct_output(payload, "main")
    return None


WORKFLOW_SPECS: dict[str, WorkflowSpec] = {
    "simulation.run": WorkflowSpec(
        id="simulation.run",
        category="Simulations",
        title="Run Simulation",
        description="Run a single SUMA simulation or a multi-run batch from a YAML config.",
        module="suma.simulation.runner",
        progress_mode="simulation",
        fields=[
            WorkflowField("config", "Config", type="config", default="configs/thessaloniki/default.yaml", required=True),
            WorkflowField("runs", "Runs", type="number", default=1),
            WorkflowField("log_level", "Log Level", type="choice", default="INFO", options=["DEBUG", "INFO", "WARNING", "ERROR"]),
            WorkflowField("live_progress", "Live Progress", type="boolean", default=True, help="Refresh live_progress.png during a single run."),
        ],
    ),
    "assessment.run": WorkflowSpec(
        id="assessment.run",
        category="Simulations",
        title="Run Resilience Assessment",
        description="Execute the one-click resilience assessment workflow and generate report outputs.",
        module="suma.analysis.resilience_assessment",
        progress_mode="phase4",
        fields=[
            WorkflowField("config", "Config", type="config", default="configs/thessaloniki/default.yaml", required=True),
            WorkflowField("workers", "Workers", type="number"),
            WorkflowField("demand_levels", "Demand Levels", type="number_list", help="Space- or comma-separated period values."),
            WorkflowField("seeds", "Seeds", type="number"),
            WorkflowField("quick", "Quick Mode", type="boolean", default=False),
            WorkflowField("output_dir", "Output Directory", type="text", placeholder="results/custom_assessment"),
            WorkflowField("skip_routes", "Skip Routes", type="boolean", default=False),
            WorkflowField("base_port", "Base Port", type="number", default=10000),
            WorkflowField("log_level", "Log Level", type="choice", default="INFO", options=["DEBUG", "INFO", "WARNING", "ERROR"]),
        ],
    ),
    "generator.city": WorkflowSpec(
        id="generator.city",
        category="Generators",
        title="Generate City Network",
        description="Build SUMO assets for any extracted city under data/cities/<slug>/ using either randomTrips demand or OD inputs when available.",
        module="suma.generators.generate_city",
        progress_mode="phase4",
        fields=[
            WorkflowField("city_slug", "City", type="city", required=True, help="Choose the extracted city folder to build. The generator reads from data/cities/<slug>/ and writes the SUMO artifacts back into that city workspace."),
            WorkflowField("osm_file", "OSM File Override", type="text", placeholder="Leave blank to use the city's default .osm extract", help="Optional explicit OSM input. In normal use, leave this blank so the generator picks the standard city extract automatically. Override it only when a city has more than one candidate OSM file."),
            WorkflowField("out_dir", "Output Directory", type="text", placeholder="Leave blank to use data/cities/<slug>/network", help="Output directory for the generated .net.xml, .rou.xml, and .sumocfg files. The standard choice is the city's network folder so Config Studio and the simulator discover the files automatically."),
            WorkflowField("period", "Random Route Period", type="number", default=1.5, help="Vehicle insertion period in seconds when using random demand. Lower values mean denser demand and usually heavier congestion. This field is ignored when demand source is OD."),
            WorkflowField("demand_source", "Demand Source", type="choice", default="random", options=["random", "od"], help="Choose whether demand is created synthetically with randomTrips or from OD support files. Use OD when a city has a compatible OD matrix and centroid node file; otherwise use random."),
            WorkflowField("od_file", "OD File Override", type="text", placeholder="Optional *_od.csv path", help="Optional explicit OD matrix. Leave blank to let the generator scan the city folder for a compatible OD CSV. Required only when demand source is OD and auto-discovery is not enough."),
            WorkflowField("node_file", "Node File Override", type="text", placeholder="Optional *_node.csv path", help="Optional explicit node/centroid CSV. Leave blank to let the generator scan the city folder for a compatible node CSV. Required only when demand source is OD and auto-discovery is not enough."),
            WorkflowField("od_scale", "OD Scale", type="number", default=0.02, help="Scale factor applied to OD counts before trip generation. Use this to thin or amplify a raw OD matrix for exploratory runs without editing the source file itself."),
            WorkflowField("edges_per_zone", "Edges Per Zone", type="number", default=3, help="Number of nearby passenger edges used per OD zone when mapping centroids onto the network. Higher values add route diversity but can make OD demand less tightly anchored."),
            WorkflowField("seed", "OD Seed", type="number", default=42, help="Random seed used for OD trip sampling and centroid-edge assignment choices."),
            WorkflowField("end", "End Time (s)", type="number", default=7200, help="Simulation end time written into the generated .sumocfg and used as the route-generation horizon."),
            WorkflowField("update_config", "Update Config", type="boolean", default=True, help="When enabled, patch the selected config so sumo.config_file points at the newly generated .sumocfg."),
            WorkflowField("config", "Config Override", type="text", placeholder="Leave blank to use configs/<slug>/default.yaml", help="Optional explicit YAML config to patch. In normal city workflows, leave this blank and let the generator use configs/<slug>/default.yaml."),
            WorkflowField("gui", "Use sumo-gui", type="boolean", default=False, help="When config patching is enabled, switch the configured SUMO binary from sumo to sumo-gui."),
            WorkflowField("skip_network", "Skip Network Build", type="boolean", default=False, help="Reuse an existing <slug>.net.xml in the output directory and rebuild only demand and .sumocfg. Useful when tuning demand repeatedly against the same compiled network."),
        ],
    ),
    "generator.sioux_falls": WorkflowSpec(
        id="generator.sioux_falls",
        category="Generators",
        title="Generate Sioux Falls Network",
        description="Build the Sioux Falls benchmark network.",
        module="suma.generators.generate_sioux_falls",
        progress_mode="phase4",
        fields=[
            WorkflowField("out_dir", "Output Directory", type="text", default="data/benchmarks/sioux_falls/network"),
            WorkflowField("period", "Route Period", type="number", default=1.5),
            WorkflowField("update_config", "Update Config", type="boolean", default=False),
            WorkflowField("config", "Config", type="config", default="configs/thessaloniki/default.yaml"),
        ],
    ),
    "generator.riverside": WorkflowSpec(
        id="generator.riverside",
        category="Generators",
        title="Generate Riverside Network",
        description="Build the synthetic Riverside network and optionally patch a config.",
        module="suma.generators.generate_network",
        progress_mode="phase4",
        fields=[
            WorkflowField("out_dir", "Output Directory", type="text", default="data/synthetic/riverside/network"),
            WorkflowField("update_config", "Update Config", type="boolean", default=False),
        ],
    ),
    "integration.fetch_osm": WorkflowSpec(
        id="integration.fetch_osm",
        category="Data & Integrations",
        title="Download OSM Extract",
        description="Use Nominatim and Overpass to fetch an OSM extract for a place.",
        module="suma.integrations.download_osm_place",
        progress_mode="indeterminate",
        fields=[
            WorkflowField("place", "Place", type="text", required=True, placeholder="Seattle, Washington, USA", help="Human-readable locality query sent to Nominatim."),
            WorkflowField("city_slug", "City Folder / Slug", type="text", placeholder="seattle", help="Used for data/cities/<slug>/ and configs/<slug>/ naming."),
            WorkflowField("out", "Output File", type="text", placeholder="data/cities/seattle/network/seattle.osm", help="Raw OSM XML destination. The recommended layout is data/cities/<slug>/network/<slug>.osm."),
            WorkflowField("pad_km", "Padding (km)", type="number", default=0, help="Expand the resolved boundary on every side. Use small positive values when the locality boundary is too tight for route generation."),
            WorkflowField("south", "South", type="number"),
            WorkflowField("west", "West", type="number"),
            WorkflowField("north", "North", type="number"),
            WorkflowField("east", "East", type="number"),
            WorkflowField("road_types", "Road Types", type="choice_list", default=["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential", "living_street", "service", "road"], options=["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential", "living_street", "service", "road", "track", "busway", "pedestrian", "cycleway", "footway", "path"], help="OSM highway classes to extract. The default set keeps the road classes that are generally useful for SUMO network building and excludes pedestrian-only infrastructure."),
            WorkflowField("all_features", "All Features", type="boolean", default=False, help="Advanced override. When enabled, the road-type filter is ignored and the full OSM node/way/relation set inside the boundary is downloaded."),
            WorkflowField("bootstrap_layout", "Bootstrap City Layout", type="boolean", default=True, help="Create data/cities/<slug>/network and city metadata before downloading."),
            WorkflowField("bootstrap_config", "Bootstrap Default Config", type="boolean", default=True, help="Create configs/<slug>/default.yaml from the template when missing."),
            WorkflowField("config_out", "Config Output", type="text", placeholder="configs/seattle/default.yaml", help="Default config path for the new city scaffold."),
            WorkflowField("config_template", "Config Template", type="text", default="configs/templates/city_default.yaml", help="Template copied into the new city's default config."),
            WorkflowField("nominatim_url", "Nominatim URL", type="text", default=DEFAULT_NOMINATIM_URL, help="Geocoding/search endpoint used to resolve a place name and bounding box. The public OSM endpoint is the recommended default."),
            WorkflowField("overpass_url", "Overpass URL", type="text", default=DEFAULT_OVERPASS_URL, help="Overpass interpreter endpoint used to fetch the raw OSM XML extract. The default public endpoint is suitable for normal use."),
            WorkflowField("user_agent", "User Agent", type="text", default=DEFAULT_USER_AGENT, help="HTTP User-Agent sent to public OSM services. Keep this populated for polite API usage."),
            WorkflowField("email", "Contact Email", type="text", help="Optional contact email for Nominatim requests; useful for heavier or repeated usage of the public endpoint."),
        ],
    ),
    "integration.govgr_download": WorkflowSpec(
        id="integration.govgr_download",
        category="Data & Integrations",
        title="Download gov.gr Data",
        description="Download realtime and/or historical Thessaloniki traffic datasets from the IMET/CERTH feeds used in the current Greek integration.",
        module="suma.integrations.govgr_downloader",
        progress_mode="indeterminate",
        fields=[
            WorkflowField("dataset", "Dataset", type="choice", default="all", options=["all", "speed", "congestion", "travel_times"]),
            WorkflowField("source", "Source", type="choice", default="realtime", options=["realtime", "historical", "both"]),
            WorkflowField("limit", "Page Size", type="number", default=500),
            WorkflowField("offset_start", "Offset Start", type="number", default=0),
            WorkflowField("max_pages", "Max Pages", type="number"),
            WorkflowField("timeout", "Timeout (s)", type="number", default=30),
            WorkflowField("retries", "Retries", type="number", default=3),
            WorkflowField("backoff", "Backoff (s)", type="number", default=1),
            WorkflowField("historical_max_files", "Historical Max Files", type="number"),
            WorkflowField("historical_pattern", "Historical Pattern", type="text"),
            WorkflowField("no_extract_historical", "Skip Historical Extract", type="boolean", default=False),
            WorkflowField("output_dir", "Output Directory", type="text", placeholder="data/cities/<city>/govgr/downloads", help="City-level downloads root or an explicit run folder. When the path ends with /downloads, the downloader creates a timestamped run folder under it."),
            WorkflowField("dry_run", "Dry Run", type="boolean", default=False),
            WorkflowField("skip_parquet", "Skip Parquet", type="boolean", default=False),
            WorkflowField("log_level", "Log Level", type="choice", default="INFO", options=["DEBUG", "INFO", "WARNING", "ERROR"]),
        ],
    ),
    "integration.govgr_targets": WorkflowSpec(
        id="integration.govgr_targets",
        category="Data & Integrations",
        title="Build gov.gr Targets",
        description="Build calibration and validation target tables from downloaded Thessaloniki gov.gr traffic feeds.",
        module="suma.integrations.govgr_targets",
        progress_mode="indeterminate",
        fields=[
            WorkflowField("downloads_root", "Downloads Root", type="text", placeholder="data/cities/<city>/govgr/downloads", help="Root folder containing downloader run subfolders for the selected target city."),
            WorkflowField("calibration_year", "Calibration Year", type="number", default=2025),
            WorkflowField("validation_year", "Validation Year", type="number", default=2026),
            WorkflowField("output_dir", "Output Directory", type="text", placeholder="data/cities/<city>/govgr/targets/calibration_2025_validation_2026", help="Target export folder. The GUI derives a default from the target city plus the calibration and validation years."),
            WorkflowField("chunksize", "Chunk Size", type="number", default=200000),
        ],
    ),
    "analysis.batch": WorkflowSpec(
        id="analysis.batch",
        category="Analysis",
        title="Analyse Batch Runs",
        description="Analyse a batch folder and optionally compare it to another batch.",
        module="suma.analysis.analyse_batch",
        progress_mode="indeterminate",
        fields=[
            WorkflowField("batch_dir", "Batch Directory", type="text", required=True),
            WorkflowField("compare_dir", "Compare Directory", type="text"),
            WorkflowField("net_xml", "Network .net.xml", type="text"),
            WorkflowField("no_compare", "Skip Compare", type="boolean", default=False),
        ],
    ),
    "analysis.sweep": WorkflowSpec(
        id="analysis.sweep",
        category="Analysis",
        title="Run Failure-Point Sweep",
        description="Run a parameter grid sweep across demand periods and base probabilities.",
        module="suma.tools.experiment_sweep",
        progress_mode="indeterminate",
        fields=[
            WorkflowField("config", "Config", type="config", default="configs/thessaloniki/default.yaml"),
            WorkflowField("runner", "Runner Override", type="text"),
            WorkflowField("out_dir", "Output Directory", type="text", default="results/sweep"),
            WorkflowField("periods", "Periods", type="number_list", default=[5.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5]),
            WorkflowField("probs", "Probabilities", type="number_list", default=[0.0, 0.00005, 0.00015, 0.0005, 0.001]),
            WorkflowField("seeds", "Seeds", type="number", default=2),
        ],
    ),
    "analysis.visualise_sweep": WorkflowSpec(
        id="analysis.visualise_sweep",
        category="Analysis",
        title="Visualise Sweep",
        description="Generate publication-style figures from sweep CSV output.",
        module="suma.tools.visualise_sweep",
        progress_mode="indeterminate",
        fields=[
            WorkflowField("csv", "Sweep CSV", type="text", required=True),
            WorkflowField("out_dir", "Output Directory", type="text", default="results/sweep/figures"),
        ],
    ),
    "analysis.merge_report": WorkflowSpec(
        id="analysis.merge_report",
        category="Analysis",
        title="Merge MFD And Update Report",
        description="Merge supplementary MFD data into an assessment and regenerate its report.",
        module="suma.tools.merge_mfd_and_update_report",
        progress_mode="phase4",
        fields=[
            WorkflowField("main", "Main Assessment Directory", type="text", required=True),
            WorkflowField("extra", "Supplementary Directory", type="text", required=True),
            WorkflowField("config", "Config", type="config", default="configs/thessaloniki/default.yaml"),
        ],
    ),
    "analysis.compare_seattle": WorkflowSpec(
        id="analysis.compare_seattle",
        category="Analysis",
        title="Compare Seattle With Real Data",
        description="Compare Seattle simulation accidents with the historical Seattle collisions dataset.",
        module="suma.tools.compare_seattle_real",
        progress_mode="indeterminate",
        fields=[
            WorkflowField("sim_dir", "Simulation Directory", type="text", required=True),
            WorkflowField("real_csv", "Real CSV", type="text", default="data/cities/seattle/bundle/crash_data/sdot_collisions_all_years.csv"),
            WorkflowField("out_dir", "Output Directory", type="text"),
            WorkflowField("start_hour", "Start Hour", type="number", default=8),
            WorkflowField("window_hours", "Window Hours", type="number"),
            WorkflowField("year_from", "Year From", type="number"),
            WorkflowField("year_to", "Year To", type="number"),
            WorkflowField("bin_minutes", "Bin Minutes", type="number", default=15),
            WorkflowField("grid_size", "Grid Size", type="number", default=24),
        ],
    ),
}
