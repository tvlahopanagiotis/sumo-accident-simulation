"""
generate_city.py
================

Build a runnable SUMO network for any city folder under ``data/cities/<slug>/``.

This workflow assumes the OSM extract step has already happened and then:
  1. locates the city OSM extract
  2. builds the SUMO network with netconvert
  3. generates demand using either randomTrips or OD inputs
  4. writes a city-scoped ``.sumocfg``
  5. optionally patches the city's YAML config
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from ..app.config import CONFIGS_DIR, PROJECT_ROOT
from ..simulation.sumo_paths import find_random_trips_path, find_typemap_path, resolve_sumo_home

SUMO_HOME = resolve_sumo_home()
TYPEMAP = find_typemap_path(SUMO_HOME)
RANDOM_TRIPS = find_random_trips_path(SUMO_HOME)


def _run(cmd: list[str], desc: str = "") -> subprocess.CompletedProcess:
    label = f"  $ {' '.join(cmd)}"
    if len(label) > 140:
        label = label[:137] + "..."
    print(label)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\nERROR: {desc or cmd[0]} failed")
        print(result.stderr[-4000:])
        sys.exit(1)
    return result


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    print(f"    wrote {path}")


def _pretty_xml(raw: str) -> str:
    from xml.dom import minidom

    return minidom.parseString(raw).toprettyxml(indent="    ")


def _city_root(city_slug: str) -> Path:
    return PROJECT_ROOT / "data" / "cities" / city_slug


def _default_network_dir(city_slug: str) -> Path:
    return _city_root(city_slug) / "network"


def _default_config_path(city_slug: str) -> Path:
    return CONFIGS_DIR / city_slug / "default.yaml"


def _read_city_metadata(city_slug: str) -> dict[str, object]:
    path = _city_root(city_slug) / "city_metadata.json"
    if not path.exists():
        return {}
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _path_from_metadata(value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _resolve_osm_source(city_slug: str, explicit_path: str | None) -> Path:
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()

    metadata = _read_city_metadata(city_slug)
    candidates: list[Path] = []
    default_path = _default_network_dir(city_slug) / f"{city_slug}.osm"
    candidates.append(default_path)

    metadata_path = _path_from_metadata(metadata.get("osm_extract"))
    if metadata_path is not None:
        candidates.append(metadata_path)

    network_dir = _default_network_dir(city_slug)
    if network_dir.exists():
        candidates.extend(sorted(network_dir.glob("*.osm")))
    city_root = _city_root(city_slug)
    if city_root.exists():
        candidates.extend(sorted(city_root.rglob("*.osm")))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        f"No OSM extract found for city '{city_slug}'. Expected something like "
        f"{default_path.relative_to(PROJECT_ROOT)} after running the OSM extract step."
    )


def _matches_support_file(path: Path, token: str) -> bool:
    stem = path.stem.lower()
    return bool(re.search(rf"(?:^|[_-]){re.escape(token)}(?:[_-]|$)", stem))


def _resolve_support_csv(city_slug: str, explicit_path: str | None, token: str) -> Path | None:
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()

    city_root = _city_root(city_slug)
    if not city_root.exists():
        return None

    for candidate in sorted(city_root.rglob("*.csv")):
        if _matches_support_file(candidate, token):
            return candidate.resolve()
    return None


def _import_sumolib():
    tools_dir = os.path.join(SUMO_HOME, "tools")
    if tools_dir not in sys.path:
        sys.path.append(tools_dir)
    try:
        import sumolib
    except Exception as exc:
        print("ERROR: failed to import sumolib from SUMO tools.")
        print(f"       expected tools dir: {tools_dir}")
        print(f"       resolved SUMO_HOME: {SUMO_HOME or '(empty)'}")
        print(f"       error: {exc}")
        sys.exit(1)
    return sumolib


def build_network(osm_path: Path, net_path: Path, typemap: str) -> None:
    if not os.path.exists(typemap):
        print(f"ERROR: SUMO typemap not found: {typemap}")
        print(f"       resolved SUMO_HOME: {SUMO_HOME or '(empty)'}")
        sys.exit(1)

    _run(
        [
            "netconvert",
            "--osm-files",
            str(osm_path),
            "--type-files",
            typemap,
            "--output-file",
            str(net_path),
            "--geometry.remove",
            "--roundabouts.guess",
            "--ramps.guess",
            "--junctions.join",
            "--tls.guess-signals",
            "true",
            "--tls.discard-simple",
            "false",
            "--tls.join",
            "true",
            "--keep-edges.by-vclass",
            "passenger,truck",
            "--no-internal-links",
            "false",
            "--no-warnings",
            "true",
        ],
        desc="netconvert",
    )
    size_mb = net_path.stat().st_size / 1_048_576
    print(f"    Network compiled -> {net_path} ({size_mb:.1f} MB)")


def _count_trips(rou_path: Path) -> int:
    tree = ET.parse(rou_path)
    root = tree.getroot()
    return len(root.findall("vehicle")) + len(root.findall("trip"))


def generate_routes_random(net_path: Path, rou_path: Path, period: float, end_seconds: int) -> None:
    if not os.path.exists(RANDOM_TRIPS):
        print(f"ERROR: randomTrips.py not found: {RANDOM_TRIPS}")
        print(f"       resolved SUMO_HOME: {SUMO_HOME or '(empty)'}")
        sys.exit(1)

    _run(
        [
            sys.executable,
            RANDOM_TRIPS,
            "-n",
            str(net_path),
            "-o",
            str(rou_path),
            "--period",
            str(period),
            "--begin",
            "0",
            "--end",
            str(end_seconds),
            "--validate",
            "--remove-loops",
            "--fringe-factor",
            "2.0",
        ],
        desc="randomTrips.py",
    )

    try:
        n_veh = _count_trips(rou_path)
        print(f"    Routes generated -> {rou_path} ({n_veh:,} trips)")
    except Exception:
        print(f"    Routes generated -> {rou_path}")


def _read_zone_centroids(node_file: Path) -> dict[str, tuple[float, float]]:
    centroids: dict[str, tuple[float, float]] = {}
    with node_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            node_id = int(float(row["Node_ID"]))
            if node_id >= 10_000_000:
                lon = float(row["Lon"])
                lat = float(row["Lat"])
                centroids[str(node_id)] = (lon, lat)
    if not centroids:
        raise RuntimeError("No centroid nodes (Node_ID >= 10000000) found in node file.")
    return centroids


def _read_od_rows(od_file: Path) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    with od_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            origin = str(int(float(row["O_ID"])))
            destination = str(int(float(row["D_ID"])))
            count = float(row["OD_Number"])
            rows.append((origin, destination, count))
    if not rows:
        raise RuntimeError("OD file appears empty.")
    return rows


def _zone_edge_candidates(
    net_path: Path,
    centroids: dict[str, tuple[float, float]],
    edges_per_zone: int,
) -> dict[str, list[str]]:
    sumolib = _import_sumolib()
    net = sumolib.net.readNet(str(net_path))
    radius_steps = [100, 250, 500, 1000, 2000]
    zone_edges: dict[str, list[str]] = {}
    failed: list[str] = []

    for zone_id, (lon, lat) in centroids.items():
        x, y = net.convertLonLat2XY(lon, lat)
        candidates: list[tuple[str, float]] = []
        seen: set[str] = set()

        for radius in radius_steps:
            nearby = net.getNeighboringEdges(x, y, r=radius, includeJunctions=False, allowFallback=True)
            if not nearby:
                continue
            for edge, dist in sorted(nearby, key=lambda item: item[1]):
                edge_id = edge.getID()
                if edge_id.startswith(":") or edge_id in seen:
                    continue
                try:
                    if not edge.allows("passenger"):
                        continue
                except Exception:
                    pass
                seen.add(edge_id)
                candidates.append((edge_id, dist))
                if len(candidates) >= edges_per_zone:
                    break
            if len(candidates) >= edges_per_zone:
                break

        if not candidates:
            failed.append(zone_id)
            continue
        zone_edges[zone_id] = [edge_id for edge_id, _ in candidates]

    if failed:
        sample = ", ".join(failed[:8])
        raise RuntimeError(
            f"Could not map {len(failed)} OD zones to nearby edges (sample: {sample})."
        )
    return zone_edges


def generate_routes_od(
    net_path: Path,
    rou_path: Path,
    od_file: Path,
    node_file: Path,
    begin: int,
    end_seconds: int,
    od_scale: float,
    edges_per_zone: int,
    seed: int,
) -> None:
    if od_scale <= 0:
        raise ValueError("od_scale must be > 0.")
    if edges_per_zone < 1:
        raise ValueError("edges_per_zone must be >= 1.")
    if end_seconds <= begin:
        raise ValueError("end_seconds must be greater than begin.")

    od_rows = _read_od_rows(od_file)
    centroids = _read_zone_centroids(node_file)
    od_zone_ids = {origin for origin, _, _ in od_rows} | {destination for _, destination, _ in od_rows}
    missing = sorted(zone_id for zone_id in od_zone_ids if zone_id not in centroids)
    if missing:
        sample = ", ".join(missing[:8])
        raise RuntimeError(
            f"Missing centroid coordinates for {len(missing)} OD zones (sample: {sample})."
        )

    zone_edges = _zone_edge_candidates(
        net_path=net_path,
        centroids={zone_id: centroids[zone_id] for zone_id in od_zone_ids},
        edges_per_zone=edges_per_zone,
    )

    rng = random.Random(seed)
    duration = end_seconds - begin
    trips: list[tuple[float, str, str, str]] = []
    dropped_intrazonal_raw = 0.0
    generated = 0
    trip_counter = 0

    for origin, destination, count in od_rows:
        if origin == destination:
            dropped_intrazonal_raw += count
            continue

        scaled = count * od_scale
        n_trips = int(math.floor(scaled))
        if rng.random() < (scaled - n_trips):
            n_trips += 1
        if n_trips <= 0:
            continue

        origin_edges = zone_edges[origin]
        destination_edges = zone_edges[destination]

        for index in range(n_trips):
            depart = begin + ((index + rng.random()) / n_trips) * duration
            from_edge = origin_edges[rng.randrange(len(origin_edges))]
            to_edge = destination_edges[rng.randrange(len(destination_edges))]

            if from_edge == to_edge and len(destination_edges) > 1:
                for _ in range(4):
                    candidate = destination_edges[rng.randrange(len(destination_edges))]
                    if candidate != from_edge:
                        to_edge = candidate
                        break
            if from_edge == to_edge and len(origin_edges) > 1:
                for _ in range(4):
                    candidate = origin_edges[rng.randrange(len(origin_edges))]
                    if candidate != to_edge:
                        from_edge = candidate
                        break

            trip_id = f"od_{trip_counter:07d}_{origin}_{destination}"
            trip_counter += 1
            trips.append((depart, trip_id, from_edge, to_edge))
            generated += 1

    trips.sort(key=lambda item: item[0])

    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="passenger", vClass="passenger")
    for depart, trip_id, from_edge, to_edge in trips:
        ET.SubElement(
            root,
            "trip",
            attrib={
                "id": trip_id,
                "type": "passenger",
                "depart": f"{depart:.2f}",
                "from": from_edge,
                "to": to_edge,
            },
        )

    pretty = _pretty_xml(ET.tostring(root, encoding="unicode"))
    _write(rou_path, pretty)

    print(f"    OD routes generated -> {rou_path} ({generated:,} trips)")
    print(f"    OD zones mapped     : {len(zone_edges)}")
    print(f"    Intrazonal demand dropped (raw OD units): {dropped_intrazonal_raw:.0f}")


def write_sumocfg(cfg_path: Path, net_file: str, rou_file: str, end_seconds: int) -> None:
    root = ET.Element("configuration")

    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file", value=net_file)
    ET.SubElement(inp, "route-files", value=rou_file)

    tim = ET.SubElement(root, "time")
    ET.SubElement(tim, "begin", value="0")
    ET.SubElement(tim, "end", value=str(end_seconds))

    proc = ET.SubElement(root, "processing")
    ET.SubElement(proc, "ignore-route-errors", value="true")

    pretty = _pretty_xml(ET.tostring(root, encoding="unicode"))
    _write(cfg_path, pretty)


def update_config_yaml(config_path: Path, sumocfg_path: Path, binary: str) -> None:
    text = config_path.read_text(encoding="utf-8")
    abs_cfg = os.path.abspath(sumocfg_path)
    text = re.sub(
        r"^(\s*config_file\s*:\s*).*$",
        rf"\g<1>{abs_cfg}",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^(\s*binary\s*:\s*).*$",
        rf"\g<1>{binary}",
        text,
        flags=re.MULTILINE,
    )
    config_path.write_text(text, encoding="utf-8")
    print("\n  Config updated:")
    print(f"    sumo.config_file = {abs_cfg}")
    print(f"    sumo.binary      = {binary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SUMO assets for a city under data/cities/<slug>/")
    parser.add_argument("--city-slug", required=True, help="City folder slug under data/cities/")
    parser.add_argument("--osm-file", default=None, help="Optional explicit source OSM file. By default the city folder is scanned.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Default: data/cities/<slug>/network")
    parser.add_argument("--period", type=float, default=1.5, help="randomTrips insertion period in seconds when using random demand.")
    parser.add_argument("--demand-source", choices=["random", "od"], default="random", help="Demand generation method.")
    parser.add_argument("--od-file", default=None, help="Optional OD CSV with O_ID,D_ID,OD_Number.")
    parser.add_argument("--node-file", default=None, help="Optional node CSV with centroid coordinates.")
    parser.add_argument("--od-scale", type=float, default=0.02, help="Scale factor applied to OD_Number before trip generation.")
    parser.add_argument("--edges-per-zone", type=int, default=3, help="Number of nearest passenger edges used per OD zone.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for OD sampling.")
    parser.add_argument("--end", type=int, default=7200, help="Simulation end time in seconds for generated routes/config.")
    parser.add_argument("--update-config", action="store_true", help="Patch the city YAML config with the generated sumocfg.")
    parser.add_argument("--config", default=None, help="Optional YAML config path to patch. Default: configs/<slug>/default.yaml")
    parser.add_argument("--gui", action="store_true", help="Set binary to sumo-gui when patching config.")
    parser.add_argument("--skip-network", action="store_true", help="Reuse an existing <slug>.net.xml in the output directory.")
    args = parser.parse_args()

    city_slug = args.city_slug.strip()
    if not city_slug:
        print("ERROR: --city-slug is required.")
        sys.exit(1)

    city_root = _city_root(city_slug)
    if not city_root.exists():
        print(f"ERROR: city folder not found: {city_root}")
        print("       Run the OSM extract step first.")
        sys.exit(1)

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else _default_network_dir(city_slug)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_osm = _resolve_osm_source(city_slug, args.osm_file)
    target_osm = out_dir / f"{city_slug}.osm"
    net_path = out_dir / f"{city_slug}.net.xml"
    rou_path = out_dir / f"{city_slug}.rou.xml"
    cfg_path = out_dir / f"{city_slug}.sumocfg"
    config_path = Path(args.config).expanduser() if args.config else _default_config_path(city_slug)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()

    od_file = _resolve_support_csv(city_slug, args.od_file, "od")
    node_file = _resolve_support_csv(city_slug, args.node_file, "node")

    if args.demand_source == "od":
        if od_file is None:
            print(f"ERROR: no OD file found for city '{city_slug}'.")
            print("       Provide --od-file explicitly or add an *_od.csv file under the city folder.")
            sys.exit(1)
        if node_file is None:
            print(f"ERROR: no node file found for city '{city_slug}'.")
            print("       Provide --node-file explicitly or add a *_node.csv file under the city folder.")
            sys.exit(1)

    print(f"\n{'=' * 64}")
    print(f"  City Generator: {city_slug}")
    print(f"  Source OSM : {source_osm}")
    print(f"  Output dir : {out_dir}")
    print(f"  Demand     : {args.demand_source}")
    if args.demand_source == "random":
        print(f"  Period     : {args.period}s")
    else:
        print(f"  OD file    : {od_file}")
        print(f"  Node file  : {node_file}")
        print(f"  OD scale   : {args.od_scale}")
    print(f"{'=' * 64}\n")

    if args.skip_network:
        if not net_path.exists():
            print(f"ERROR: --skip-network requested but net file does not exist: {net_path}")
            sys.exit(1)
        print("[ 1/4 ]  Reusing existing network artifacts ...")
    else:
        print("[ 1/4 ]  Preparing city OSM input ...")
        if source_osm.resolve() != target_osm.resolve():
            shutil.copy2(source_osm, target_osm)
            print(f"    OSM copied -> {target_osm}")
        else:
            print(f"    OSM already in place -> {target_osm}")

    if args.skip_network:
        print("\n[ 2/4 ]  Skipping netconvert ...")
    else:
        print("\n[ 2/4 ]  Building SUMO network ...")
        build_network(target_osm, net_path, TYPEMAP)

    if args.demand_source == "random":
        print(f"\n[ 3/4 ]  Generating randomTrips routes (period={args.period}s) ...")
        generate_routes_random(net_path, rou_path, args.period, args.end)
    else:
        print("\n[ 3/4 ]  Generating OD-driven routes ...")
        generate_routes_od(
            net_path=net_path,
            rou_path=rou_path,
            od_file=od_file,
            node_file=node_file,
            begin=0,
            end_seconds=args.end,
            od_scale=args.od_scale,
            edges_per_zone=args.edges_per_zone,
            seed=args.seed,
        )

    print("\n[ 4/4 ]  Writing run configuration ...")
    write_sumocfg(cfg_path, net_path.name, rou_path.name, args.end)

    if args.update_config:
        if config_path.exists():
            binary = "sumo-gui" if args.gui else "sumo"
            update_config_yaml(config_path, cfg_path, binary)
        else:
            print(f"\n  NOTE: config file not found, skipped patching -> {config_path}")

    print(f"\n{'=' * 64}")
    print(f"  City assets ready for {city_slug}")
    print(f"  SUMO cfg: {cfg_path}")
    print("  Run with:")
    print(f"    sumo -c {cfg_path}")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
