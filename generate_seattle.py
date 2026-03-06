"""
generate_seattle.py
===================
Build a Seattle SUMO network and routes from the local Seattle OSM extract.

Default input:
  seattle_bundle/traffic_dataset/02_Seattle/01_Input_data/Seattle.osm

Outputs (default out-dir: ./seattle_network):
  seattle.osm
  seattle.net.xml
  seattle.rou.xml
  seattle.sumocfg
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OSM = os.path.join(
    ROOT_DIR,
    "seattle_bundle",
    "traffic_dataset",
    "02_Seattle",
    "01_Input_data",
    "Seattle.osm",
)
DEFAULT_OD = os.path.join(
    ROOT_DIR,
    "seattle_bundle",
    "traffic_dataset",
    "02_Seattle",
    "01_Input_data",
    "Seattle_od.csv",
)
DEFAULT_NODE = os.path.join(
    ROOT_DIR,
    "seattle_bundle",
    "traffic_dataset",
    "02_Seattle",
    "01_Input_data",
    "Seattle_node.csv",
)
DEFAULT_OUT_DIR = os.path.join(ROOT_DIR, "seattle_network")

SUMO_HOME = os.environ.get(
    "SUMO_HOME",
    "/opt/homebrew/Cellar/sumo/1.20.0/share/sumo",
)
TYPEMAP = os.path.join(SUMO_HOME, "data", "typemap", "osmNetconvert.typ.xml")
RANDOM_TRIPS = os.path.join(SUMO_HOME, "tools", "randomTrips.py")


def _run(cmd: list[str], desc: str = "") -> subprocess.CompletedProcess:
    label = f"  $ {' '.join(cmd)}"
    if len(label) > 120:
        label = label[:117] + "..."
    print(label)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n❌  {desc or cmd[0]} failed:")
        print(result.stderr[-3000:])
        sys.exit(1)
    return result


def _write(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"    wrote {path}")


def _pretty_xml(raw: str) -> str:
    from xml.dom import minidom

    return minidom.parseString(raw).toprettyxml(indent="    ")


def _import_sumolib():
    tools_dir = os.path.join(SUMO_HOME, "tools")
    if tools_dir not in sys.path:
        sys.path.append(tools_dir)
    try:
        import sumolib
    except Exception as exc:
        print("❌  Failed to import sumolib from SUMO tools.")
        print(f"    Expected tools dir: {tools_dir}")
        print(f"    Error: {exc}")
        sys.exit(1)
    return sumolib


def build_network(osm_path: str, net_path: str, typemap: str) -> None:
    if not os.path.exists(typemap):
        print(f"❌  SUMO typemap not found: {typemap}")
        print("    Set SUMO_HOME correctly.")
        sys.exit(1)

    _run(
        [
            "netconvert",
            "--osm-files",
            osm_path,
            "--type-files",
            typemap,
            "--output-file",
            net_path,
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
    size_mb = os.path.getsize(net_path) / 1_048_576
    print(f"    Network compiled → {net_path} ({size_mb:.1f} MB)")


def _count_trips(rou_path: str) -> int:
    tree = ET.parse(rou_path)
    root = tree.getroot()
    return len(root.findall("vehicle")) + len(root.findall("trip"))


def generate_routes_random(net_path: str, rou_path: str, period: float, end_seconds: int) -> None:
    if not os.path.exists(RANDOM_TRIPS):
        print(f"❌  randomTrips.py not found: {RANDOM_TRIPS}")
        print("    Set SUMO_HOME correctly.")
        sys.exit(1)

    _run(
        [
            sys.executable,
            RANDOM_TRIPS,
            "-n",
            net_path,
            "-o",
            rou_path,
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
        print(f"    Routes generated → {rou_path} ({n_veh:,} trips)")
    except Exception:
        print(f"    Routes generated → {rou_path}")


def _read_zone_centroids(node_file: str) -> dict[str, tuple[float, float]]:
    centroids: dict[str, tuple[float, float]] = {}
    with open(node_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = int(float(row["Node_ID"]))
            if node_id >= 10_000_000:
                lon = float(row["Lon"])
                lat = float(row["Lat"])
                centroids[str(node_id)] = (lon, lat)
    if not centroids:
        raise RuntimeError("No centroid nodes (Node_ID >= 10000000) found in node file.")
    return centroids


def _read_od_rows(od_file: str) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    with open(od_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            o = str(int(float(row["O_ID"])))
            d = str(int(float(row["D_ID"])))
            count = float(row["OD_Number"])
            rows.append((o, d, count))
    if not rows:
        raise RuntimeError("OD file appears empty.")
    return rows


def _zone_edge_candidates(
    net_path: str,
    centroids: dict[str, tuple[float, float]],
    edges_per_zone: int,
) -> dict[str, list[str]]:
    sumolib = _import_sumolib()
    net = sumolib.net.readNet(net_path)
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
            f"Could not map {len(failed)} zones to nearby edges (sample: {sample})."
        )
    return zone_edges


def generate_routes_od(
    net_path: str,
    rou_path: str,
    od_file: str,
    node_file: str,
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
    od_zone_ids = {o for o, _, _ in od_rows} | {d for _, d, _ in od_rows}

    missing = sorted(z for z in od_zone_ids if z not in centroids)
    if missing:
        sample = ", ".join(missing[:8])
        raise RuntimeError(
            f"Missing centroid coordinates for {len(missing)} OD zones (sample: {sample})."
        )

    zone_edges = _zone_edge_candidates(
        net_path=net_path,
        centroids={z: centroids[z] for z in od_zone_ids},
        edges_per_zone=edges_per_zone,
    )

    rng = random.Random(seed)
    duration = end_seconds - begin
    trips: list[tuple[float, str, str, str]] = []
    dropped_intrazonal_raw = 0.0
    generated = 0
    trip_counter = 0

    for o, d, count in od_rows:
        if o == d:
            dropped_intrazonal_raw += count
            continue

        scaled = count * od_scale
        n = int(math.floor(scaled))
        if rng.random() < (scaled - n):
            n += 1
        if n <= 0:
            continue

        o_edges = zone_edges[o]
        d_edges = zone_edges[d]

        for i in range(n):
            depart = begin + ((i + rng.random()) / n) * duration
            from_edge = o_edges[rng.randrange(len(o_edges))]
            to_edge = d_edges[rng.randrange(len(d_edges))]

            if from_edge == to_edge and len(d_edges) > 1:
                for _ in range(4):
                    candidate = d_edges[rng.randrange(len(d_edges))]
                    if candidate != from_edge:
                        to_edge = candidate
                        break
            if from_edge == to_edge and len(o_edges) > 1:
                for _ in range(4):
                    candidate = o_edges[rng.randrange(len(o_edges))]
                    if candidate != to_edge:
                        from_edge = candidate
                        break

            trip_id = f"od_{trip_counter:07d}_{o}_{d}"
            trip_counter += 1
            trips.append((depart, trip_id, from_edge, to_edge))
            generated += 1

    trips.sort(key=lambda t: t[0])

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

    print(f"    OD routes generated → {rou_path} ({generated:,} trips)")
    print(f"    OD zones mapped     : {len(zone_edges)}")
    print(f"    Intrazonal demand dropped (raw OD units): {dropped_intrazonal_raw:.0f}")


def write_sumocfg(cfg_path: str, net_file: str, rou_file: str, end_seconds: int) -> None:
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


def update_config_yaml(config_path: str, sumocfg_path: str, binary: str) -> None:
    import re

    with open(config_path, encoding="utf-8") as f:
        text = f.read()

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

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(text)

    print("\n  ✅  Config updated:")
    print(f"      sumo.config_file = {abs_cfg}")
    print(f"      sumo.binary      = {binary}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Seattle SUMO network + routes")
    ap.add_argument(
        "--osm-file",
        default=DEFAULT_OSM,
        help="Path to Seattle OSM file",
    )
    ap.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Output directory (default: ./seattle_network)",
    )
    ap.add_argument(
        "--period",
        type=float,
        default=2.0,
        help="randomTrips insertion period in seconds (lower = more demand)",
    )
    ap.add_argument(
        "--demand-source",
        choices=["od", "random"],
        default="od",
        help="Demand generation method: OD matrix ('od') or randomTrips ('random')",
    )
    ap.add_argument(
        "--od-file",
        default=DEFAULT_OD,
        help="Seattle OD matrix CSV (O_ID,D_ID,OD_Number)",
    )
    ap.add_argument(
        "--node-file",
        default=DEFAULT_NODE,
        help="Seattle node CSV with centroid coordinates",
    )
    ap.add_argument(
        "--od-scale",
        type=float,
        default=0.03,
        help="Scale factor applied to OD_Number before trip generation",
    )
    ap.add_argument(
        "--edges-per-zone",
        type=int,
        default=5,
        help="Number of nearest passenger edges used per OD zone",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for OD trip sampling",
    )
    ap.add_argument(
        "--end",
        type=int,
        default=7200,
        help="Simulation end time in seconds for generated routes/config",
    )
    ap.add_argument(
        "--update-config",
        action="store_true",
        help="Patch a YAML config file with generated sumocfg path",
    )
    ap.add_argument(
        "--config",
        default=os.path.join(ROOT_DIR, "config.seattle.yaml"),
        help="YAML config path to patch (used with --update-config)",
    )
    ap.add_argument(
        "--gui",
        action="store_true",
        help="Set binary to sumo-gui when patching config",
    )
    ap.add_argument(
        "--skip-network",
        action="store_true",
        help="Reuse existing seattle.net.xml in out-dir (skip OSM copy + netconvert)",
    )
    args = ap.parse_args()

    if not os.path.exists(args.osm_file):
        print(f"❌  OSM file not found: {args.osm_file}")
        print("    Make sure seattle_bundle has been downloaded.")
        sys.exit(1)
    if args.demand_source == "od":
        if not os.path.exists(args.od_file):
            print(f"❌  OD file not found: {args.od_file}")
            sys.exit(1)
        if not os.path.exists(args.node_file):
            print(f"❌  Node file not found: {args.node_file}")
            sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    osm_path = os.path.join(args.out_dir, "seattle.osm")
    net_path = os.path.join(args.out_dir, "seattle.net.xml")
    rou_path = os.path.join(args.out_dir, "seattle.rou.xml")
    cfg_path = os.path.join(args.out_dir, "seattle.sumocfg")

    print(f"\n{'=' * 60}")
    print("  Seattle Network Build")
    print(f"  Source OSM: {os.path.abspath(args.osm_file)}")
    print(f"  Output    : {os.path.abspath(args.out_dir)}")
    print(f"  demand    : {args.demand_source}")
    if args.demand_source == "random":
        print(f"  period    : {args.period}s")
    else:
        print(f"  od_scale  : {args.od_scale}")
        print(f"  od_file   : {os.path.abspath(args.od_file)}")
        print(f"  node_file : {os.path.abspath(args.node_file)}")
    print(f"{'=' * 60}\n")

    if args.skip_network:
        if not os.path.exists(net_path):
            print(f"❌  --skip-network requested but net file does not exist: {net_path}")
            sys.exit(1)
        print("[ 1/4 ]  Skipping OSM copy (reuse existing network artifacts) …")
    else:
        print("[ 1/4 ]  Copying OSM into output folder …")
        shutil.copy2(args.osm_file, osm_path)
        print(f"    OSM copied → {osm_path}")

    if args.skip_network:
        print("\n[ 2/4 ]  Skipping netconvert (reusing existing seattle.net.xml) …")
    else:
        print("\n[ 2/4 ]  Building SUMO network …")
        build_network(osm_path, net_path, TYPEMAP)

    if args.demand_source == "random":
        print(f"\n[ 3/4 ]  Generating randomTrips routes (period={args.period}s) …")
        generate_routes_random(net_path, rou_path, args.period, args.end)
    else:
        print("\n[ 3/4 ]  Generating OD-driven routes …")
        generate_routes_od(
            net_path=net_path,
            rou_path=rou_path,
            od_file=args.od_file,
            node_file=args.node_file,
            begin=0,
            end_seconds=args.end,
            od_scale=args.od_scale,
            edges_per_zone=args.edges_per_zone,
            seed=args.seed,
        )

    print("\n[ 4/4 ]  Writing run configuration …")
    write_sumocfg(
        cfg_path,
        net_file=os.path.basename(net_path),
        rou_file=os.path.basename(rou_path),
        end_seconds=args.end,
    )

    if args.update_config and os.path.exists(args.config):
        binary = "sumo-gui" if args.gui else "sumo"
        update_config_yaml(args.config, cfg_path, binary)

    print(f"\n{'=' * 60}")
    print("  Seattle artifacts ready")
    print(f"  SUMO cfg: {cfg_path}")
    print("  Run with:")
    print(f"    sumo -c {cfg_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
