"""
generate_thessaloniki.py
========================
Download and build the Thessaloniki city-centre SUMO network from OpenStreetMap.

This script:
  1. Downloads the OSM road network from the Overpass API
  2. Converts it to a SUMO network with netconvert (using SUMO's OSM typemaps)
  3. Generates randomised vehicle routes with randomTrips.py
  4. Writes a .sumocfg run configuration
  5. Optionally patches config.yaml so runner.py uses this network immediately

Bounding box — Thessaloniki city centre (~5.5 km × 5 km):
  The area covers the waterfront, Aristotelous Square, the upper city
  (Ano Poli) to the north, and the eastern suburbs as far as Kalamaria.
  It is large enough to exhibit realistic congestion dynamics while
  remaining feasible for a 2-hour simulation on a laptop.

Usage:
  python generate_thessaloniki.py                  # build network + routes
  python generate_thessaloniki.py --update-config  # also patch config.yaml
  python generate_thessaloniki.py --period 2.0     # sparser traffic (fewer vehicles)
  python generate_thessaloniki.py --gui            # open sumo-gui after building

Reference area:
  OpenStreetMap contributors, © OpenStreetMap — openstreetmap.org/copyright
  Data downloaded via Overpass API — overpass-api.de
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import requests
import yaml


# ---------------------------------------------------------------------------
# Network bounding box — Thessaloniki city centre
# ---------------------------------------------------------------------------

# Overpass API bounding box: (south, west, north, east)
# This covers roughly: waterfront → Ano Poli (N), Neapoli (E), Kalamaria (SE)
BBOX = (40.610, 22.925, 40.660, 22.995)

SUMO_HOME = os.environ.get(
    "SUMO_HOME",
    "/opt/homebrew/Cellar/sumo/1.20.0/share/sumo",
)
TYPEMAP = os.path.join(SUMO_HOME, "data", "typemap", "osmNetconvert.typ.xml")
RANDOM_TRIPS = os.path.join(SUMO_HOME, "tools", "randomTrips.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], desc: str = "") -> subprocess.CompletedProcess:
    """Run a subprocess, print the command, exit on failure."""
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


# ---------------------------------------------------------------------------
# Step 1 — Download OSM data
# ---------------------------------------------------------------------------

def download_osm(out_path: str, bbox: tuple[float, float, float, float]) -> None:
    """
    Download road network OSM data for the bounding box via the Overpass API.

    Fetches only driveable ways (highway=*) and their associated nodes —
    much smaller than a full OSM dump of the area.
    """
    south, west, north, east = bbox
    query = f"""
[out:xml][timeout:120];
(
  way["highway"]({south},{west},{north},{east});
  >;
);
out body;
"""
    url = "https://overpass-api.de/api/interpreter"
    print(f"    Querying Overpass API ({south},{west} → {north},{east}) …")
    print("    (this may take 20–60 seconds on first download)")

    for attempt in range(1, 4):
        try:
            resp = requests.post(url, data={"data": query}, timeout=180)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(resp.content)
            size_mb = os.path.getsize(out_path) / 1_048_576
            print(f"    Downloaded {size_mb:.1f} MB → {out_path}")
            return
        except requests.RequestException as exc:
            if attempt == 3:
                print(f"❌  Overpass download failed after 3 attempts: {exc}")
                sys.exit(1)
            print(f"    Attempt {attempt} failed ({exc}), retrying in 10 s …")
            time.sleep(10)


# ---------------------------------------------------------------------------
# Step 2 — Convert OSM → SUMO network
# ---------------------------------------------------------------------------

def build_network(osm_path: str, net_path: str, typemap: str) -> None:
    """
    Run netconvert to convert the OSM file into a SUMO network.

    Key options chosen for a dense urban network:
      --geometry.remove             merge short edges and remove redundant geometry
      --roundabouts.guess           detect roundabouts and apply correct right-of-way
      --ramps.guess                 detect on/off ramps and add acceleration lanes
      --junctions.join              merge nearby junction nodes (OSM has many duplicates)
      --tls.guess-signals           detect traffic lights from OSM data
      --tls.discard-simple          remove TLS from simple junctions (reduces clutter)
      --tls.join                    merge nearby TLS programmes
      --keep-edges.by-vclass        keep only passenger/truck roads (drop footpaths etc.)
    """
    if not os.path.exists(typemap):
        print(f"⚠   typemap not found at {typemap}")
        print(f"    Set SUMO_HOME correctly or install SUMO.")
        sys.exit(1)

    _run([
        "netconvert",
        "--osm-files",               osm_path,
        "--type-files",              typemap,
        "--output-file",             net_path,
        "--geometry.remove",
        "--roundabouts.guess",
        "--ramps.guess",
        "--junctions.join",
        "--tls.guess-signals",       "true",
        "--tls.discard-simple",      "false",
        "--tls.join",                "true",
        "--keep-edges.by-vclass",    "passenger,truck",
        "--no-internal-links",       "false",
        "--no-warnings",             "true",
    ], desc="netconvert")
    size_mb = os.path.getsize(net_path) / 1_048_576
    print(f"    Network compiled → {net_path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Step 3 — Generate routes
# ---------------------------------------------------------------------------

def generate_routes(net_path: str, rou_path: str, period: float) -> None:
    """
    Generate randomised vehicle trips using SUMO's randomTrips.py.

    period controls vehicle density:
      0.5  → ~400–600 simultaneous vehicles (dense, heavy congestion)
      1.0  → ~200–300 simultaneous vehicles  ← recommended first run
      2.0  → ~100–150 simultaneous vehicles (light, faster simulation)
    """
    if not os.path.exists(RANDOM_TRIPS):
        print(f"❌  randomTrips.py not found at {RANDOM_TRIPS}")
        print(f"    Set SUMO_HOME correctly.")
        sys.exit(1)

    _run([
        sys.executable, RANDOM_TRIPS,
        "-n",          net_path,
        "-o",          rou_path,
        "--period",    str(period),
        "--begin",     "0",
        "--end",       "7200",
        "--validate",
        "--remove-loops",
    ], desc="randomTrips.py")

    # Count generated vehicles
    try:
        tree   = ET.parse(rou_path)
        root   = tree.getroot()
        n_veh  = len(root.findall("vehicle")) + len(root.findall("trip"))
        print(f"    Routes generated → {rou_path}  ({n_veh:,} trips)")
    except Exception:
        print(f"    Routes generated → {rou_path}")


# ---------------------------------------------------------------------------
# Step 4 — Write .sumocfg
# ---------------------------------------------------------------------------

def write_sumocfg(cfg_path: str, net_file: str, rou_file: str) -> None:
    root = ET.Element("configuration")

    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file",    value=net_file)
    ET.SubElement(inp, "route-files", value=rou_file)

    tim = ET.SubElement(root, "time")
    ET.SubElement(tim, "begin", value="0")
    ET.SubElement(tim, "end",   value="7200")

    proc = ET.SubElement(root, "processing")
    ET.SubElement(proc, "ignore-route-errors", value="true")

    raw    = ET.tostring(root, encoding="unicode")
    pretty = _pretty_xml(raw)
    _write(cfg_path, pretty)


def _pretty_xml(raw: str) -> str:
    from xml.dom import minidom
    return minidom.parseString(raw).toprettyxml(indent="    ")


# ---------------------------------------------------------------------------
# Step 5 — Patch config.yaml
# ---------------------------------------------------------------------------

def update_config_yaml(config_path: str, sumocfg_path: str, binary: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["sumo"]["config_file"] = os.path.abspath(sumocfg_path)
    cfg["sumo"]["binary"]      = binary

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\n  ✅  config.yaml updated:")
    print(f"      sumo.config_file = {os.path.abspath(sumocfg_path)}")
    print(f"      sumo.binary      = {binary}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate the Thessaloniki city-centre SUMO network from OSM"
    )
    ap.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "thessaloniki_network"),
        help="Output directory (default: ./thessaloniki_network/)",
    )
    ap.add_argument(
        "--period", type=float, default=1.0,
        help=(
            "randomTrips vehicle insertion period in seconds "
            "(lower = more vehicles; 1.0 → ~200–300 simultaneous, default)"
        ),
    )
    ap.add_argument(
        "--update-config", action="store_true",
        help="Patch config.yaml so runner.py uses this network immediately",
    )
    ap.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
        help="Path to config.yaml to update (default: ./config.yaml)",
    )
    ap.add_argument(
        "--gui", action="store_true",
        help="Set binary to sumo-gui in config.yaml (opens graphical interface)",
    )
    ap.add_argument(
        "--skip-download", action="store_true",
        help="Skip OSM download if thessaloniki.osm already exists in --out-dir",
    )
    args = ap.parse_args()

    out     = args.out_dir
    os.makedirs(out, exist_ok=True)

    osm_path = os.path.join(out, "thessaloniki.osm")
    net_path = os.path.join(out, "thessaloniki.net.xml")
    rou_path = os.path.join(out, "thessaloniki.rou.xml")
    cfg_path = os.path.join(out, "thessaloniki.sumocfg")

    print(f"\n{'='*60}")
    print(f"  Thessaloniki City-Centre Network")
    print(f"  OSM bbox: {BBOX[0]}°N {BBOX[1]}°E  →  {BBOX[2]}°N {BBOX[3]}°E")
    print(f"  Output   → {out}")
    print(f"{'='*60}\n")

    # 1. Download
    if args.skip_download and os.path.exists(osm_path):
        print(f"[ 1/4 ]  Skipping download — using existing {osm_path}")
    else:
        print("[ 1/4 ]  Downloading OSM data …")
        download_osm(osm_path, BBOX)

    # 2. Build network
    print("\n[ 2/4 ]  Building SUMO network …")
    build_network(osm_path, net_path, TYPEMAP)

    # 3. Generate routes
    print(f"\n[ 3/4 ]  Generating routes  (period={args.period}s) …")
    generate_routes(net_path, rou_path, args.period)

    # 4. Write .sumocfg
    print("\n[ 4/4 ]  Writing run configuration …")
    write_sumocfg(
        cfg_path,
        net_file=os.path.basename(net_path),
        rou_file=os.path.basename(rou_path),
    )

    # 5. Optionally update config.yaml
    if args.update_config and os.path.exists(args.config):
        binary = "sumo-gui" if args.gui else "sumo"
        update_config_yaml(args.config, cfg_path, binary)

    print(f"\n{'='*60}")
    print(f"  Network ready!")
    print(f"\n  Next steps:")
    print(f"    1. Update config.yaml → sumo.config_file: {cfg_path}")
    print(f"       (or re-run with --update-config to do this automatically)")
    print(f"    2. Run the simulation:")
    print(f"         python runner.py")
    print(f"    3. Or open in SUMO GUI first to inspect the network:")
    print(f"         sumo-gui -c {cfg_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
