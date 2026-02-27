"""
generate_sioux_falls.py
=======================
Generate the Sioux Falls benchmark road network for SUMO.

The Sioux Falls network is the canonical small benchmark in transportation
research — introduced by LeBlanc, Morlok & Pierskalla (1975) and used in
hundreds of papers on traffic assignment, network reliability, and resilience.

Network summary
---------------
  Nodes  : 24
  Links  : 76 directed  (38 bidirectional roads)
  Area   : ~7.5 km × 3.5 km
  Speed  : 50 km/h uniform (arterial grid)
  Lanes  : 2 per direction

Compared with the Thessaloniki OSM network (8,848 edges, 2,000+ vehicles)
this is ~100× smaller, making single development runs finish in seconds.

Usage
-----
    python generate_sioux_falls.py                  # generate network + routes
    python generate_sioux_falls.py --update-config  # also patch config.yaml
    python generate_sioux_falls.py --period 2.0     # adjust vehicle density

Reference
---------
LeBlanc, L. J., Morlok, E. K., & Pierskalla, W. P. (1975).
An efficient approach to solving the road network equilibrium traffic
assignment problem. Transportation Research, 9(5), 309-318.
"""

import argparse
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

import yaml


# ---------------------------------------------------------------------------
# Network topology  (LeBlanc et al. 1975)
# ---------------------------------------------------------------------------

# Node coordinates in metres.
# Scaled so 1 abstract unit ≈ 1 km; network spans ~7.5 km (E-W) × 3.5 km (N-S).
NODES: dict[int, tuple[int, int]] = {
     1: (1000, 5000),   2: (4000, 5000),   3: (7000, 5000),
     4: (2000, 4500),   5: (6000, 4500),
     6: (3000, 4000),   7: (5000, 4000),
     8: (3000, 3500),   9: (5000, 3500),  10: (7000, 3500),
    11: (1500, 3000),  12: (3000, 3000),  13: (5000, 3000),  14: (7000, 3000),
    15: (1500, 2500),  16: (3000, 2500),  17: (5000, 2500),  18: (7000, 2500),
    19: (1500, 2000),  20: (3500, 2000),  21: (5000, 2000),
    22: (1500, 1500),  23: (3500, 1500),  24: (5000, 1500),
}

# 38 undirected links — each becomes TWO directed SUMO edges (A→B and B→A).
LINKS: list[tuple[int, int]] = [
    ( 1,  2), ( 1,  3),
    ( 2,  6),
    ( 3,  4), ( 3, 12),
    ( 4,  5), ( 4, 11),
    ( 5,  6), ( 5,  9),
    ( 6,  7), ( 6,  8),
    ( 7,  8), ( 7,  9),
    ( 8,  9), ( 8, 11), ( 8, 16),
    ( 9, 10), ( 9, 13),
    (10, 14),
    (11, 12), (11, 15),
    (12, 13),
    (13, 14), (13, 17),
    (14, 18),
    (15, 16), (15, 19),
    (16, 17), (16, 20),
    (17, 18), (17, 21),
    (19, 20), (19, 22),
    (20, 21), (20, 23),
    (21, 24),
    (22, 23),
    (23, 24),
]

# Road parameters — uniform arterial grid
SPEED_MS = 50.0 / 3.6   # 50 km/h ≈ 13.89 m/s
LANES    = 2
PRIORITY = 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _link_length_m(a: int, b: int) -> float:
    """Euclidean distance between two nodes (metres)."""
    x1, y1 = NODES[a]
    x2, y2 = NODES[b]
    return math.hypot(x2 - x1, y2 - y1)


def _pretty_xml(root: ET.Element) -> str:
    """Return nicely indented XML string."""
    raw = ET.tostring(root, encoding="unicode")
    return minidom.parseString(raw).toprettyxml(indent="    ")


def _run(cmd: list[str]):
    """Run a subprocess, printing the command; exit on failure."""
    print(f"    $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-3000:])
        sys.exit(f"\nERROR: command failed — {cmd[0]}")
    return result


def _find_random_trips() -> str | None:
    """Locate randomTrips.py via SUMO_HOME or common install paths."""
    candidates = [
        os.path.join(os.environ.get("SUMO_HOME", ""), "tools", "randomTrips.py"),
        "/opt/homebrew/share/sumo/tools/randomTrips.py",
        "/usr/share/sumo/tools/randomTrips.py",
        "/usr/local/share/sumo/tools/randomTrips.py",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def write_nodes(path: str):
    root = ET.Element("nodes")
    for nid, (x, y) in NODES.items():
        ET.SubElement(root, "node",
                      id=str(nid), x=str(x), y=str(y),
                      type="priority")
    with open(path, "w") as f:
        f.write(_pretty_xml(root))
    print(f"    nodes.xml  — {len(NODES)} nodes")


def write_edges(path: str):
    root = ET.Element("edges")
    total_km = 0.0
    for (a, b) in LINKS:
        length = _link_length_m(a, b)
        total_km += length / 500.0   # both directions, in km
        for frm, to in [(a, b), (b, a)]:
            ET.SubElement(root, "edge",
                          id=f"{frm}to{to}",
                          **{"from": str(frm), "to": str(to)},
                          numLanes=str(LANES),
                          speed=str(round(SPEED_MS, 4)),
                          priority=str(PRIORITY))
    with open(path, "w") as f:
        f.write(_pretty_xml(root))
    print(f"    edges.xml  — {len(LINKS) * 2} directed edges  "
          f"({total_km:.1f} km total road length)")


def write_sumocfg(cfg_path: str, net_path: str, rou_path: str):
    root = ET.Element("configuration")
    inp  = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file",    value=net_path)
    ET.SubElement(inp, "route-files", value=rou_path)
    tim  = ET.SubElement(root, "time")
    ET.SubElement(tim, "begin", value="0")
    ET.SubElement(tim, "end",   value="7200")
    with open(cfg_path, "w") as f:
        f.write(_pretty_xml(root))
    print(f"    sumocfg    — {cfg_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate the Sioux Falls benchmark network for SUMO"
    )
    ap.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "sioux_falls_network"),
        help="Output directory (default: ./sioux_falls_network/)",
    )
    ap.add_argument(
        "--period", type=float, default=1.5,
        help=(
            "randomTrips vehicle insertion period in seconds "
            "(lower = more vehicles; default: 1.5 → ~150 simultaneous)"
        ),
    )
    ap.add_argument(
        "--update-config", action="store_true",
        help="Patch config.yaml so runner.py uses this network immediately",
    )
    ap.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "config.yaml"),
        help="Path to config.yaml to update (default: ./config.yaml)",
    )
    args = ap.parse_args()

    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    nodes_path = os.path.join(out, "sioux_falls.nod.xml")
    edges_path = os.path.join(out, "sioux_falls.edg.xml")
    net_path   = os.path.join(out, "sioux_falls.net.xml")
    rou_path   = os.path.join(out, "sioux_falls.rou.xml")
    cfg_path   = os.path.join(out, "sioux_falls.sumocfg")

    print(f"\n{'='*55}")
    print(f"  Sioux Falls benchmark network  (LeBlanc et al. 1975)")
    print(f"  24 nodes · 76 directed links · 50 km/h arterial grid")
    print(f"{'='*55}\n")

    # 1. Write source XML
    print("[ 1/4 ]  Writing network source files …")
    write_nodes(nodes_path)
    write_edges(edges_path)

    # 2. netconvert
    print("\n[ 2/4 ]  Running netconvert …")
    _run([
        "netconvert",
        "--node-files",   nodes_path,
        "--edge-files",   edges_path,
        "--output-file",  net_path,
        "--no-warnings",  "true",
        "--junctions.join", "false",
    ])
    print(f"    net.xml    — {net_path}")

    # 3. Generate routes
    print(f"\n[ 3/4 ]  Generating routes  (period={args.period}s) …")
    rand_trips = _find_random_trips()
    if rand_trips is None:
        print("    WARNING: randomTrips.py not found.")
        print("    Set $SUMO_HOME or run randomTrips.py manually:")
        print(f"      randomTrips.py -n {net_path} -o {rou_path} "
              f"--period {args.period} --begin 0 --end 7200 --validate")
    else:
        _run([
            sys.executable, rand_trips,
            "-n", net_path,
            "-o", rou_path,
            "--period", str(args.period),
            "--begin",  "0",
            "--end",    "7200",
            "--validate",
        ])
        # Count generated trips (randomTrips --validate emits <trip> elements)
        try:
            tree  = ET.parse(rou_path)
            root_ = tree.getroot()
            n_veh = len(root_.findall("vehicle")) + len(root_.findall("trip"))
            avg_trip_s = (7000 / SPEED_MS) / 3   # rough estimate: 1/3 of network
            simult     = round(n_veh * avg_trip_s / 7200)
            print(f"    routes.xml — {n_veh:,} trips  (~{simult} simultaneous)")
        except Exception:
            print(f"    routes.xml — {rou_path}")

    # 4. Write .sumocfg
    print("\n[ 4/4 ]  Writing simulation config …")
    write_sumocfg(cfg_path,
                  os.path.basename(net_path),
                  os.path.basename(rou_path))

    # 5. Optionally patch config.yaml
    if args.update_config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        cfg["sumo"]["config_file"] = cfg_path
        with open(args.config, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"\n  config.yaml updated → sumo.config_file = {cfg_path}")

    # Summary
    total_links_km = sum(_link_length_m(a, b) for a, b in LINKS) / 500.0
    print(f"\n{'='*55}")
    print(f"  Network ready!")
    print(f"  Nodes  : {len(NODES)}")
    print(f"  Links  : {len(LINKS) * 2} directed  ({len(LINKS)} bidirectional)")
    print(f"  Length : {total_links_km:.1f} km total road")
    print(f"\n  Test run  : python runner.py")
    print(f"  GUI view  : sumo-gui -c {cfg_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
