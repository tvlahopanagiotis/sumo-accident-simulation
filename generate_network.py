#!/usr/bin/env python3
"""
generate_network.py
===================
Generates a realistic SUMO urban road network for the Accident Simulation.

Network: "Riverside District"
  A medium-sized urban district (~1.6 km × 1.6 km) featuring:
    - 5 × 4 arterial intersection grid with traffic lights
    - A northern expressway bypass (3-lane, 90 km/h)
    - Mixed road types: outer ring (60 km/h), inner arterials (50 km/h)
    - Realistic AM peak / off-peak / PM peak traffic demand
    - Multiple vehicle types (cars, trucks, motorcycles)

Output folder (--out-dir, default: riverside_network/):
  riverside.nod.xml     junction definitions
  riverside.edg.xml     edge definitions
  riverside.typ.xml     road type library
  riverside.net.xml     compiled SUMO network  (via netconvert)
  riverside.rou.xml     traffic demand
  riverside.sumocfg     SUMO run configuration

Usage:
  python generate_network.py                         # build in ./riverside_network/
  python generate_network.py --out-dir /tmp/mynet    # custom output folder
  python generate_network.py --update-config         # also patch config.yaml
"""

import argparse
import os
import subprocess
import sys
import textwrap


# ---------------------------------------------------------------------------
# Network geometry
# ---------------------------------------------------------------------------

# Column labels A-E and their x-coordinates (metres)
COLS = [("A", 0), ("B", 400), ("C", 800), ("D", 1200), ("E", 1600)]

# Row labels 0-3 and their y-coordinates (metres)
# Deliberately non-uniform to break the sterile toy-grid feel
ROWS = [(0, 0), (1, 380), (2, 780), (3, 1200)]

# Bypass nodes (north expressway)
BYPASS_W = "BP_W"
BYPASS_E = "BP_E"
BYPASS_Y = 1540


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nid(col: str, row: int) -> str:
    """Return the junction ID for column letter + row index, e.g. 'C2'."""
    return f"{col}{row}"


def eid(src: str, dst: str) -> str:
    """Return an edge ID from src junction to dst junction."""
    return f"{src}{dst}"


# ---------------------------------------------------------------------------
# Node generation
# ---------------------------------------------------------------------------

def build_nodes() -> list[dict]:
    """
    Return a list of node dicts with keys: id, x, y, type.

    Junction types:
      traffic_light   — signalised arterial intersections
      priority        — un-signalised / expressway nodes
    """
    nodes = []

    # ── Expressway bypass ────────────────────────────────────────────────────
    nodes.append({"id": BYPASS_W, "x": 0,    "y": BYPASS_Y, "type": "priority"})
    nodes.append({"id": BYPASS_E, "x": 1600, "y": BYPASS_Y, "type": "priority"})

    # ── Main arterial grid ───────────────────────────────────────────────────
    for col, x in COLS:
        for row, y in ROWS:
            node_id = nid(col, row)
            # Outer-perimeter nodes use priority (free flow); inner = lights
            is_inner = (col not in ("A", "E")) and (row not in (0, 3))
            ntype = "traffic_light" if is_inner else "priority"
            nodes.append({"id": node_id, "x": x, "y": y, "type": ntype})

    return nodes


# ---------------------------------------------------------------------------
# Edge generation
# ---------------------------------------------------------------------------

# Road type identifiers defined in the .typ.xml
TYPE_EXPRESSWAY  = "expressway"
TYPE_OUTER_ART   = "arterial_outer"   # perimeter of the grid
TYPE_INNER_ART   = "arterial_inner"   # inner cross streets
TYPE_RAMP        = "ramp"             # bypass on/off ramps


def build_edges() -> list[dict]:
    """
    Return a list of edge dicts with keys:
      id, from, to, type, (optional) numLanes, speed
    Both directions of every road are represented as separate edges.
    """
    edges = []

    col_ids  = [c for c, _ in COLS]
    row_ids  = [r for r, _ in ROWS]

    # ── Expressway bypass (E→W and W→E one-way edges) ────────────────────────
    edges.append({"id": "BP_WE", "from": BYPASS_W, "to": BYPASS_E, "type": TYPE_EXPRESSWAY})
    edges.append({"id": "BP_EW", "from": BYPASS_E, "to": BYPASS_W, "type": TYPE_EXPRESSWAY})

    # ── Bypass ramps: A3 ↔ BP_W  and  E3 ↔ BP_E ────────────────────────────
    edges.append({"id": "A3BP_W", "from": nid("A", 3), "to": BYPASS_W, "type": TYPE_RAMP})
    edges.append({"id": "BP_WA3", "from": BYPASS_W, "to": nid("A", 3), "type": TYPE_RAMP})
    edges.append({"id": "E3BP_E", "from": nid("E", 3), "to": BYPASS_E, "type": TYPE_RAMP})
    edges.append({"id": "BP_EE3", "from": BYPASS_E, "to": nid("E", 3), "type": TYPE_RAMP})

    # ── Vertical edges (N–S within each column) ──────────────────────────────
    for col in col_ids:
        for i in range(len(row_ids) - 1):
            lo_row = row_ids[i]
            hi_row = row_ids[i + 1]
            src = nid(col, lo_row)
            dst = nid(col, hi_row)

            etype = _edge_type_for(col, lo_row, hi_row, axis="vertical")
            edges.append({"id": eid(src, dst), "from": src, "to": dst, "type": etype})
            edges.append({"id": eid(dst, src), "from": dst, "to": src, "type": etype})

    # ── Horizontal edges (E–W within each row) ───────────────────────────────
    for row in row_ids:
        for i in range(len(col_ids) - 1):
            lo_col = col_ids[i]
            hi_col = col_ids[i + 1]
            src = nid(lo_col, row)
            dst = nid(hi_col, row)

            etype = _edge_type_for(lo_col, row, row, axis="horizontal",
                                   col2=hi_col)
            edges.append({"id": eid(src, dst), "from": src, "to": dst, "type": etype})
            edges.append({"id": eid(dst, src), "from": dst, "to": src, "type": etype})

    return edges


def _edge_type_for(col: str, row_a: int, row_b: int, axis: str,
                   col2: str = "") -> str:
    """Choose road type based on which part of the grid an edge sits in."""
    col_ids = [c for c, _ in COLS]
    row_ids = [r for r, _ in ROWS]

    if axis == "vertical":
        # Outermost columns (A and E) → outer arterial
        if col in (col_ids[0], col_ids[-1]):
            return TYPE_OUTER_ART
        # Top or bottom row edge → outer arterial
        if row_a == row_ids[0] or row_b == row_ids[-1]:
            return TYPE_OUTER_ART
        return TYPE_INNER_ART

    else:  # horizontal
        # Top and bottom rows → outer arterial
        if row_a in (row_ids[0], row_ids[-1]):
            return TYPE_OUTER_ART
        # Leftmost or rightmost column transition → outer arterial
        if col in (col_ids[0],) or col2 in (col_ids[-1],):
            return TYPE_OUTER_ART
        return TYPE_INNER_ART


# ---------------------------------------------------------------------------
# XML writers
# ---------------------------------------------------------------------------

def write_types(path: str):
    xml = textwrap.dedent(f"""\
    <?xml version="1.0" encoding="UTF-8"?>
    <types>
        <!-- 3-lane motorway / bypass: 90 km/h -->
        <type id="{TYPE_EXPRESSWAY}"
              priority="12" numLanes="3" speed="25.00"
              oneway="1" allow="passenger truck"/>

        <!-- On/off ramps for the bypass: 60 km/h, 1 lane -->
        <type id="{TYPE_RAMP}"
              priority="10" numLanes="1" speed="16.67"
              oneway="0" allow="passenger truck"/>

        <!-- Outer-ring arterial roads: 2 lanes, 60 km/h -->
        <type id="{TYPE_OUTER_ART}"
              priority="9" numLanes="2" speed="16.67"
              oneway="0" allow="passenger truck motorcycle"/>

        <!-- Inner-district arterial roads: 2 lanes, 50 km/h -->
        <type id="{TYPE_INNER_ART}"
              priority="7" numLanes="2" speed="13.89"
              oneway="0" allow="passenger truck motorcycle"/>
    </types>
    """)
    _write(path, xml)


def write_nodes(path: str, nodes: list[dict]):
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<nodes>"]
    for n in nodes:
        lines.append(
            f'    <node id="{n["id"]}" x="{n["x"]}" y="{n["y"]}"'
            f' type="{n["type"]}"/>'
        )
    lines.append("</nodes>")
    _write(path, "\n".join(lines))


def write_edges(path: str, edges: list[dict]):
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<edges>"]
    for e in edges:
        lines.append(
            f'    <edge id="{e["id"]}" from="{e["from"]}" to="{e["to"]}"'
            f' type="{e["type"]}"/>'
        )
    lines.append("</edges>")
    _write(path, "\n".join(lines))


# ---------------------------------------------------------------------------
# Traffic demand
# ---------------------------------------------------------------------------

# Edge IDs that serve as sensible entry/exit points
ENTRY_SOUTH  = [eid(nid("A", 0), nid("A", 1)),   # A0 → A1  (west side, from south)
                eid(nid("B", 0), nid("B", 1)),
                eid(nid("C", 0), nid("C", 1)),
                eid(nid("D", 0), nid("D", 1)),
                eid(nid("E", 0), nid("E", 1))]

ENTRY_NORTH  = [eid(nid("A", 3), nid("A", 2)),
                eid(nid("C", 3), nid("C", 2)),
                eid(nid("E", 3), nid("E", 2))]

ENTRY_WEST   = [eid(nid("A", 0), nid("B", 0)),
                eid(nid("A", 1), nid("B", 1)),
                eid(nid("A", 2), nid("B", 2)),
                eid(nid("A", 3), nid("B", 3))]

ENTRY_EAST   = [eid(nid("E", 0), nid("D", 0)),
                eid(nid("E", 1), nid("D", 1)),
                eid(nid("E", 2), nid("D", 2)),
                eid(nid("E", 3), nid("D", 3))]

BYPASS_ENTRY = ["BP_WE", "BP_EW"]

EXIT_NORTH   = [eid(nid("A", 2), nid("A", 3)),
                eid(nid("C", 2), nid("C", 3)),
                eid(nid("E", 2), nid("E", 3))]

EXIT_SOUTH   = ENTRY_NORTH[::-1]   # reverse direction edges
EXIT_EAST    = [eid(nid("D", r), nid("E", r)) for r in range(4)]
EXIT_WEST    = [eid(nid("B", r), nid("A", r)) for r in range(4)]


def write_routes(path: str):
    """
    Generate realistic AM-peak / off-peak / PM-peak traffic demand.

    Periods (seconds):
      0   – 1800  : AM peak     — heavy inbound (S→N, W→E)
      1800 – 3600 : Shoulder    — moderate all-direction
      3600 – 5400 : Midday      — light background
      5400 – 7200 : PM peak     — heavy outbound (N→S, E→W)
    """

    flows = []
    fid = 0

    def add_flow(from_edge, to_edge, begin, end, vph, vtype="car"):
        nonlocal fid
        flows.append(
            f'    <flow id="f{fid:04d}" type="{vtype}"'
            f' from="{from_edge}" to="{to_edge}"'
            f' begin="{begin}" end="{end}"'
            f' vehsPerHour="{vph}"'
            f' departLane="best" departSpeed="max"/>'
        )
        fid += 1

    # ── AM peak 0–1800 s  (heavy inbound) ───────────────────────────────────
    for e_from in ENTRY_SOUTH:
        for e_to in EXIT_NORTH:
            add_flow(e_from, e_to, 0, 1800, 300)
    for e_from in ENTRY_WEST:
        for e_to in EXIT_EAST:
            add_flow(e_from, e_to, 0, 1800, 200)
    # bypass through-traffic AM
    add_flow("BP_WE", EXIT_EAST[2], 0, 1800, 400, "truck")
    add_flow("BP_WE", EXIT_EAST[2], 0, 1800, 600)

    # ── Shoulder 1800–3600 s ─────────────────────────────────────────────────
    for e_from in ENTRY_SOUTH + ENTRY_WEST:
        for e_to in EXIT_NORTH[:2] + EXIT_EAST[:2]:
            add_flow(e_from, e_to, 1800, 3600, 120)
    # cross-district
    for e_from in ENTRY_EAST:
        for e_to in EXIT_WEST[:2]:
            add_flow(e_from, e_to, 1800, 3600, 100)

    # ── Midday 3600–5400 s  (light background) ──────────────────────────────
    for e_from in ENTRY_SOUTH + ENTRY_NORTH + ENTRY_WEST + ENTRY_EAST:
        for e_to in EXIT_SOUTH[:1] + EXIT_NORTH[:1] + EXIT_EAST[:1]:
            add_flow(e_from, e_to, 3600, 5400, 60)
    add_flow("BP_WE", "BP_EW", 3600, 5400, 200)   # bypass recirculation

    # ── PM peak 5400–7200 s  (heavy outbound) ───────────────────────────────
    for e_from in ENTRY_NORTH:
        for e_to in EXIT_SOUTH:
            add_flow(e_from, e_to, 5400, 7200, 300)
    for e_from in ENTRY_EAST:
        for e_to in EXIT_WEST:
            add_flow(e_from, e_to, 5400, 7200, 250)
    add_flow("BP_EW", EXIT_WEST[1], 5400, 7200, 400, "truck")
    add_flow("BP_EW", EXIT_WEST[1], 5400, 7200, 600)

    xml = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <routes>
        <!-- ── Vehicle types ─────────────────────────────────── -->
        <vType id="car"
               accel="2.6" decel="4.5" sigma="0.5"
               length="4.5" maxSpeed="16.67"
               color="0.2,0.6,1.0"/>
        <vType id="truck"
               accel="1.0" decel="3.0" sigma="0.3"
               length="12.0" maxSpeed="11.11"
               color="0.8,0.4,0.1" guiShape="truck"/>
        <vType id="motorcycle"
               accel="4.0" decel="5.0" sigma="0.6"
               length="2.0" maxSpeed="19.44"
               color="0.9,0.8,0.1" guiShape="motorcycle"/>

        <!-- ── Flows ──────────────────────────────────────────── -->
    """)
    xml += "\n".join(flows)
    xml += "\n</routes>\n"
    _write(path, xml)


# ---------------------------------------------------------------------------
# SUMO config
# ---------------------------------------------------------------------------

def write_sumocfg(path: str, net_file: str, rou_file: str):
    xml = textwrap.dedent(f"""\
    <configuration>
        <input>
            <net-file   value="{net_file}"/>
            <route-files value="{rou_file}"/>
        </input>
        <time>
            <begin value="0"/>
            <end   value="7200"/>
        </time>
        <processing>
            <ignore-route-errors value="true"/>
        </processing>
    </configuration>
    """)
    _write(path, xml)


# ---------------------------------------------------------------------------
# netconvert
# ---------------------------------------------------------------------------

def run_netconvert(out_dir: str, nod: str, edg: str, typ: str, net: str):
    cmd = [
        "netconvert",
        "--node-files",        nod,
        "--edge-files",        edg,
        "--type-files",        typ,
        "--output-file",       net,
        "--no-warnings",       "true",
        "--junctions.join",    "false",
        "--tls.discard-simple","false",
        "--roundabouts.guess", "false",
    ]
    print(f"\n▶  Running netconvert …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌  netconvert failed:")
        print(result.stderr)
        sys.exit(1)
    print(f"✅  Network compiled → {net}")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _write(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"   wrote {path}")


def update_config_yaml(config_path: str, sumocfg_path: str):
    """Patch the binary and config_file keys in config.yaml."""
    import yaml  # only needed here

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["sumo"]["config_file"] = os.path.abspath(sumocfg_path)
    cfg["sumo"]["binary"] = "sumo"   # headless for the runner

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\n✅  config.yaml updated → config_file now points to the new network.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate the Riverside District SUMO network"
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "riverside_network"
        ),
        help="Output folder for network files (default: ../riverside_network/)",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Also update config.yaml to point at the new network",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Riverside District — Network Generator")
    print(f"  Output → {out_dir}")
    print(f"{'='*60}\n")

    # File paths
    nod_file = os.path.join(out_dir, "riverside.nod.xml")
    edg_file = os.path.join(out_dir, "riverside.edg.xml")
    typ_file = os.path.join(out_dir, "riverside.typ.xml")
    net_file = os.path.join(out_dir, "riverside.net.xml")
    rou_file = os.path.join(out_dir, "riverside.rou.xml")
    cfg_file = os.path.join(out_dir, "riverside.sumocfg")

    # Build data
    nodes = build_nodes()
    edges = build_edges()

    print(f"Network summary:")
    print(f"  Junctions : {len(nodes)}")
    print(f"  Edges     : {len(edges)}")

    # Write XML sources
    write_types(typ_file)
    write_nodes(nod_file, nodes)
    write_edges(edg_file, edges)
    write_routes(rou_file)
    write_sumocfg(cfg_file,
                  net_file=os.path.basename(net_file),
                  rou_file=os.path.basename(rou_file))

    # Compile network
    run_netconvert(out_dir, nod_file, edg_file, typ_file, net_file)

    # Optionally patch config.yaml
    if args.update_config:
        config_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "config.yaml")
        if os.path.exists(config_yaml):
            update_config_yaml(config_yaml, cfg_file)
        else:
            print(f"⚠  config.yaml not found at {config_yaml} — skipping update.")

    print(f"\n{'='*60}")
    print(f"  Done!  Run the simulation with:")
    print(f"    python runner.py")
    print(f"  (after updating config.yaml → sumo.config_file: {cfg_file})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
