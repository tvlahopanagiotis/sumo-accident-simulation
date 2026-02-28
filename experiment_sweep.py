"""
experiment_sweep.py
===================
Parameter-grid sweep for network failure-point analysis.

Sweeps across:
  - Traffic load         (PERIODS — vehicle insertion period in seconds;
                          lower = denser traffic)
  - Accident probability (PROBS — base_probability per vehicle per step)
  - Random seeds         (for stochastic robustness)

For each (period, prob, seed) cell the script:
  1. Generates SUMO routes at the target period  (cached — not regenerated
     if the file already exists)
  2. Writes a per-cell config YAML with overridden values
  3. Runs runner.py as a subprocess
  4. Reads the output JSONs and network_metrics.csv
  5. Appends one row to  results/sweep/sweep_results.csv

Usage
-----
    python experiment_sweep.py                          # full grid (~70 runs)
    python experiment_sweep.py --seeds 3                # triplicate every cell
    python experiment_sweep.py --periods 2.0 1.0 0.5   # subset of periods
    python experiment_sweep.py --probs 0 1.5e-4 1e-3   # subset of probs

Output
------
  results/sweep/
    sweep_results.csv          ← one row per (period, prob, seed)
    runs/                      ← per-cell runner output trees
    sweep_log.txt              ← stdout/stderr from every subprocess call
"""

import argparse
import copy
import csv
import datetime
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom

import yaml


# ---------------------------------------------------------------------------
# Default parameter grid
# ---------------------------------------------------------------------------

# Vehicle insertion periods (seconds).  Low period = many vehicles = high load.
DEFAULT_PERIODS: list[float] = [5.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5]

# Accident base-probability levels.  0.0 → baseline run (effectively no accidents).
DEFAULT_PROBS: list[float] = [0.0, 5e-5, 1.5e-4, 5e-4, 1e-3]

# Stochastic replicates per (period, prob) cell
DEFAULT_SEEDS: int = 2

# Seeds used for replicates: BASE_SEED, BASE_SEED+1, …
BASE_SEED: int = 42


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "period",            # float  — vehicle insertion period (s)
    "prob",              # float  — base_probability setting
    "seed",              # int    — random seed used
    "n_accidents",       # int    — total accidents triggered
    "mean_speed_ms",     # float  — mean network speed over run (m/s)
    "mean_speed_kmh",    # float  — same, converted to km/h
    "mean_throughput",   # float  — mean vehicles/hour completing routes
    "mean_speed_ratio",  # float  — mean(speed / free-flow baseline)
    "ai",                # float  — Antifragility Index (None if no accidents)
    "ci_low",            # float  — 95 % CI lower bound
    "ci_high",           # float  — 95 % CI upper bound
    "n_events_measured", # int    — accidents contributing to AI
    "run_dir",           # str    — path to this cell's output folder
]


# ---------------------------------------------------------------------------
# Route / config helpers
# ---------------------------------------------------------------------------

def _find_random_trips() -> str | None:
    """Locate SUMO's randomTrips.py via SUMO_HOME or common install paths."""
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


def _generate_routes(
    net_path: str,
    rou_path: str,
    period: float,
    end: int = 7200,
) -> bool:
    """
    Run randomTrips.py to produce *rou_path* for the given *period*.
    Silently skips if the file already exists (cache hit).

    Returns True on success, False on failure.
    """
    if os.path.exists(rou_path):
        print(f"    [cache]  {os.path.basename(rou_path)}")
        return True

    rand_trips = _find_random_trips()
    if rand_trips is None:
        print("ERROR: randomTrips.py not found. Set $SUMO_HOME or install SUMO.")
        return False

    cmd = [
        sys.executable, rand_trips,
        "-n", net_path,
        "-o", rou_path,
        "--period",  str(period),
        "--begin",   "0",
        "--end",     str(end),
        "--validate",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: randomTrips.py failed for period={period}")
        print(result.stderr[-2000:])
        return False

    print(f"    [gen]    {os.path.basename(rou_path)}")
    return True


def _make_sumocfg(
    cfg_path: str,
    net_path: str,
    rou_path: str,
    end: int = 7200,
):
    """Write a minimal .sumocfg pointing at *net_path* and *rou_path*."""
    root = ET.Element("configuration")
    inp  = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file",    value=net_path)
    ET.SubElement(inp, "route-files", value=rou_path)
    tim  = ET.SubElement(root, "time")
    ET.SubElement(tim, "begin", value="0")
    ET.SubElement(tim, "end",   value=str(end))
    raw    = ET.tostring(root, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="    ")
    with open(cfg_path, "w") as f:
        f.write(pretty)


# ---------------------------------------------------------------------------
# Result parsers
# ---------------------------------------------------------------------------

def _read_metrics_csv(run_dir: str) -> dict:
    """
    Parse network_metrics.csv in *run_dir*.

    Returns a dict with:
        mean_speed_ms, mean_speed_kmh, mean_throughput, mean_speed_ratio
    (any unavailable field is None).
    """
    csv_path = os.path.join(run_dir, "network_metrics.csv")
    if not os.path.exists(csv_path):
        return {
            "mean_speed_ms":    None,
            "mean_speed_kmh":   None,
            "mean_throughput":  None,
            "mean_speed_ratio": None,
        }

    speeds_ms, speeds_kmh, throughputs, ratios = [], [], [], []

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                speeds_ms.append(float(row["mean_speed_ms"]))
            except (KeyError, ValueError):
                pass
            try:
                speeds_kmh.append(float(row["mean_speed_kmh"]))
            except (KeyError, ValueError):
                pass
            try:
                throughputs.append(float(row["throughput_per_hour"]))
            except (KeyError, ValueError):
                pass
            try:
                ratios.append(float(row["speed_ratio"]))
            except (KeyError, ValueError):
                pass

    def _mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    return {
        "mean_speed_ms":    _mean(speeds_ms),
        "mean_speed_kmh":   _mean(speeds_kmh),
        "mean_throughput":  _mean(throughputs),
        "mean_speed_ratio": _mean(ratios),
    }


def _read_ai_json(run_dir: str) -> dict:
    """
    Parse antifragility_index.json in *run_dir*.

    Returns a dict with: ai, ci_low, ci_high, n_events_measured.
    """
    ai_path = os.path.join(run_dir, "antifragility_index.json")
    if not os.path.exists(ai_path):
        return {"ai": None, "ci_low": None, "ci_high": None, "n_events_measured": 0}

    with open(ai_path) as f:
        data = json.load(f)

    return {
        "ai":                data.get("antifragility_index"),
        "ci_low":            data.get("ci_95_low"),
        "ci_high":           data.get("ci_95_high"),
        "n_events_measured": data.get("n_events_measured", 0),
    }


def _read_accidents(run_dir: str) -> int:
    """Return total accident count from metadata.json."""
    meta_path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return 0
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("summary", {}).get("total_accidents", 0)
    except (json.JSONDecodeError, OSError):
        return 0


# ---------------------------------------------------------------------------
# ETA helper
# ---------------------------------------------------------------------------

def _eta_str(t_start: float, done: int, total: int) -> str:
    """Return a concise ETA string, e.g. '(ETA ~18 min)'."""
    if done == 0:
        return ""
    elapsed   = time.monotonic() - t_start
    per_run   = elapsed / done
    remaining = per_run * (total - done)
    if remaining < 90:
        return f"(ETA ~{remaining:.0f}s)"
    return f"(ETA ~{remaining / 60:.0f} min)"


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def run_sweep(
    base_config_path: str,
    periods: list[float],
    probs: list[float],
    n_seeds: int,
    sweep_dir: str,
    runner_script: str,
):
    """
    Execute the full (period × prob × seed) parameter grid.

    Writes one CSV row per completed cell immediately so partial sweeps are
    recoverable — re-running will append to (not overwrite) the CSV.
    """
    os.makedirs(sweep_dir, exist_ok=True)
    runs_dir = os.path.join(sweep_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    csv_path = os.path.join(sweep_dir, "sweep_results.csv")
    log_path = os.path.join(sweep_dir, "sweep_log.txt")

    # ── Load base config ───────────────────────────────────────────────────
    with open(base_config_path) as f:
        base_cfg = yaml.safe_load(f)

    current_cfg_file = base_cfg["sumo"]["config_file"]
    net_dir  = os.path.dirname(current_cfg_file)
    net_path = os.path.join(net_dir, "sioux_falls.net.xml")
    end_time = base_cfg["sumo"].get("total_steps", 7200)

    # ── Phase 0 : pre-generate route files ────────────────────────────────
    print(f"\n{'='*60}")
    print("  Phase 0 — pre-generating route files")
    print(f"{'='*60}")

    cfg_cache: dict[float, str] = {}   # period → .sumocfg path

    for period in sorted(set(periods)):
        tag      = f"period_{period:.3f}".replace(".", "p")
        rou_path = os.path.join(net_dir, f"sioux_falls_{tag}.rou.xml")
        cfg_path = os.path.join(net_dir, f"sioux_falls_{tag}.sumocfg")

        ok = _generate_routes(net_path, rou_path, period, end=end_time)
        if not ok:
            print(f"    [skip]   period={period} — route generation failed")
            continue

        _make_sumocfg(cfg_path, net_path, rou_path, end=end_time)
        cfg_cache[period] = cfg_path

    # Remove periods whose route generation failed
    periods = [p for p in periods if p in cfg_cache]
    if not periods:
        sys.exit("ERROR: No valid periods — exiting.")

    # ── Phase 1 : run the grid ─────────────────────────────────────────────
    total_runs = len(periods) * len(probs) * n_seeds
    done       = 0
    t_start    = time.monotonic()

    print(f"\n{'='*60}")
    print(f"  Phase 1 — running {total_runs} simulations")
    print(f"  Grid : {len(periods)} periods × {len(probs)} probs × {n_seeds} seeds")
    print(f"{'='*60}\n")

    # Open CSV (append mode — safe to resume a partial sweep)
    csv_exists = os.path.exists(csv_path)
    csv_fh     = open(csv_path, "a", newline="")
    writer     = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
    if not csv_exists:
        writer.writeheader()
        csv_fh.flush()

    log_fh = open(log_path, "a")
    log_fh.write(
        f"\n\n{'='*60}\n"
        f"  Sweep started  {datetime.datetime.now()}\n"
        f"  Grid: {len(periods)} periods × {len(probs)} probs × {n_seeds} seeds\n"
        f"{'='*60}\n"
    )

    try:
        for period in periods:
            for prob in probs:
                for seed_offset in range(n_seeds):
                    seed  = BASE_SEED + seed_offset
                    done += 1

                    eta   = _eta_str(t_start, done - 1, total_runs)
                    label = (
                        f"[{done:3d}/{total_runs}]  "
                        f"period={period:4.2f}s  "
                        f"prob={prob:.1e}  "
                        f"seed={seed}"
                    )
                    print(f"{label}  {eta}", flush=True)

                    # ── Per-cell output directory ──────────────────────────
                    prob_tag = f"{prob:.1e}".replace("+", "").replace("-", "m")
                    cell_tag = (
                        f"p{period:.3f}_q{prob_tag}_s{seed}"
                        .replace(".", "p")
                    )
                    run_dir = os.path.join(runs_dir, cell_tag)
                    os.makedirs(run_dir, exist_ok=True)

                    # ── Per-cell config ────────────────────────────────────
                    cell_cfg = copy.deepcopy(base_cfg)
                    cell_cfg["sumo"]["config_file"]      = cfg_cache[period]
                    cell_cfg["sumo"]["seed"]             = seed
                    # prob=0.0 → tiny-but-valid probability so validate_config
                    # doesn't reject it; 1e-9 produces effectively zero accidents.
                    cell_cfg["risk"]["base_probability"] = max(prob, 1e-9)
                    cell_cfg["output"]["output_folder"]  = run_dir

                    cell_cfg_path = os.path.join(run_dir, "cell_config.yaml")
                    with open(cell_cfg_path, "w") as f:
                        yaml.dump(
                            cell_cfg, f,
                            default_flow_style=False, sort_keys=False,
                        )

                    # ── Run simulation ─────────────────────────────────────
                    t0  = time.monotonic()
                    cmd = [sys.executable, runner_script, "--config", cell_cfg_path]
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        cwd=os.path.dirname(os.path.abspath(runner_script)),
                    )
                    elapsed = time.monotonic() - t0

                    # ── Log output ─────────────────────────────────────────
                    log_fh.write(f"\n{'─'*60}\n{label}\n{'─'*60}\n")
                    if result.stdout:
                        log_fh.write(result.stdout[-4000:])
                    if result.returncode != 0:
                        log_fh.write(
                            f"\n[STDERR rc={result.returncode}]\n"
                            f"{result.stderr[-2000:]}\n"
                        )
                        print(f"    !! FAILED (rc={result.returncode})")
                    log_fh.flush()

                    # ── Parse results ──────────────────────────────────────
                    metrics_data = _read_metrics_csv(run_dir)
                    ai_data      = _read_ai_json(run_dir)
                    n_acc        = _read_accidents(run_dir)

                    # ── Write CSV row ──────────────────────────────────────
                    row = {
                        "period":            period,
                        "prob":              prob,
                        "seed":              seed,
                        "n_accidents":       n_acc,
                        "mean_speed_ms":     metrics_data["mean_speed_ms"],
                        "mean_speed_kmh":    metrics_data["mean_speed_kmh"],
                        "mean_throughput":   metrics_data["mean_throughput"],
                        "mean_speed_ratio":  metrics_data["mean_speed_ratio"],
                        "ai":                ai_data["ai"],
                        "ci_low":            ai_data["ci_low"],
                        "ci_high":           ai_data["ci_high"],
                        "n_events_measured": ai_data["n_events_measured"],
                        "run_dir":           run_dir,
                    }
                    writer.writerow(row)
                    csv_fh.flush()

                    # ── Progress line ──────────────────────────────────────
                    spd_str = (
                        f"{metrics_data['mean_speed_kmh']:.1f} km/h"
                        if metrics_data["mean_speed_kmh"] is not None else "N/A"
                    )
                    ai_str = (
                        f"{ai_data['ai']:+.4f}"
                        if ai_data["ai"] is not None else "N/A"
                    )
                    print(
                        f"    ↳ {elapsed:.0f}s  "
                        f"acc={n_acc}  "
                        f"speed={spd_str}  "
                        f"AI={ai_str}"
                    )

    finally:
        csv_fh.close()
        log_fh.close()

    # ── Summary ────────────────────────────────────────────────────────────
    total_elapsed = time.monotonic() - t_start
    print(f"\n{'='*60}")
    print(f"  Sweep complete — {done} runs in {total_elapsed / 60:.1f} min")
    print(f"  Results → {csv_path}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))

    ap = argparse.ArgumentParser(
        description="SAS failure-point parameter grid sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--config",
        default=os.path.join(here, "config.yaml"),
        help="Base config.yaml",
    )
    ap.add_argument(
        "--runner",
        default=os.path.join(here, "runner.py"),
        help="Path to runner.py",
    )
    ap.add_argument(
        "--out-dir",
        default=os.path.join(here, "results", "sweep"),
        help="Output directory for sweep results",
    )
    ap.add_argument(
        "--periods", type=float, nargs="+", default=DEFAULT_PERIODS,
        metavar="P",
        help="Vehicle insertion periods (s); lower = denser traffic",
    )
    ap.add_argument(
        "--probs", type=float, nargs="+", default=DEFAULT_PROBS,
        metavar="Q",
        help="base_probability values (0.0 = no-accident baseline)",
    )
    ap.add_argument(
        "--seeds", type=int, default=DEFAULT_SEEDS,
        metavar="N",
        help="Stochastic replicates per (period, prob) cell",
    )
    args = ap.parse_args()

    total = len(args.periods) * len(args.probs) * args.seeds
    est_min = total * 45 // 60

    print(f"\n  SAS — Failure-Point Sweep")
    print(f"  {'─'*40}")
    print(f"  Config   : {args.config}")
    print(f"  Runner   : {args.runner}")
    print(f"  Out dir  : {args.out_dir}")
    print(f"  Periods  : {args.periods}")
    print(f"  Probs    : {[f'{p:.1e}' for p in args.probs]}")
    print(f"  Seeds    : {args.seeds}")
    print(f"  Total    : {total} runs  (~{est_min} min estimate)\n")

    run_sweep(
        base_config_path = args.config,
        periods          = args.periods,
        probs            = args.probs,
        n_seeds          = args.seeds,
        sweep_dir        = args.out_dir,
        runner_script    = args.runner,
    )


if __name__ == "__main__":
    main()
