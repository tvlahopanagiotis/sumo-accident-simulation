# SUMO Accident Simulation (SAS)

[![CI](https://github.com/tvlahopanagiotis/sumo-accident-simulation/actions/workflows/ci.yml/badge.svg)](https://github.com/tvlahopanagiotis/sumo-accident-simulation/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A scientifically-calibrated, probabilistic traffic accident simulator built on top of
[SUMO (Eclipse Simulation of Urban MObility)](https://sumo.dlr.de/).

SUMO is a powerful open-source traffic simulator, but it has no native concept of
accidents — vehicles are designed to avoid collisions. **SAS adds the missing layer:**
a multi-component risk model triggers accidents probabilistically, manages their full
lifecycle (blockage → emergency response → clearance → full recovery), and measures
how the network responds, including computing an **Antifragility Index** that
quantifies whether the network adapts and improves after repeated disruptions.

> **Two branches, one codebase**
> - `main` — standalone accident simulator; production code only (this README)
> - `antifragility` — adds `experiment_sweep.py` + three visualisation scripts for
>   systematically studying how the AI changes across hundreds of parameter
>   combinations; see its own README for details

---

## Context: AntifragiCity Project

SAS is the **scenario-testing and measurement engine** for the
[AntifragiCity](https://github.com/tvlahopanagiotis/sumo-accident-simulation)
project, which aims to:

1. Build an **ontology** of endemic and black-swan disruptions and their effects on
   urban mobility (accidents, weather events, demand surges, infrastructure failures, …)
2. Define **mobility triage** — a suite of short-term interventions that minimise
   the immediate impact of each disruption type
3. Propose **long-term solutions** that increase network resilience and, where
   possible, introduce antifragility
4. Deliver all of the above as a **SUMO API / Python package** with one-click
   resilience assessment, weak-point identification, and scenario testing

SAS currently covers goal 4 for one disruption type (traffic accidents). The
**Antifragility Index** computed at the end of each run is a standard output
metric — it answers *"how did this network respond to the disruptions in this
run?"* The `antifragility` branch adds the research tooling to study that
question systematically across a load × accident-probability parameter grid.

---

## Key Features

| Feature | Detail |
|---------|--------|
| **Four severity tiers** | MINOR / MODERATE / MAJOR / CRITICAL, weighted from NHTSA KABCO injury-classification data (62 / 28 / 8 / 2 %) |
| **Nilsson Power Model** | Speed risk ∝ (v / v_road_limit)^k — uses the road's posted speed limit, not the vehicle's mechanical maximum |
| **Speed-variance risk** | Captures rear-end risk from differential braking events |
| **Density risk** | Bell-curve model peaking at configurable jam-density fraction |
| **Road-type multipliers** | Highway, arterial, local, intersection — derived from SUMO speed limits |
| **Three-phase lifecycle** | ACTIVE → CLEARING (linear capacity ramp) → RESOLVED |
| **Antifragility Index** | Pre/post-disruption speed ratio, computed per event with bootstrapped 95 % CI |
| **Batch runs** | `--runs N` launches N independent seeds; aggregate statistics exported automatically |
| **Zero hot-loop TraCI calls** | BatchSubscription architecture: 2 TraCI calls/step vs ~15 000 before optimisation |

---

## System Architecture

```
config.yaml                ← single control file; every parameter documented inline
     │
     ▼
runner.py                  ← orchestrates the simulation loop
     │
     ├── risk_model.py     ← probabilistic risk calculator (per vehicle, per step)
     │       ├── 1. Speed risk        Nilsson Power Model: (v/v_limit)^k
     │       ├── 2. Speed-variance    differential braking proxy
     │       ├── 3. Density risk      Gaussian bell curve around peak density
     │       └── 4. Road-type mult.   highway / arterial / local / intersection
     │
     ├── accident_manager.py  ← accident lifecycle
     │       ├── Severity sampling    weighted draw: MINOR 62% MODERATE 28%
     │       │                        MAJOR 8%  CRITICAL 2%
     │       ├── ACTIVE phase         vehicle frozen, lane speed reduced
     │       ├── CLEARING phase       linear speed ramp after response time
     │       └── RESOLVED phase       full lane recovery, vehicle released
     │
     └── metrics.py        ← scientific output
             ├── network_metrics.csv        step-by-step network state
             ├── vehicle_snapshots.csv      per-vehicle speed/position records
             ├── accident_reports.json      per-accident impact summary
             └── antifragility_index.json   AI with 95 % CI
```

---

## Installation

### 1 — Install SUMO

**macOS (Homebrew):**
```bash
brew install sumo
```

**Ubuntu / Debian:**
```bash
sudo apt-get install sumo sumo-tools sumo-doc
```

**Windows:** download the installer from https://sumo.dlr.de/docs/Downloads.php

Verify:
```bash
sumo --version
```

### 2 — Set SUMO_HOME

**macOS / Linux** — add to `~/.bashrc` or `~/.zshrc`:
```bash
export SUMO_HOME="/opt/homebrew/share/sumo"   # macOS Homebrew
# export SUMO_HOME="/usr/share/sumo"          # Linux
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```
Then `source ~/.bashrc`.

**Windows** — add to System Environment Variables:
```
SUMO_HOME = C:\Program Files (x86)\Eclipse\Sumo
PYTHONPATH = %SUMO_HOME%\tools
```

### 3 — Python dependencies

```bash
# Install all dependencies (from pyproject.toml)
pip install .

# Or for development (includes pytest, mypy, ruff)
pip install -e ".[dev]"
```

> **Note:** `traci` ships with SUMO — no separate pip install needed once `PYTHONPATH` is set.

---

## Quick Start

> **Prerequisites:** complete the [Installation](#installation) steps first
> (install SUMO, set `SUMO_HOME`, then run `pip install .`).

```bash
git clone https://github.com/tvlahopanagiotis/sumo-accident-simulation.git
cd sumo-accident-simulation
pip install .
```

### Single run (uses seed from config.yaml)
```bash
python runner.py
```

### Multiple independent runs (different seeds, results aggregated)
```bash
python runner.py --runs 10
```

### Custom config file
```bash
python runner.py --config experiments/high_risk.yaml --runs 5
```

### Visual run (opens SUMO GUI)
In `config.yaml` set `sumo.binary: sumo-gui`, then run as normal.

---

## Bundled Networks

Three network generators ship with SAS. Each downloads or builds the required
SUMO files and can auto-patch `config.yaml` so `runner.py` is ready to go
immediately:

| Script | Network | Scale | Simultaneous vehicles |
|--------|---------|-------|-----------------------|
| `generate_thessaloniki.py` | Thessaloniki city centre (real OSM) | Urban, ~7 km² | ~300–500 |
| `generate_sioux_falls.py` | Sioux Falls benchmark (synthetic) | 24-node grid | ~150 |
| `generate_network.py` | Riverside District (synthetic) | Small urban | ~80 |

### Thessaloniki — real city, recommended for full-city runs

Downloads OSM data from OpenStreetMap, converts it with SUMO's `netconvert`,
and generates randomised vehicle routes:

```bash
python generate_thessaloniki.py --update-config
python runner.py
```

`--update-config` patches `config.yaml` automatically. Add `--gui` to also
open SUMO's graphical interface while the simulation runs.

```bash
# all options
python generate_thessaloniki.py --help

# re-run without re-downloading the OSM file
python generate_thessaloniki.py --update-config --skip-download

# denser traffic (lower period → more vehicles inserted per second)
python generate_thessaloniki.py --update-config --period 0.5
```

### Sioux Falls — classic benchmark

Classic 24-node transport benchmark. Useful for reproducing published results
and quick experiments (~30 s to generate):

```bash
python generate_sioux_falls.py --update-config
python runner.py
```

### Riverside District — small synthetic network

Fastest to generate; ideal for development and testing the pipeline end-to-end:

```bash
python generate_network.py --update-config
python runner.py
```

---

## Configuration

`config.yaml` is the single control point. Every parameter is documented inline.
The most important settings for a new user:

```yaml
sumo:
  config_file: /path/to/your/network.sumocfg   # ← only mandatory change
  total_steps: 7200    # simulated seconds (7200 = 2 hours)
  seed: 42

risk:
  # Global accident-rate scaler. Scales with simultaneous vehicle count.
  # See the calibration table below for per-network recommendations.
  # Rough targets for ~150 simultaneous vehicles (Sioux Falls baseline):
  #   ~2  accidents / 2 h → 5.0e-05
  #   ~5  accidents / 2 h → 1.5e-04   ← Sioux Falls default
  #   ~15 accidents / 2 h → 5.0e-04
  base_probability: 1.5e-04

  # Nilsson Power Model exponent:
  #   2.0 = property-damage-only  ← default
  #   3.0 = injury crashes
  #   4.0 = fatal crashes
  speed_exponent: 2.0

accident:
  max_concurrent_accidents: 2

  severity:
    minor:      { weight: 62, duration_min_s: 120,  duration_max_s: 900,   lane_capacity_fraction: 0.70, ... }
    moderate:   { weight: 28, duration_min_s: 900,  duration_max_s: 2700,  lane_capacity_fraction: 0.40, ... }
    major:      { weight:  8, duration_min_s: 2700, duration_max_s: 7200,  lane_capacity_fraction: 0.10, ... }
    critical:   { weight:  2, duration_min_s: 3600, duration_max_s: 18000, lane_capacity_fraction: 0.00, ... }
```

See `config.yaml` for the full parameter reference with inline explanations.

### Calibrating `base_probability` for your network

`base_probability` is the most important tuning knob. It scales with the
**number of simultaneous vehicles**, not the total trip count. Use these
tested values as a starting point for a target of ~5 accidents per 2-hour run:

| Network | Simultaneous vehicles | Recommended `base_probability` |
|---------|-----------------------|-------------------------------|
| Riverside District | ~80 | `2.0e-04` |
| Sioux Falls | ~150 | `1.5e-04` |
| Thessaloniki city centre | ~500 | `5.0e-05` |
| Large city (~2 000 vehicles) | ~2 000 | `1.5e-05` |

**Rule of thumb:** if you double the simultaneous vehicle count, halve
`base_probability`. Run with `--log-level INFO`, watch the accident count in
the terminal, and adjust until you see your target rate.

---

## Output Files

All output is written to `results/` (configurable via `output.output_folder`).

### `network_metrics.csv`

Step-by-step network state:

| Column | Meaning |
|--------|---------|
| `step` | Simulation time (seconds) |
| `mean_speed_ms` | Mean vehicle speed (m/s) |
| `mean_speed_kmh` | Mean vehicle speed (km/h) |
| `throughput_per_hour` | Vehicles completing trips per hour |
| `mean_delay_seconds` | Delay vs free-flow baseline |
| `active_accidents` | Number of accidents currently active |
| `speed_ratio` | 1.0 = free-flow; < 1.0 = degraded |

### `accident_reports.json`

One entry per accident, e.g.:
```json
{
  "accident_id": "ACC_0003",
  "severity": "MODERATE",
  "trigger_step": 1820,
  "resolved_step": 3240,
  "duration_s": 1420,
  "lane_capacity_fraction": 0.40,
  "location": { "edge_id": "21", "lane_id": "21_0", "x": 1842.3, "y": 967.1 },
  "impact": { "peak_queue_length": 18, "vehicles_affected_count": 31 }
}
```

### `antifragility_index.json`

```json
{
  "antifragility_index": 0.041,
  "interpretation": "ANTIFRAGILE — network performance improved post-disruption",
  "ci_95_low": 0.012,
  "ci_95_high": 0.070,
  "n_events_measured": 6
}
```

| AI value | Interpretation |
|----------|---------------|
| > +0.05 | **Antifragile** — network adapted and improved after disruptions |
| −0.05 to +0.05 | **Resilient** — returned to baseline |
| −0.20 to −0.05 | **Fragile** — lingering degradation |
| < −0.20 | **Brittle** — severe lasting damage |

### `metadata.json`

Machine-readable record of every config parameter and summary statistic, suitable
for reproducibility auditing and experiment tracking.

---

## The Risk Model

### Composite risk score

```
risk_score  =  speed_weight      × speed_risk
            +  variance_weight   × variance_risk
            +  density_weight    × density_risk

final_risk  =  clamp(risk_score × road_type_multiplier, 0, 1)
```

Default weights sum to 1.0 (0.40 + 0.30 + 0.30).

### 1. Speed risk — Nilsson Power Model

Risk scales as a power law of the speed ratio relative to the **road posted speed
limit** (not the vehicle's mechanical maximum):

```
speed_risk = (v / v_road_limit) ^ speed_exponent
```

This implements Nilsson (1981), confirmed by Elvik (2009):

| Exponent | Crash type |
|----------|-----------|
| 2.0 | Property-damage-only (default) |
| 3.0 | Injury crashes |
| 4.0 | Fatal crashes |

Using the road speed limit (fetched once per unique edge via TraCI and cached)
means that a vehicle doing 50 km/h on a 50 km/h road scores the same regardless
of whether it is a bus (max 80) or a sports car (max 250).

### 2. Speed-variance risk

```
variance_risk = |v_ego − mean(v_neighbours)| / speed_variance_threshold
```

Captures rear-end collision risk from differential braking. Saturates to 1.0 at
`speed_variance_threshold_ms` (default 5 m/s ≈ 18 km/h differential).
Neighbours are collected within `neighbor_radius_m` (default 150 m).

### 3. Density risk

Bell-curve shaped, peaking at `peak_density_vehicles_per_km`:

```
density_risk = exp( −(density − peak_density)² / (2 σ²) )   where σ = peak_density × 0.5
```

Risk is highest at medium-high density — the most accident-prone regime.

### 4. Road type multiplier

Derived from the edge's speed limit (standard proxy for road functional class):

| Speed limit | Road type | Default multiplier |
|-------------|-----------|-------------------|
| ≥ 90 km/h | Highway | 1.5 |
| ≥ 50 km/h | Arterial | 1.0 (reference) |
| < 50 km/h | Local | 0.6 |
| Junction edge | Intersection | 2.0 |

### Triggering

An accident fires only if two conditions are both met:
1. `final_risk > trigger_threshold` (default 0.35) — noise gate
2. A random draw succeeds with probability
   `base_probability × (1 + excess_risk × 10) × secondary_multiplier`

The secondary multiplier is > 1.0 when the vehicle is within the secondary-risk
radius of an existing active accident (rubbernecking / debris zone).

---

## The Severity Model

Each accident is randomly assigned a severity tier at trigger time using
`random.choices()` with the configured weights:

| Tier | Weight | Freq. | Capacity remaining | Response time | Basis |
|------|--------|-------|--------------------|--------------|-------|
| MINOR | 62 | ≈ 62 % | 70 % | 5 min | PDO — Property Damage Only |
| MODERATE | 28 | ≈ 28 % | 40 % | 10 min | Non-incapacitating injury |
| MAJOR | 8 | ≈ 8 % | 10 % | 20 min | Incapacitating injury |
| CRITICAL | 2 | ≈ 2 % | 0 % | 30 min | Fatal — forensic scene |

Weights are calibrated to NHTSA Traffic Safety Facts 2022 (KABCO classification).

Duration is drawn from a **log-normal distribution** clipped to each tier's
`[duration_min_s, duration_max_s]` range, with the geometric mean as the
log-normal mean and σ = 0.5. This gives a realistic right-skewed distribution —
most incidents clear quickly, rare ones take far longer.

After `response_time_s` seconds, the scene transitions to CLEARING and lane
speed ramps back linearly to the original limit over the remaining duration.

---

## Reproducing Results

Each run is seeded (`--seed` in config or `--runs N` increments from the base
seed). `metadata.json` records the full config alongside the summary statistics,
so any run can be exactly reproduced given the same SUMO version.

```bash
# Re-run a specific seed
python runner.py --config results/run_0047/metadata.json  # not yet implemented
python runner.py                                           # use config.yaml seed
```

---

## References

- Nilsson, G. (1981). *The Effects of Speed Limits on Traffic Accidents in Sweden.*
  VTI Rapport 68. Swedish Road and Traffic Research Institute.
- Elvik, R. (2009). The Power Model of the Relationship Between Speed and Road
  Safety. *TØI Report 1034/2009.* Institute of Transport Economics, Oslo.
- NHTSA (2022). *Traffic Safety Facts 2022.* DOT HS 813 560.
  National Highway Traffic Safety Administration.
- Lopez, P.A., et al. (2018). Microscopic Traffic Simulation using SUMO.
  *IEEE ITSC 2018.* https://doi.org/10.1109/ITSC.2018.8569938

---

## Citation

If you use SAS in academic or professional work, please cite:

```bibtex
@software{vlachopanagiotis2026sas,
  author      = {Vlachopanagiotis, Theocharis},
  title       = {{SUMO Accident Simulation (SAS)}: A Probabilistic Traffic
                 Accident Extension for {SUMO}},
  year        = {2026},
  institution = {Rho\'{e} / AntifragiCity Project},
  url         = {https://github.com/tvlahopanagiotis/sumo-accident-simulation},
  note        = {Implements the Nilsson (1981) Power Model with four-tier
                 NHTSA-calibrated severity classification}
}
```

---

## Development

### Setup

```bash
git clone https://github.com/tvlahopanagiotis/sumo-accident-simulation.git
cd sumo-accident-simulation
pip install -e ".[dev]"
```

### Running tests

The test suite contains **55 unit tests** across four modules. All unit tests
run without SUMO installed (TraCI is mocked at the session level in
`tests/conftest.py`):

```bash
# All unit tests (no SUMO required)
pytest tests/ -v

# With coverage report
pytest tests/ --cov=. --cov-report=term-missing

# Integration tests (requires a running SUMO installation)
pytest tests/ -m integration
```

### Code quality

```bash
# Linting
ruff check .

# Type checking
mypy runner.py risk_model.py accident_manager.py metrics.py
```

---

## Roadmap

- [ ] Weather overlay (wet/icy road multipliers configurable per step)
- [ ] Multi-vehicle pile-up propagation model
- [ ] Variable demand profiles (AM/PM peak, school runs)
- [ ] REST API for live scenario injection
- [ ] Direct integration with SUMO's `--device.rerouting.probability`
