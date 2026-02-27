# SUMO Accident Simulation (SAS)

A probabilistic traffic accident simulation extension for [SUMO](https://sumo.dlr.de/), built for the AntifragiCity project and the broader scientific community.

---

## What does this do?

SUMO is a powerful traffic simulator, but it has no concept of accidents. Vehicles are programmed to avoid collisions. **SAS adds the missing layer**: it uses a risk model to probabilistically trigger accidents, manages their full lifecycle (blockage → emergency response → clearance → recovery), and measures how the network responds — including an **Antifragility Index** to assess whether the network actually adapts after disruptions.

---

## System Architecture

```
config.yaml          ← You control everything here
     │
     ▼
runner.py            ← Main loop: TraCI ↔ SUMO
     ├── risk_model.py        Risk score per vehicle per timestep
     │       ├── Speed risk
     │       ├── Speed variance risk
     │       ├── Density risk (bell curve)
     │       └── Road type multiplier
     │
     ├── accident_manager.py  Accident lifecycle
     │       ├── ACTIVE:    Vehicle stopped, lane capacity reduced
     │       ├── CLEARING:  Gradual capacity restoration
     │       └── RESOLVED:  Full recovery, vehicle released
     │
     └── metrics.py           Scientific output
             ├── network_metrics.csv     (timestep-by-timestep)
             ├── accident_reports.json   (per-accident data)
             └── antifragility_index.json
```

---

## Setup (Step by Step for Beginners)

### Step 1: Install SUMO

**On macOS:**
```bash
brew install sumo
```

**On Ubuntu/Debian:**
```bash
sudo apt-get install sumo sumo-tools sumo-doc
```

**On Windows:**
Download the installer from https://sumo.dlr.de/docs/Downloads.php

After installing, verify it works:
```bash
sumo --version
```

### Step 2: Set SUMO_HOME (needed so Python can find TraCI)

**macOS/Linux** — add to your `~/.bashrc` or `~/.zshrc`:
```bash
export SUMO_HOME="/usr/share/sumo"           # Linux
export SUMO_HOME="/opt/homebrew/share/sumo"  # macOS with Homebrew
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```

Then reload: `source ~/.bashrc`

**Windows** — add to Environment Variables:
```
SUMO_HOME = C:\Program Files (x86)\Eclipse\Sumo
```
And add `%SUMO_HOME%\tools` to your `PYTHONPATH`.

### Step 3: Install Python dependencies

```bash
pip install pyyaml
```

(TraCI comes bundled with SUMO — no separate install needed once PYTHONPATH is set.)

### Step 4: Clone/copy this project

Put the `sumo_accident_sim/` folder anywhere you like.

---

## Running Your First Simulation

### Option A: Use a real SUMO network

If you already have a SUMO `.sumocfg` file:

1. Edit `config.yaml` and point `sumo.config_file` to your `.sumocfg`
2. Run:
```bash
cd sumo_accident_sim/
python sas/runner.py
```

### Option B: Test with SUMO's built-in example networks

SUMO comes with example networks. Try this one:
```bash
# Find your examples folder
ls $SUMO_HOME/docs/examples/sumo/

# Edit config.yaml:
# sumo.config_file: "/usr/share/sumo/docs/examples/sumo/busses/test.sumocfg"
```

### Option C: Use sumo-gui to see it visually

In `config.yaml`, change:
```yaml
sumo:
  binary: "sumo-gui"   # ← this line
```

---

## Understanding the Output

After the simulation, check the `results/` folder:

### `network_metrics.csv`
A timestep-by-timestep record of network performance:
| Column | Meaning |
|--------|---------|
| `step` | Simulation timestep (seconds) |
| `mean_speed_kmh` | Average speed of all vehicles |
| `throughput_per_hour` | Vehicles completing routes |
| `mean_delay_seconds` | Delay compared to free-flow |
| `active_accidents` | How many accidents are active right now |
| `speed_ratio` | 1.0 = normal, 0.5 = half speed (big disruption) |

### `accident_reports.json`
One entry per accident:
```json
{
  "accident_id": "ACC_0001",
  "trigger_step": 1240,
  "resolved_step": 1843,
  "duration_seconds": 603,
  "location": {"edge_id": "E14", "x": 312.4, "y": 89.1},
  "impact": {"peak_queue_length_vehicles": 23, "vehicles_affected": 41}
}
```

### `antifragility_index.json`
The key scientific output:
```json
{
  "antifragility_index": 0.032,
  "baseline_speed_kmh": 38.4,
  "post_disruption_speed_kmh": 39.6,
  "interpretation": "ANTIFRAGILE — Network performance improved after disruptions",
  "total_accidents": 7
}
```

**Interpreting the Antifragility Index:**
- `> 0.05` → Antifragile (network adapted and improved — vehicles found better routes)
- `-0.05 to 0.05` → Resilient (returned to baseline)
- `-0.20 to -0.05` → Fragile (lingering degradation)
- `< -0.20` → Brittle (severe lasting damage)

---

## Tuning the Risk Model

Everything is controlled via `config.yaml`. Key parameters to experiment with:

**To get more accidents:**
```yaml
risk:
  base_probability: 0.00005   # increase from 0.000005
  trigger_threshold: 0.5      # lower from 0.7
```

**To model a wet/rainy day:**
```yaml
risk:
  road_type_multipliers:
    highway: 2.5    # more dangerous in wet conditions
    arterial: 1.5
```

**To model faster emergency response:**
```yaml
accident:
  response_time_seconds: 120   # reduce from 300
```

---

## How the Risk Model Works

The risk score for each vehicle is calculated as:

```
risk_score = speed_weight × speed_risk
           + variance_weight × speed_variance_risk
           + density_weight × density_risk

final_risk = risk_score × road_type_multiplier
```

- **Speed risk**: quadratic — double the speed = 4× the risk
- **Speed variance**: if your vehicle is going 80 km/h and the car ahead is at 20 km/h, risk spikes
- **Density risk**: bell-curve shaped — peaks at medium-high density (~40 veh/km), drops in both gridlock and empty roads
- **Road type**: intersections are 2× more dangerous than baseline

An accident triggers only when `risk_score > trigger_threshold` AND a random draw with probability `base_probability × excess_risk` succeeds.

---

## Citation

If you use SAS in academic work, please cite:

```
Theocharis [Surname], Rhoé Mobility Consultancy.
"SUMO Accident Simulation (SAS): A Probabilistic Traffic Accident
Extension for Antifragility Research." AntifragiCity Project, 2026.
```

---

## Roadmap

- [ ] Weather condition overlay (rain, ice multipliers)
- [ ] Multi-vehicle pile-up modelling
- [ ] Integration with SUMO's actual demand data for peak-hour calibration
- [ ] REST API for live scenario injection
- [ ] Visualisation dashboard (network heatmaps, recovery curves)
