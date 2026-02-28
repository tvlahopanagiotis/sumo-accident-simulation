# SUMO Accident Simulation — Antifragility Research Branch

> **Branch:** `antifragility` — antifragility analysis layer built on the core simulator
> **Core simulator:** see [`main` branch](../../tree/main) and its README

This branch extends the core SAS simulator with tools to study **network
antifragility**: the capacity of a traffic network to not merely recover from
disruptions (resilience) but to actually *improve* its performance because of them.
The concept was formalised by Taleb (2012); this project operationalises it for
urban traffic networks using SUMO.

---

## What Is Antifragility in Traffic?

A **fragile** network degrades under accidents and takes a long time to recover.
A **resilient** network recovers to its baseline.
An **antifragile** network exploits disruptions — drivers learn faster routes,
platoons dissolve, natural filtering of over-loaded corridors occurs — and ends
up performing *better* after the disruption than before it.

We measure this with the **Antifragility Index (AI)**:

```
AI = (mean_speed_post_accident − mean_speed_pre_accident) / mean_speed_pre_accident
```

Computed per accident event (over configurable pre/post windows) and averaged
with a 95 % bootstrapped confidence interval across all events in a run:

| AI | Regime |
|----|--------|
| > +0.05 | **Antifragile** — performance improved post-disruption |
| −0.05 to +0.05 | **Resilient** — returned to baseline |
| −0.20 to −0.05 | **Fragile** — lingering degradation |
| < −0.20 | **Brittle** — severe, lasting damage |

---

## Repository Structure

```
.
├── runner.py               Core simulation loop (from main)
├── risk_model.py           Probabilistic risk model (from main)
├── accident_manager.py     Accident lifecycle manager (from main)
├── metrics.py              Scientific output collector (from main)
├── config.yaml             Simulation configuration
│
├── experiment_sweep.py     ← Parameter sweep across traffic load × accident prob
├── visualise_sweep.py      ← Academic-style figures from sweep results
├── visualise_batch.py      ← Multi-run aggregate dashboards
│
└── results/                Output (generated, not committed)
    ├── sweep_results.csv   Sweep grid output (period × prob × seed)
    └── run_*/              Per-run output folders
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

**Windows:** https://sumo.dlr.de/docs/Downloads.php

### 2 — Set SUMO_HOME

```bash
# Add to ~/.bashrc or ~/.zshrc
export SUMO_HOME="/opt/homebrew/share/sumo"   # macOS Homebrew
# export SUMO_HOME="/usr/share/sumo"          # Linux
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
source ~/.bashrc
```

### 3 — Python dependencies

```bash
pip install pyyaml matplotlib numpy
```

---

## Running the Core Simulation

### Single run
```bash
python runner.py
```

### Batch of independent runs (seeds base_seed … base_seed + N − 1)
```bash
python runner.py --runs 10
```

### Custom config
```bash
python runner.py --config experiments/high_load.yaml --runs 5
```

Point `config.yaml → sumo.config_file` at your `.sumocfg` before running.

---

## The Parameter Sweep Experiment

`experiment_sweep.py` runs a full factorial grid across two key dimensions:

| Axis | Values (default) | Meaning |
|------|-----------------|---------|
| **Vehicle insertion period (s)** | 5.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5 | Controls traffic load (demand) |
| **Accident base probability** | 0, 5e-5, 1.5e-4, 5e-4, 1e-3 | Controls accident frequency |
| **Seeds per cell** | 2 | For variance estimation |

Total grid: 7 × 5 × 2 = **70 simulation runs**.

### Converting insertion period to demand

```
vehicles_per_hour  ≈  3600 / period_s
```

| Period (s) | Load (veh/h) | Regime |
|------------|-------------|--------|
| 5.0 | 720 | Light |
| 3.0 | 1200 | Moderate |
| 2.0 | 1800 | Heavy |
| 1.5 | 2400 | Near-capacity |
| 1.0 | 3600 | At-capacity |
| 0.75 | 4800 | Over-capacity |
| 0.5 | 7200 | Severe overload |

### Running the sweep

```bash
python experiment_sweep.py
```

Results are written row-by-row to `sweep_results.csv` as each cell completes,
so the sweep is safely resumable after interruption.

**Optional arguments:**
```bash
python experiment_sweep.py \
    --periods 5.0 3.0 2.0 1.5 1.0 \
    --probs   0 5e-5 1.5e-4 5e-4 \
    --seeds   3 \
    --out     my_sweep.csv
```

### Sweep output schema (`sweep_results.csv`)

| Column | Meaning |
|--------|---------|
| `period` | Vehicle insertion period (s) |
| `prob` | Accident base probability |
| `seed` | Random seed used |
| `n_accidents` | Total accidents triggered |
| `mean_speed_kmh` | Mean network speed (km/h) |
| `mean_speed_ratio` | Speed relative to free-flow baseline |
| `ai` | Antifragility Index (mean across events) |
| `ci_low`, `ci_high` | 95 % confidence interval on AI |
| `n_events_measured` | Number of events used for AI |

---

## Visualising Results

### Sweep figures (four panels)

```bash
python visualise_sweep.py sweep_results.csv
```

Produces four publication-ready figures in the same directory as the CSV:

| Figure | Content |
|--------|---------|
| `fig1_speed_vs_load.pdf` | Mean network speed vs traffic load, one curve per accident probability |
| `fig2_ai_vs_load.pdf` | Antifragility Index vs load with ±1 SE shaded bands and 95 % CI bars |
| `fig3_heatmaps.pdf` | 2-D heatmaps: speed and AI over the full (load × prob) grid |
| `fig4_phase_diagram.pdf` | Phase diagram (antifragile / resilient / fragile / brittle zones) with baseline strip |

All figures use the **Wong (2011) colourblind-safe palette** and IEEE/Nature
style (white background, explicit markers, no glow, dashed grid).

### Batch run dashboard (single probability, multiple seeds)

```bash
python visualise_batch.py results/
```

Produces aggregate speed, throughput, and AI time-series plots with shaded ±1 SD bands.

---

## Key Findings — Sioux Falls Network

The Sioux Falls benchmark network (24 nodes, 76 directed links) was used for all
experiments (LeBlanc 1975; Bar-Gera 2009).

### Network failure point

| Traffic load | Mean speed | Speed ratio | Regime |
|-------------|-----------|-------------|--------|
| 720 veh/h (period 5 s) | ≈ 33.9 km/h | 1.00 | Free-flow baseline |
| 1200 veh/h (period 3 s) | ≈ 32.1 km/h | 0.95 | Stable |
| 1800 veh/h (period 2 s) | ≈ 29.4 km/h | 0.87 | Moderate congestion |
| 2400 veh/h (period 1.5 s) | ≈ 27.0 km/h | 0.80 | High congestion |
| 3600 veh/h (period 1.0 s) | ≈ 22.0 km/h | 0.65 | **Failure boundary** |

The network crosses the **30 % degradation threshold** (speed ratio ≈ 0.70) at
approximately 3 600 veh/h — consistent with the macroscopic fundamental diagram
(MFD) capacity breakdown point for the Sioux Falls topology.

### Antifragility regime

- At **low to moderate load** (≤ 1 800 veh/h), the network exhibits **antifragile or
  resilient** behaviour: AI > 0 for most accident probability levels, indicating that
  disruptions prompt re-routing that reduces overall congestion.
- At **near-capacity load** (≥ 2 400 veh/h), the network transitions to **fragile**:
  re-routing options are exhausted, and accidents cause persistent degradation.
- **Accident probability** has a secondary effect: AI decreases monotonically with
  increasing base_probability at all load levels, consistent with a saturation model
  where too-frequent disruptions prevent full recovery between events.

---

## Theoretical Background

### Antifragility (Taleb 2012)

Taleb defines antifragility as the property of systems that benefit from volatility,
stressors, or disorder. In traffic, this manifests as drivers finding better routes
after an accident clears, or platoon dissolution reducing vehicle clustering.

### Nilsson Power Model (1981)

Speed risk in the accident triggering model follows:

```
accident_rate  ∝  (v / v_road_limit) ^ k
```

where `k = 2` for property-damage-only, `k = 3` for injury, and `k = 4` for fatal
crashes. Confirmed empirically by Elvik (2009) across 98 datasets.

### KABCO Severity Classification (NHTSA)

Accident severity tiers are weighted from the NHTSA 2022 KABCO classification:

| KABCO class | SAS tier | Weight |
|-------------|----------|--------|
| PDO (property damage only) | MINOR | 62 % |
| C (possible injury) | MODERATE | 28 % |
| A (incapacitating injury) | MAJOR | 8 % |
| K (fatal) | CRITICAL | 2 % |

---

## References

- Taleb, N.N. (2012). *Antifragile: Things That Gain from Disorder.* Random House.
- Nilsson, G. (1981). *The Effects of Speed Limits on Traffic Accidents in Sweden.*
  VTI Rapport 68. Swedish Road and Traffic Research Institute.
- Elvik, R. (2009). The Power Model of the Relationship Between Speed and Road
  Safety. *TØI Report 1034/2009.* Institute of Transport Economics, Oslo.
- NHTSA (2022). *Traffic Safety Facts 2022.* DOT HS 813 560.
- LeBlanc, L.J. (1975). An Algorithm for the Discrete Network Design Problem.
  *Transportation Science*, 9(3), 183–199.
- Bar-Gera, H. (2009). Sioux Falls transportation network dataset.
  https://github.com/bstabler/TransportationNetworks
- Wong, B. (2011). Points of view: Color blindness. *Nature Methods*, 8(6), 441.
- Lopez, P.A., et al. (2018). Microscopic Traffic Simulation using SUMO.
  *IEEE ITSC 2018.* https://doi.org/10.1109/ITSC.2018.8569938

---

## Citation

If you use this work in academic or professional publications, please cite:

```bibtex
@software{vlachopanagiotis2026sas,
  author    = {Vlachopanagiotis, Theocharis and Rho\'{e}},
  title     = {{SUMO Accident Simulation (SAS)}: A Probabilistic Traffic
               Accident Extension for {SUMO} with Antifragility Analysis},
  year      = {2026},
  url       = {https://github.com/tvlahopanagiotis/sumo-accident-simulation},
  note      = {Branch: antifragility. Implements Nilsson (1981) Power Model,
               NHTSA KABCO severity tiers, and a network antifragility index}
}
```

---

## Roadmap

- [ ] Merge antifragility analysis into a standalone Python package
- [ ] Multi-city replication (Anaheim, Chicago, Barcelona networks)
- [ ] Weather-conditioned risk multipliers (rain, ice, fog)
- [ ] Adaptive signal control response to detected accidents
- [ ] Formal MFD calibration pipeline per network
