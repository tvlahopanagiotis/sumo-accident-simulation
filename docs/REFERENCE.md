# Reference

This document collects the lower-level reference material that does not belong
in the root `README.md`.

## Output Files

All output is written to `results/` unless overridden by config.

### `network_metrics.csv`

| Column | Meaning |
|--------|---------|
| `timestamp_seconds` | Simulation time (seconds) |
| `vehicle_count` | Active vehicles currently present on the network |
| `mean_speed_ms` | Mean vehicle speed (m/s) |
| `mean_speed_kmh` | Mean vehicle speed (km/h) |
| `throughput_per_hour` | Vehicles completing trips per hour |
| `mean_delay_seconds` | Delay vs free-flow baseline |
| `active_accidents` | Number of accidents currently active |
| `speed_ratio` | 1.0 = free-flow; < 1.0 = degraded |

### `accident_reports.json`

One entry per accident, including queue impact, blocked-lane counts, and
rerouted-vehicle counts.

### `antifragility_index.json`

Contains the aggregate Antifragility Index, confidence interval, and
interpretation.

| AI value | Interpretation |
|----------|---------------|
| > +0.05 | Antifragile |
| -0.05 to +0.05 | Resilient |
| -0.20 to -0.05 | Fragile |
| < -0.20 | Brittle |

### `metadata.json`

Full config plus summary metrics for reproducibility.

### `live_progress.png`

Continuously refreshed image of the Python live dashboard.

### `report.html`

Self-contained HTML summary of a completed run.

Use [`operations/README.md`](operations/README.md) for command-by-command
workflow instructions. This file only covers parameter semantics and output
definitions.

## Risk Model

### Composite score

```text
risk_score  =  speed_weight      × speed_risk
            +  variance_weight   × variance_risk
            +  density_weight    × density_risk

final_risk  =  clamp(risk_score × road_type_multiplier, 0, 1)
```

### Trigger rule

An incident is only sampled when `final_risk >= trigger_threshold`.

```text
effective_probability
  = base_probability
  × (1 + 10 × (final_risk − trigger_threshold))
  × secondary_multiplier
```

### Speed risk

```text
speed_risk = (v / v_road_limit) ^ speed_exponent
```

This is based on the Nilsson power model and uses the road posted speed limit,
not the vehicle’s mechanical maximum.

### Speed-variance risk

```text
variance_risk = |v_ego − mean(v_neighbours)| / speed_variance_threshold
```

### Density risk

```text
density_risk = exp( −(density − peak_density)² / (2 σ²) )
```

with `σ = peak_density × 0.5`.

`density` is measured in vehicles per lane-km.

### Road-type multiplier

| Speed limit | Road type | Default multiplier |
|-------------|-----------|-------------------|
| ≥ 90 km/h | Highway | 1.5 |
| ≥ 50 km/h | Arterial | 1.0 |
| < 50 km/h | Local | 0.6 |
| Junction edge | Intersection | 2.0 |

## Severity Model

Configured tiers:

| Tier | Weight | Capacity remaining | Response time |
|------|--------|--------------------|---------------|
| MINOR | 62 | 70 % | 5 min |
| MODERATE | 28 | 40 % | 10 min |
| MAJOR | 8 | 10 % | 20 min |
| CRITICAL | 2 | 0 % | 30 min |

Durations are sampled from clipped log-normal ranges per tier.

## Incident Effect Model

Incidents can operate in three modes:

- `speed_limit`: reduce lane max speeds only
- `lane_closure`: close lanes without adding extra speed degradation
- `hybrid`: combine discrete lane closures with speed reductions on the
  remaining open lanes

The default `hybrid` mode translates `lane_capacity_fraction` into:

- a discrete blocked-lane count on multi-lane edges
- a residual speed factor on the remaining open lanes
- periodic local rerouting around the incident when enabled

On single-lane edges, partial severities are represented through speed
reduction, while full-closure incidents close the only lane.

## Reproducibility

- Runs are seeded through config.
- Batch runs increment seeds from the configured base seed.
- `metadata.json` records the config used for each run.

## Development

### Setup

```bash
pip install -e ".[dev]"
```

### Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src/sas --cov-report=term-missing
pytest tests/ -m integration
```

### Code Quality

```bash
ruff check .
mypy src/sas
```

## References

- Nilsson, G. (1981). *The Effects of Speed Limits on Traffic Accidents in Sweden.*
- Elvik, R. (2009). *The Power Model of the Relationship Between Speed and Road Safety.*
- NHTSA (2022). *Traffic Safety Facts 2022.*
- Lopez, P.A., et al. (2018). *Microscopic Traffic Simulation using SUMO.*
