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

Use [`operations/README.md`](operations/README.md) for workflow instructions.
This file only covers parameter semantics and output definitions.

## Macroscopic Metrics

The current SUMA simulator exports system-wide measures from SUMO telemetry at
the configured metrics interval. These are operational first-pass indicators for
WP5/SUMA experimentation; they are not yet calibrated antifragility indicators.

### Free-flow baseline and speed ratio

The baseline speed is estimated from clean early snapshots with no active
incidents:

```text
V_base = mean(V_t) for t <= baseline_window and active_accidents_t = 0
```

Each later snapshot reports:

```text
speed_ratio_t = V_t / V_base
```

where `V_t` is the network mean speed at the snapshot. Values below `1.0`
indicate degraded network operation relative to the early free-flow baseline.

### Delay proxy

The current delay field is a speed-ratio proxy over one reporting interval:

```text
mean_delay_seconds_t = (1 - speed_ratio_t) x metrics_interval_seconds
```

This is useful for comparing runs internally, but it is not yet a calibrated
vehicle-hours-lost measure. Future versions should compute delay from completed
trip travel times, route alternatives, and/or edge-level reference speeds.

### Throughput

Throughput is calculated from vehicles that arrived during the last metrics
interval:

```text
throughput_per_hour_t = arrivals_interval x 3600 / metrics_interval_seconds
```

### Antifragility Index

For each resolved incident with enough clean pre/post samples:

```text
event_AI = (V_post / V_pre) - 1
AI       = mean(event_AI)
```

`V_pre` is the mean network speed in the pre-incident lookback window. `V_post`
is the mean network speed after the incident is resolved, using only clean
snapshots with no active incidents. When at least two events are measured, SUMA
reports a 95 percent confidence interval:

```text
CI95 = AI +/- t_critical(n) x std(event_AI) / sqrt(n)
```

This implementation follows the SUMA development context by exposing the
calculation mode and sample counts. The thresholds below are working
interpretation bands and must be validated against project methodology and pilot
observations before they are used as formal antifragility evidence.

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
pytest tests/ --cov=src/suma --cov-report=term-missing
pytest tests/ -m integration
```

### Code Quality

```bash
ruff check .
mypy src/suma
```

## References

- Nilsson, G. (1981). *The Effects of Speed Limits on Traffic Accidents in Sweden.*
- Elvik, R. (2009). *The Power Model of the Relationship Between Speed and Road Safety.*
- NHTSA (2022). *Traffic Safety Facts 2022.*
- Lopez, P.A., et al. (2018). *Microscopic Traffic Simulation using SUMO.*
