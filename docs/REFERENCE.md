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

One entry per accident.

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

## Risk Model

### Composite score

```text
risk_score  =  speed_weight      × speed_risk
            +  variance_weight   × variance_risk
            +  density_weight    × density_risk

final_risk  =  clamp(risk_score × road_type_multiplier, 0, 1)
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
pytest tests/ --cov=. --cov-report=term-missing
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
