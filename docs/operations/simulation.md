# Simulation Operations

This guide covers the simulator-facing command layer.

## Main Command

### `sas`

This is the main SAS simulator entry point. It runs SUMO, injects incidents
through the SAS risk and accident layers, records metrics, and writes the run
artifacts under the configured output folder.

#### Typical usage

Default Thessaloniki run:

```bash
sas
```

Run a different config:

```bash
sas --config configs/seattle/default.yaml
```

Repeat the same setup multiple times with different seeds:

```bash
sas --config configs/thessaloniki/default.yaml --runs 10
```

Enable the Python live dashboard:

```bash
sas --live-progress
```

#### Main outputs

- `network_metrics.csv`
- `accident_reports.json`
- `antifragility_index.json`
- `metadata.json`
- figures and HTML report files when enabled

## Operational Notes

- The simulator is now tuned around `sumo.step_length: 1` for reference runs.
- Incident behavior is controlled mainly by the `risk`, `accident`, and
  `output` sections of the YAML.
- The current default incident model is stochastic incident injection on top of
  SUMO, not native collision generation.

## Related Docs

- [`../modules/simulator.md`](../modules/simulator.md)
- [`../REFERENCE.md`](../REFERENCE.md)
- [`../SUMO_ACCIDENT_SIMULATOR_REVIEW.md`](../SUMO_ACCIDENT_SIMULATOR_REVIEW.md)
