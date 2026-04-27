# Command: `sas-assess`

Run the one-click resilience assessment workflow across multiple demand levels,
incident settings, and seeds.

## What It Does

1. builds a scenario matrix
2. prepares route files per demand level
3. runs the scenarios in parallel
4. aggregates outputs across scenarios
5. produces resilience summaries, figures, and an HTML report

## Main Inputs

- base YAML config
- `resilience_assessment` section of that config
- optional worker, demand-level, and seed overrides

## Main Outputs

Under the assessment output directory:

- scenario matrix files
- per-run result folders
- aggregate summaries
- MFD datasets and figures
- resilience report HTML

## Typical Usage

Run the configured assessment:

```bash
sas-assess
```

Run a smaller quick mode:

```bash
sas-assess --quick
```

Run a targeted subset:

```bash
sas-assess --workers 4 --demand-levels 1.0 2.0 5.0 --seeds 5
```

## Operational Notes

- This is the main macroscopic resilience workflow in the project.
- The incident-rate ladder now assumes the calibrated `1 s` runtime regime.
