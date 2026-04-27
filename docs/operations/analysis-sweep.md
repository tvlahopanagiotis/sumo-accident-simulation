# Command: `sas-sweep`

Run the parameter-grid sweep for failure-point and regime analysis.

## What It Does

1. sweeps over demand periods, base accident probabilities, and seeds
2. generates or reuses route files per demand level
3. writes per-cell configs
4. runs SAS as a subprocess for each cell
5. stores one summary row per cell in a sweep CSV

## Main Inputs

- configured period grid
- configured probability grid
- seed count

## Main Outputs

Under `results/sweep/` by default:

- `sweep_results.csv`
- per-cell run directories
- sweep logs

## Typical Usage

Full default grid:

```bash
sas-sweep
```

Subset of demand levels:

```bash
sas-sweep --periods 2.0 1.0 0.5
```

Subset of probability levels:

```bash
sas-sweep --probs 0 1.5e-4 1e-3
```

## Operational Notes

- This is useful for regime discovery and sensitivity analysis.
- Use [`analysis-visualise-sweep.md`](analysis-visualise-sweep.md) next to turn
  the CSV into figures.
