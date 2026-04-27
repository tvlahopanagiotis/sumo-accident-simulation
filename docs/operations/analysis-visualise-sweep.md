# Command: `sas-visualise-sweep`

Turn the sweep CSV from `sas-sweep` into failure-point and phase-diagram
figures.

## What It Does

1. reads `sweep_results.csv`
2. aggregates repeated cells across seeds
3. writes publication-style figures for speed, AI, and regime behavior

## Main Inputs

- sweep CSV, usually `results/sweep/sweep_results.csv`

## Main Outputs

- `fig1_speed_vs_load.png`
- `fig2_ai_vs_load.png`
- `fig3_heatmaps.png`
- `fig4_phase_diagram.png`

## Typical Usage

```bash
sas-visualise-sweep
```

```bash
sas-visualise-sweep --csv results/sweep/sweep_results.csv --out-dir results/sweep/figs
```

## Operational Notes

- This is a pure visualization pass. It does not launch any new simulations.
