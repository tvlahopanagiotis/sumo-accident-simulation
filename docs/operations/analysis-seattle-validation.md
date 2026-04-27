# Command: `sas-compare-seattle-real`

Compare Seattle simulation accident outputs against historical Seattle
collision data.

## What It Does

1. loads one simulation directory or a batch of Seattle runs
2. loads the Seattle real-collision dataset
3. aligns severity, time-window, and spatial summaries
4. generates comparative figures and a summary JSON

## Main Inputs

- simulation run directory or batch root
- Seattle real collisions CSV

## Main Outputs

- `comparison_dashboard.png`
- `spatial_density_comparison.png`
- `comparison_summary.json`

## Typical Usage

```bash
sas-compare-seattle-real \
  --sim-dir results/Seattle_Batch_2026-03-06_11:10 \
  --real-csv data/cities/seattle/bundle/crash_data/sdot_collisions_all_years.csv
```

## Operational Notes

- This is the project's main historical validation comparison workflow for the
  Seattle bundle.
- See [`../SEATTLE_DATA.md`](../SEATTLE_DATA.md) for the Seattle dataset layout.
