# Command: `sas-merge-report`

Merge supplementary MFD data into an existing assessment and rebuild the MFD
figures and HTML report without re-running the simulations.

## What It Does

1. merges `mfd_data.csv` from an extra run into the main assessment
2. rewrites the merged CSV
3. regenerates the MFD figures
4. rebuilds the HTML resilience report

## Main Inputs

- `--main` assessment directory
- `--extra` supplementary assessment directory

## Main Outputs

- updated MFD CSV
- regenerated MFD figures
- refreshed HTML resilience report in the main directory

## Typical Usage

```bash
sas-merge-report \
  --main results/resilience_2026-03-06_1418 \
  --extra results/resilience_low_demand_0p1_0p3
```

## Operational Notes

- Use this when additional demand cases or supplementary runs are available
  after the main assessment has already finished.
