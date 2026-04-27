# Command: `sas-analyse-batch`

Analyse an existing batch or assessment directory and produce comparative
figures and summary outputs.

## What It Does

1. reads completed batch results
2. aggregates run-level and event-level outputs
3. produces comparative charts and summary tables

## Main Inputs

- a batch directory under `results/`

## Main Outputs

- analysis figures
- summary tables
- derived comparative artifacts in the chosen output area

## Typical Usage

```bash
sas-analyse-batch --batch-dir results/Thessaloniki_Batch_2026-03-05_16:53
```

## Operational Notes

- Use this when the simulations already exist and you want a cleaner
  post-processing pass without re-running SUMO.
