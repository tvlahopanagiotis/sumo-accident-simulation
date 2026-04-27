# Analysis Operations

This guide covers the analyst-facing command layer in SAS.

## Resilience Assessment

### `sas-assess`

Run the one-click resilience assessment workflow across multiple demand levels,
incident settings, and seeds.

Typical usage:

```bash
sas-assess
```

```bash
sas-assess --quick
```

```bash
sas-assess --workers 4 --demand-levels 1.0 2.0 5.0 --seeds 5
```

## Batch Post-Processing

### `sas-analyse-batch`

Analyse an existing batch or assessment directory and produce comparative
figures and summary outputs.

Typical usage:

```bash
sas-analyse-batch --batch-dir results/Thessaloniki_Batch_2026-03-05_16:53
```

## Parameter Sweep

### `sas-sweep`

Run the parameter-grid sweep for failure-point and regime analysis.

Typical usage:

```bash
sas-sweep
```

```bash
sas-sweep --periods 2.0 1.0 0.5
```

```bash
sas-sweep --probs 0 1.5e-4 1e-3
```

### `sas-visualise-sweep`

Turn the sweep CSV into failure-point and phase-diagram figures.

Typical usage:

```bash
sas-visualise-sweep
```

```bash
sas-visualise-sweep --csv results/sweep/sweep_results.csv --out-dir results/sweep/figs
```

## Report Regeneration

### `sas-merge-report`

Merge supplementary MFD data into an existing assessment and rebuild the MFD
figures and report without re-running simulations.

Typical usage:

```bash
sas-merge-report \
  --main results/resilience_2026-03-06_1418 \
  --extra results/resilience_low_demand_0p1_0p3
```

## Seattle Historical Comparison

### `sas-compare-seattle-real`

Compare Seattle simulation outputs against historical Seattle collision data.

Typical usage:

```bash
sas-compare-seattle-real \
  --sim-dir results/Seattle_Batch_2026-03-06_11:10 \
  --real-csv data/cities/seattle/bundle/crash_data/sdot_collisions_all_years.csv
```

## Related Docs

- [`../modules/analysis.md`](../modules/analysis.md)
- [`new-location-workflow.md`](new-location-workflow.md)
- [`../SEATTLE_DATA.md`](../SEATTLE_DATA.md)
