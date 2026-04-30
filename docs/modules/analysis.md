# Analysis Module

This guide covers the workflows that transform SUMA run outputs into resilience
findings, figures, comparative summaries, and validation artifacts.

## Scope

The analysis layer is mainly implemented in:

- `src/suma/analysis/`
- `src/suma/tools/`
- parts of `src/suma/visualization/`

## What The Analysis Module Does

The analysis workflows consume simulation outputs and convert them into
decision-oriented summaries.

They currently support:

- resilience assessment across many scenarios
- post-processing of existing batches
- sweep-based sensitivity studies
- MFD figure regeneration
- Seattle historical comparison

## Main Analysis Paths

### Resilience assessment

This is the largest integrated analysis workflow. It creates a scenario matrix,
runs many simulations, aggregates the results, and produces an HTML resilience
report.

### Batch and sweep analysis

These workflows help quantify how the network behaves across demand and incident
regimes rather than only for one chosen configuration.

### Historical comparison

The Seattle comparison path is the validation-oriented analysis workflow in the
current repository.

## Where It Fits In The Full Workflow

Analysis comes after simulation, but it also feeds back into configuration and
model improvement.

Typical chain:

1. generate or fetch inputs
2. run simulation
3. analyse results
4. refine parameters, demand, or incident logic

## Related Command Guides

- [`../operations/analysis.md`](../operations/analysis.md)
- [`../operations/new-location-workflow.md`](../operations/new-location-workflow.md)

## Recommendations For Improvement

### Unify the analysis data contract

Several workflows read overlapping artifacts and rebuild overlapping summary
logic. A clearer shared analysis schema would make the tools easier to combine
and extend.

### Expand resilience metrics beyond speed-centered recovery

The current analysis layer already does useful work, but it should increasingly
treat throughput, travel time, queue dissipation, and welfare-oriented measures
as first-class outputs alongside AI.

### Tighten calibration and validation loops

The strongest next step is to connect analysis outputs back to model tuning
more directly, especially for demand realism, severity calibration, and
historical fit.

### Improve reproducibility bundles

Large assessments should export a clearer package of:

- config snapshot
- scenario matrix
- analysis version metadata
- generated figures
- summary JSON tables

That would make external review and handoff easier.
