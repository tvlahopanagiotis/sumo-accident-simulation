# Operations

This section is the operator-facing workflow map for the CLI tools in SAS.
Each file covers one command or one tightly related operation.

Use this section when you need to run the project. Use
[`../REFERENCE.md`](../REFERENCE.md) when you need formulas, parameter
semantics, or output definitions.

## Simulation

- [simulation-runner.md](simulation-runner.md): `sas`

## Generators

- [generator-thessaloniki.md](generator-thessaloniki.md): `sas-generate-thessaloniki`
- [generator-seattle.md](generator-seattle.md): `sas-generate-seattle`
- [generator-sioux-falls.md](generator-sioux-falls.md): `sas-generate-sioux-falls`
- [generator-riverside.md](generator-riverside.md): `sas-generate-riverside`

## Data And Integrations

- [data-osm.md](data-osm.md): `sas-fetch-osm`
- [data-govgr-download.md](data-govgr-download.md): `sas-fetch-govgr`
- [data-govgr-targets.md](data-govgr-targets.md): `sas-build-govgr-targets`

## Analysis

- [analysis-resilience-assessment.md](analysis-resilience-assessment.md): `sas-assess`
- [analysis-batch.md](analysis-batch.md): `sas-analyse-batch`
- [analysis-sweep.md](analysis-sweep.md): `sas-sweep`
- [analysis-visualise-sweep.md](analysis-visualise-sweep.md): `sas-visualise-sweep`
- [analysis-merge-report.md](analysis-merge-report.md): `sas-merge-report`
- [analysis-seattle-validation.md](analysis-seattle-validation.md): `sas-compare-seattle-real`

## Recommended Reading Order

For a new city workflow:

1. build or fetch data in `Data And Integrations`
2. run the relevant generator
3. run [`simulation-runner.md`](simulation-runner.md)
4. run one or more analysis workflows

For parameter semantics and the current incident logic review:

- [`../REFERENCE.md`](../REFERENCE.md)
- [`../SUMO_ACCIDENT_SIMULATOR_REVIEW.md`](../SUMO_ACCIDENT_SIMULATOR_REVIEW.md)
