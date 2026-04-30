# Operations

This section is the operator-facing workflow map for the CLI tools in SUMA.
The canonical commands are `suma` and `suma-*`. Historical `sas` and `sas-*`
aliases remain available for older scripts, but new work should use the SUMA
names.
It is intentionally consolidated so the command documentation is easier to
browse and less fragmented.

Use this section when you need to run the project. Use
[`../modules/README.md`](../modules/README.md) when you want the higher-level
story of how the major parts fit together. Use
[`../REFERENCE.md`](../REFERENCE.md) when you need formulas, parameter
semantics, or output definitions.

## Start Here

- [new-location-workflow.md](new-location-workflow.md):
  end-to-end guide for taking a new location from zero data to analysis

## Operational Guides

- [data-integrations.md](data-integrations.md):
  external data acquisition and target-building commands
- [generators.md](generators.md):
  all bundled network and demand generation commands
- [simulation.md](simulation.md):
  simulator command layer, centered on `suma`
- [analysis.md](analysis.md):
  resilience, batch, sweep, and validation analysis commands

## Recommended Reading Order

For a new city workflow:

1. start with [new-location-workflow.md](new-location-workflow.md) for a brand-new city
2. use [data-integrations.md](data-integrations.md) to fetch source data
3. use [generators.md](generators.md) to build runnable SUMO inputs
4. use [simulation.md](simulation.md) to run SUMA
5. use [analysis.md](analysis.md) for results and resilience workflows

For parameter semantics and the current incident logic review:

- [`../REFERENCE.md`](../REFERENCE.md)
- [`../SUMO_ACCIDENT_SIMULATOR_REVIEW.md`](../SUMO_ACCIDENT_SIMULATOR_REVIEW.md)
