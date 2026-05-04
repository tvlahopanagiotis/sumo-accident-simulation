# Module Guides

This section is the narrative layer of the SUMA documentation.

Use it when you want to understand the project by responsibility rather than
by individual command. The module guides explain what each major part of the
system is for, how it fits into the overall workflow, and where the main
improvement opportunities are.

## Reading Order

For a first pass through the repository in pipeline order:

1. [data-integrations.md](data-integrations.md)
2. [generators.md](generators.md)
3. [simulator.md](simulator.md)
4. [analysis.md](analysis.md)

Then move to:

- [`../operations/README.md`](../operations/README.md) for the operational runbooks
- [`../REFERENCE.md`](../REFERENCE.md) for formulas, parameters, and outputs
- [`../SUMO_ACCIDENT_SIMULATOR_REVIEW.md`](../SUMO_ACCIDENT_SIMULATOR_REVIEW.md) for the detailed technical review of the current incident model

## How To Use These Guides

- `simulator.md`: the runtime core, accident logic, outputs, and modeling limits
- `generators.md`: how city and benchmark networks are built and where demand comes from
- `data-integrations.md`: how external data enters the repository
- `analysis.md`: how SUMA turns simulation outputs into resilience findings
