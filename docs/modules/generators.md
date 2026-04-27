# Generators Module

This guide covers the workflows that build SUMO-ready networks, route files,
and `.sumocfg` files for the bundled study areas and benchmarks.

## Scope

The generators live in:

- `src/sas/generators/`

The current bundled generators are:

- Thessaloniki
- Seattle
- Sioux Falls
- Riverside

## What The Generators Do

The generators turn raw or built-in spatial definitions into runnable SUMO
inputs.

Depending on the case, they:

- download or read OSM data
- compile networks with `netconvert`
- generate demand from `randomTrips.py` or city-specific inputs
- write route files
- write `.sumocfg` files
- optionally patch SAS YAML configs to point at the generated artifacts

## Generator Types

### Real-city generators

The Thessaloniki and Seattle workflows are closer to production use because
they start from city-scale network data.

- Thessaloniki is primarily an OSM-driven build with synthetic demand.
- Seattle mixes OSM and bundled OD-oriented inputs.

### Benchmark and synthetic generators

Sioux Falls and Riverside are useful for testing, experimentation, and faster
turnaround.

- Sioux Falls is the compact benchmark case.
- Riverside is the synthetic development network.

## Where They Fit In The Full Workflow

Generators are the bridge between data acquisition and simulation.

Typical chain:

1. fetch external data if needed
2. run a generator
3. run the simulator
4. analyse results

## Related Command Guides

- [`../operations/generators.md`](../operations/generators.md)
- [`../operations/new-location-workflow.md`](../operations/new-location-workflow.md)

## Recommendations For Improvement

### Separate network building from demand building more cleanly

Some generator scripts still combine network compilation, demand generation,
and config patching in one operational path. Splitting those concerns more
clearly would make recalibration and reruns easier.

### Create a shared generator utility layer

The generators still duplicate patterns such as subprocess launching, config
patching, and route-file writing. A shared internal helper layer would reduce
drift across city workflows.

### Strengthen demand realism where data exist

The biggest generator-side scientific opportunity is to move more workflows
away from pure random-trip demand toward O-D, TAZ, or time-profile-driven
demand.

### Emit richer metadata after generation

Each generator should write a small metadata file summarizing:

- source inputs
- build timestamp
- demand method
- key command parameters
- output artifacts

That would improve traceability for later analysis and reporting.
