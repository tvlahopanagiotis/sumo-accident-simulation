# OD Generators Module

This guide covers the workflows that build SUMO-ready networks, route files,
and `.sumocfg` files for extracted cities plus the benchmark/synthetic cases.
The GUI labels this area `OD Generators` because demand creation is now a first
class part of the workflow, even when the selected demand source is random.

## Scope

The generators live in:

- `src/sas/generators/`

The current active generator paths are:

- generic city generation for any extracted city under `data/cities/<slug>/`
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
- optionally patch SUMA/SAS YAML configs to point at the generated artifacts

## Generator Types

### Real-city generator

The main real-city workflow is now the generic city generator.

It starts from the standard city layout created by the OSM extract flow and
then:

- compiles the `.osm` into a SUMO `.net.xml`
- generates demand from either:
  - `randomTrips.py`
  - OD inputs when the city folder contains compatible support files
- writes `<city>.sumocfg`
- optionally patches `configs/<city>/default.yaml`

The GUI now complements that with a generator-side input viewer so operators
can inspect discovered OD matrices and centroid nodes before building routes.
For cities without OD files, the same tab still explains random-demand settings
and estimates rough requested trips from the configured period and end time.

For random-demand runs, the main density control is the route period. Lower
period values request departures more often, but the observed number of valid
trips and simultaneous vehicles still depends on network connectivity and trip
length, not only on the period itself.

### Benchmark and synthetic generators

Benchmark and synthetic generators are useful for testing, experimentation, and
faster turnaround.

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
