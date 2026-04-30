# OD Generator Operations

This guide covers the active generator commands. Use it when you want the
practical runbook for creating networks, demand, and `.sumocfg` files.

## Generic City Generator

### `suma-generate-city`

Build a runnable SUMO network for any extracted city under `data/cities/<slug>/`.

What it does:

1. reads the city `.osm` extract created earlier
2. converts it to SUMO with `netconvert`
3. generates demand with either:
   - `randomTrips.py`
   - OD inputs when a compatible OD and node file pair exists
4. writes `<city>.sumocfg`
5. optionally patches a YAML config

Typical usage:

```bash
suma-generate-city --city-slug thermi
```

```bash
suma-generate-city --city-slug thermi --update-config
```

```bash
suma-generate-city --city-slug thermi --demand-source random --period 2.0
```

```bash
suma-generate-city --city-slug seattle --demand-source od --od-scale 0.02
```

Use `suma-generate-city` for new SUMA-facing workflows. The older
`sas-generate-city` command remains available as a compatibility alias.

Random-demand note:

- the main trip-volume control is `--period`
- lower `--period` means more requested departures over the same simulation
  horizon
- a rough first estimate is:
  - requested trips ≈ `end_time / period`
- the final generated trip count can still vary with network connectivity and
  route validation, so a small or fragmented network may yield fewer valid
  trips than the simple estimate suggests

Operational note:

- If the city `.osm` extract is missing, fetch it first through the OSM data
  workflow.
- If `--demand-source od` is used and `--od-file` / `--node-file` are not
  passed explicitly, the generator scans the city folder for compatible
  `*_od.csv` and `*_node.csv` files.
- In the GUI, use the generator page `View Inputs` tab to inspect those OD and
  node files before launching the build.
- In the GUI, the city generator form now shows random-only or OD-only fields
  depending on the selected demand source.
- The `View Inputs` tab also reports a rough random-demand trip estimate using
  `end_time / period`, and can show origin and destination demand totals for a
  selected OD zone.

## Benchmark Generators

### `suma-generate-sioux-falls`

Generate compact benchmark networks for SUMO. Sioux Falls is the current
implemented benchmark workflow.

Typical usage:

```bash
suma-generate-sioux-falls
```

```bash
suma-generate-sioux-falls --update-config
```

```bash
suma-generate-sioux-falls --period 2.0
```

## Synthetic Generators

### `suma-generate-riverside`

Generate synthetic development networks. Riverside is the current implemented
synthetic workflow.

Typical usage:

```bash
suma-generate-riverside
```

```bash
suma-generate-riverside --out-dir /tmp/riverside-test
```

```bash
suma-generate-riverside --update-config
```

## After Generation

After any generator run:

1. verify the `.sumocfg`
2. point your YAML config at it
3. run the simulator

Next step:

- [`simulation.md`](simulation.md)

## Related Docs

- [`../modules/generators.md`](../modules/generators.md)
- [`new-location-workflow.md`](new-location-workflow.md)
- [`../SEATTLE_DATA.md`](../SEATTLE_DATA.md)
