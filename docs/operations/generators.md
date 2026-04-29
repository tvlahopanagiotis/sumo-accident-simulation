# Generator Operations

This guide covers the active generator commands. Use it when you want the
practical runbook for creating networks, demand, and `.sumocfg` files.

## Generic City Generator

### `sas-generate-city`

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
sas-generate-city --city-slug thermi
```

```bash
sas-generate-city --city-slug thermi --update-config
```

```bash
sas-generate-city --city-slug thermi --demand-source random --period 2.0
```

```bash
sas-generate-city --city-slug seattle --demand-source od --od-scale 0.02
```

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

## Sioux Falls

### `sas-generate-sioux-falls`

Generate the Sioux Falls benchmark network for SUMO.

Typical usage:

```bash
sas-generate-sioux-falls
```

```bash
sas-generate-sioux-falls --update-config
```

```bash
sas-generate-sioux-falls --period 2.0
```

## Riverside

### `sas-generate-riverside`

Generate the synthetic Riverside District SUMO network.

Typical usage:

```bash
sas-generate-riverside
```

```bash
sas-generate-riverside --out-dir /tmp/riverside-test
```

```bash
sas-generate-riverside --update-config
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
