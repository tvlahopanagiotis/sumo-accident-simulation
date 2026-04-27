# Generator Operations

This guide covers the bundled generator commands. Use it when you want the
practical runbook for creating networks, demand, and `.sumocfg` files.

## Thessaloniki

### `sas-generate-thessaloniki`

Build the Thessaloniki SUMO network and demand set from OpenStreetMap.

What it does:

1. downloads the OSM road network from Overpass
2. converts it to SUMO with `netconvert`
3. generates demand with `randomTrips.py`
4. writes `thessaloniki.sumocfg`
5. optionally patches a YAML config

Typical usage:

```bash
sas-generate-thessaloniki
```

```bash
sas-generate-thessaloniki --update-config
```

```bash
sas-generate-thessaloniki --period 2.0
```

## Seattle

### `sas-generate-seattle`

Build the Seattle SUMO network and demand from the local Seattle bundle.

Typical usage:

```bash
sas-generate-seattle --update-config --config configs/seattle/default.yaml
```

```bash
sas-generate-seattle --demand-source od --od-scale 0.02
```

```bash
sas-generate-seattle --demand-source random --period 1.5
```

Operational note:

- If the source OSM extract is missing, fetch it first through the OSM data
  workflow.

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
