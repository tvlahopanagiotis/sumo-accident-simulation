# Seattle Simulation Setup

This folder contains a runnable SUMO setup for Seattle.

## Files

- `seattle.osm` — source OSM extract
- `seattle.net.xml` — compiled SUMO network
- `seattle.rou.xml` — OD-driven demand generated from `Seattle_od.csv`
- `seattle.sumocfg` — SUMO scenario file

## Run in this repository

- SAS single run:
  - `python runner.py --config config.seattle.yaml`
- SAS batch:
  - `python runner.py --config config.seattle.yaml --runs 5`
- Raw SUMO smoke run:
  - `sumo -c seattle.sumocfg`

## Rebuild

From repository root:

- `python generate_seattle.py --demand-source od`

Optional:

- `python generate_seattle.py --demand-source od --od-scale 0.02`
- `python generate_seattle.py --demand-source random --period 1.5`
- `python generate_seattle.py --skip-network --demand-source od`
- `python generate_seattle.py --update-config --config config.seattle.yaml`
