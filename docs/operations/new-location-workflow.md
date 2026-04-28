# New Location Workflow

This guide is the end-to-end path for adding a new study area to SAS and
taking it from zero data to runnable analysis.

It is written for a user who is starting a new location rather than using one
of the bundled cases.

## 1. Prepare The Environment

Install SUMO, create a Python environment, and install SAS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For the GUI:

```bash
cd frontend
npm install
```

Then confirm SUMO is reachable:

```bash
sumo --version
```

## 2. Choose The New Location

Decide the operational scope before downloading anything:

- city center only or wider metro area
- congestion study or resilience stress test
- synthetic demand only or future O-D calibration target

This decision affects:

- the OSM extract size
- simulation runtime
- route realism
- whether the current generator set is enough or a new one is needed

## 3. Fetch The Base OSM Data

Use `sas-fetch-osm` to get a raw `.osm` file:

```bash
sas-fetch-osm \
  --place "Athens, Greece" \
  --city-slug athens \
  --pad-km 1.5
```

If you prefer the GUI, use the `Data & Integrations` page to:

- open `New Extract`
- search the place
- inspect the locality boundary
- refine the bbox
- bootstrap the city folder and default config
- launch the download

The current default bootstrap path is:

- `data/cities/athens/network/athens.osm`
- `configs/athens/default.yaml`

Before moving on, open `Extracted Network` and do a quick input-quality pass:

- inspect speed-limit coverage
- inspect road types and lane/direction tagging
- review signalized intersections
- optionally fill missing speed tags for clear groups such as local roads with
  unknown speeds

## 4. Review The City Folder And Config

The minimum decisions are:

- output folder
- SUMO `.sumocfg` path
- runtime horizon
- seed

The OSM workflow now creates the starter config automatically, so this step is
mainly a review-and-adjust pass rather than a blank-file creation step.

## 5. Build A Runnable Network And Demand Setup

At this point you have two paths:

### Path A: reuse an existing generator pattern

If the city is OSM-based and similar to Thessaloniki, create a new generator by
adapting the city-generator pattern in `src/sas/generators/`.

What the generator should do:

1. read the local `.osm`
2. run `netconvert`
3. generate demand
4. write a `.sumocfg`
5. optionally patch the YAML config

### Path B: assemble files manually for a first run

For a first proof-of-concept, you can manually prepare:

- `<city>.net.xml`
- `<city>.rou.xml`
- `<city>.sumocfg`

and point the YAML to the `.sumocfg`.

This is acceptable for initial validation before investing in a polished city
generator.

## 6. Decide The Demand Method

The current repository supports two practical demand modes:

- synthetic demand using `randomTrips.py`
- city-specific structured inputs where a generator already supports them

For a new location, the default first step is usually synthetic demand because
it gets the network running quickly.

Later, if the city becomes a serious study case, move toward:

- O-D matrices
- TAZ demand
- time-of-day demand profiles

## 7. Validate The Config

Before running a full simulation:

1. open the config in `Config Studio`
2. verify the `.sumocfg` path
3. confirm output paths
4. confirm `step_length`, incident settings, and metrics windows
5. validate and save

For scientific reference runs, keep:

- `sumo.step_length: 1`

## 8. Run The First Simulation

Use:

```bash
sas --config configs/athens/default.yaml
```

Start with one single run and verify:

- the network loads
- vehicles are inserted
- metrics are written
- incidents appear if probabilities are non-zero

If the city is large, keep the first run short and conservative.

## 9. Review The Outputs

Check:

- `network_metrics.csv`
- `accident_reports.json`
- `metadata.json`
- generated figures and report outputs

Use the GUI `Results` page if you want a faster visual pass over the run.

## 10. Run The Analysis Layer

After the first run works, move to analysis.

Main paths:

- one-click resilience assessment
- batch analysis
- parameter sweep
- historical comparison if real data exist

See:

- [`analysis.md`](analysis.md)

## 11. Decide Whether The Location Needs A Dedicated Generator

If the location will be used repeatedly, it is worth adding a real dedicated
generator under `src/sas/generators/` and a stable config under
`configs/<city>/`.

If it is only exploratory, the manual or lightly adapted setup may be enough.

## 12. Recommended Next Improvements For A New Location

After the first end-to-end success, the usual upgrade order should be:

1. improve demand realism
2. calibrate the incident rate and severity parameters
3. validate output plausibility against observed traffic or incident data
4. package the city as a reusable generator plus config set

## Related Guides

- [`README.md`](README.md)
- [`generators.md`](generators.md)
- [`data-integrations.md`](data-integrations.md)
- [`analysis.md`](analysis.md)
