# Pilot City Traffic Data Findings

This note captures the first-pass source scan for the pilot cities:

- Thessaloniki
- Larissa
- Bratislava
- Odesa

It also records broader open-data directions for later city-agnostic work.

## Summary

The strongest current road-traffic source is still Thessaloniki. Bratislava and
Odesa look more promising for public-transport and mobility-data integration
than for open road-speed feeds. Larissa currently looks more like a
partnership-led integration case unless a better municipal API is uncovered.

## Thessaloniki

Status:
- strong candidate
- already operational in the repository

Current sources:
- Greek National Access Point congestion dataset:
  https://data.nap.gov.gr/dataset/traffic-congestion
- Greek National Access Point floating-car-data travel-times dataset:
  https://data.nap.gov.gr/el/dataset/floating-car-data-travel-times
- Thessaloniki live traffic map:
  https://trafficthess.imet.gr/

Assessment:
- best current pilot-city source for road-traffic speed/congestion analysis
- already aligned with the existing `govgr` downloader and target builder

## Larissa

Status:
- weak open road-traffic signal so far

Current sources found:
- Municipality smart access page:
  https://www.larissa.gov.gr/en/for-citizens/pedestrian-streets-smart-access
- Operating portal:
  https://platform.cityzenapp.gr/cityzen/larisa/functions

Assessment:
- operationally interesting, especially around controlled-access / smart-city
  functions
- no clearly documented open traffic API or open downloadable traffic feed was
  confirmed in this scan
- likely next step is direct contact with the municipality or the platform
  operator

## Bratislava

Status:
- moderate candidate
- stronger for transit, restrictions, and counters than for open road-speed
  feeds

Current sources found:
- City open-data portal:
  https://bratislava.sk/en/city-of-bratislava/transparent-city/open-data
- IDS BK open data:
  https://www.idsbk.sk/en/about/open-data/
- Road restrictions and disorders:
  https://bratislava.sk/en/transport-and-maps/road-administration-and-maintenance/restrictions-and-disorders
- Bicycle counters:
  https://bratislava.sk/en/transport-and-maps/cycling/bicycle-counters

Assessment:
- promising for GTFS / public transport integration
- useful for restrictions, incidents, bicycle counts, and related operational
  context
- no Thessaloniki-like open floating-car-data speed/congestion feed was
  confirmed in this scan

## Odesa

Status:
- moderate candidate
- strongest signal is transport open data, especially GTFS-style feeds

Current sources found:
- Official open-data page reference via the transport department:
  https://omr.gov.ua/ua/city/departments/dtks/open-data/
- Transport department page:
  https://omr.gov.ua/ru/city/departments/dtks
- Mobility Database indexed official feed:
  https://mobilitydatabase.org/feeds/gtfs/mdb-2946

Assessment:
- promising for public transport feed integration
- likely useful for GTFS static and possibly related live transport surfaces
- no clearly verified open citywide road-speed/congestion feed was confirmed in
  this scan

## General Open Traffic Data Directions

The most reusable open-data paths for future city-agnostic work are:

### GTFS / GTFS-Realtime

Best for:
- public transport schedules
- vehicle positions
- trip updates
- service alerts

Useful sources:
- GTFS overview and specifications:
  https://gtfs.org/
- GTFS-Realtime:
  https://gtfs.org/spec/gtfs-realtime/
- Mobility Database:
  https://mobilitydatabase.org/
- Transitland:
  https://www.transit.land/

### DATEX II and National Access Points

Best for:
- road incidents
- closures
- travel times
- traffic counts
- national / regional road operations

Useful sources:
- DATEX II:
  https://datex2.eu/
- NAPCORE standards overview:
  https://napcore.eu/standards/

### GBFS

Best for:
- shared mobility
- bike share and related station / vehicle availability

Useful source:
- https://gbfs.org/

## Practical Conclusion For SUMA

Near-term priority order:

1. keep Thessaloniki as the benchmark road-traffic integration
2. investigate Odesa for transit-feed integration
3. investigate Bratislava for GTFS plus restrictions/counters
4. treat Larissa as a targeted municipal-integration case unless better open
   traffic feeds are found

## Recommended Next Step

When this becomes a later-version task, the next useful deliverable would be a
source-matrix and adapter plan with columns such as:

- city
- source type
- modality
- official / unofficial
- API / file / portal
- spatial join possible to OSM or SUMO
- temporal resolution
- license / access constraints
- current integration priority
