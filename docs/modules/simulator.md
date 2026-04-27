# Simulator Module

This guide covers the runtime core of SAS: the part that launches SUMO,
injects incidents, evolves them through time, and records the outputs used by
the resilience workflows.

## Scope

The simulator module is mainly implemented in:

- `src/sas/simulation/`
- `src/sas/core/`
- `src/sas/app/config.py`

The most important files are:

- `src/sas/simulation/runner.py`
- `src/sas/core/risk_model.py`
- `src/sas/core/accident_manager.py`
- `src/sas/core/metrics.py`

## What The Simulator Does

At runtime, SAS is a control layer on top of a normal SUMO simulation.

The process is:

1. load a YAML configuration
2. start SUMO through TraCI
3. subscribe vehicles and neighborhood context
4. compute step-level density and per-vehicle incident propensity
5. trigger stochastic incidents when risk and probability conditions are met
6. apply incident effects to the network
7. let incidents progress through `ACTIVE`, `CLEARING`, and `RESOLVED`
8. reroute nearby traffic when enabled
9. record network and event metrics

## Main Responsibilities

### Runtime control

`runner.py` owns the main simulation loop and the contract with SUMO.

It is responsible for:

- starting SUMO with the configured `.sumocfg`
- advancing the simulation step by step
- subscribing to vehicle and context variables
- passing state to the risk, accident, and metrics layers

### Risk evaluation

`risk_model.py` decides which vehicles are candidates for incident triggering.

The current logic uses:

- speed relative to the posted speed limit
- local speed heterogeneity
- density in vehicles per lane-km
- a speed-limit-based road-type multiplier

### Incident lifecycle

`accident_manager.py` owns what happens after an incident is triggered.

The current implementation supports:

- stopped incident vehicles
- explicit lane closures
- lane speed degradation
- clearing-phase recovery
- local travel-time-based rerouting

### Metrics and outputs

`metrics.py` records the system-wide outputs and accident-level effects.

The current output layer is built around:

- network speed
- throughput
- speed-ratio-based delay proxy
- accident reports
- antifragility windows and event AI

## Where It Fits In The Full Workflow

The simulator sits between network/demand preparation and the analysis tools.

Typical chain:

1. fetch or prepare data
2. generate network and demand
3. run `sas`
4. run resilience or post-processing analysis

## Related Command Guides

- [`../operations/simulation.md`](../operations/simulation.md)
- [`../REFERENCE.md`](../REFERENCE.md)
- [`../SUMO_ACCIDENT_SIMULATOR_REVIEW.md`](../SUMO_ACCIDENT_SIMULATOR_REVIEW.md)

## Recommendations For Improvement

### Make step-length semantics explicit

The current simulator works best at `1 s`, but some timing fields are still
effectively hardwired to second-based assumptions. The next cleanup should make
that contract explicit in code and documentation, or convert all timing to true
simulation-step units.

### Upgrade rerouting from local to route-aware

The present rerouting logic is a useful first network-response layer, but it is
still based mainly on local proximity rather than explicit downstream path
exposure. Path-aware rerouting is the next high-value improvement.

### Calibrate incident effects, not just trigger rates

The simulator now has a better operational disruption model, but the severity
tiers should still be calibrated against observed blocked-lane, clearance-time,
and residual-capacity patterns.

### Broaden the resilience output set

AI is useful, but the simulator should expose more directly analysis-ready
welfare and recovery measures such as vehicle-hours lost, recovery time, and
corridor-specific degradation.
