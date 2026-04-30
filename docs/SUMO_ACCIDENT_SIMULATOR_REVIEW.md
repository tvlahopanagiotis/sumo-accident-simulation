# SUMO Accident Simulator Review

## Scope

This is a second-round technical review of the current SUMA implementation
after the incident-model upgrades. It focuses on:

- what the simulator now actually does in SUMO and TraCI terms,
- what improved materially,
- what scientific limitations still remain,
- and what the next recommendations should be for macroscopic resilience work.

Bottom line: SUMA is now a much stronger incident-impact simulator than it was
in the first review, but it is still not a native crash-generation model. For
transport planning use, it should be framed as a stochastic incident and
network-disruption simulator built on top of SUMO.

## 1. Priority findings

### Finding 1: the runtime is effectively designed around `1 s` semantics

The current code and configuration now behave most coherently at
`sumo.step_length: 1`. The runner advances simulation time by `step_length`,
but the accident durations, response times, rerouting interval, baseline
window, and AI windows are all interpreted directly in seconds.

Relevant code:

- `src/suma/simulation/runner.py`
- `src/suma/core/accident_manager.py`
- `src/suma/core/metrics.py`
- `src/suma/app/config.py`

Implication:

- this is scientifically fine for the current default,
- but `step_length` is not truly “arbitrary” even though validation allows any
  positive value,
- and coarser values would change exposure, trigger cadence, clearance timing,
  and metrics sampling behavior.

Recommendation:

- keep `1 s` as the reference configuration,
- if faster batch runs are needed, validate `2 s` or `5 s` against the
  `1 s` benchmark before using them,
- and, in a later refactor, either convert all timing parameters to true
  simulation-step units or explicitly constrain supported step lengths.

### Finding 2: rerouting is still proximity-based rather than path-based

The new rerouting logic is a real improvement, but it still reroutes vehicles
based on local Euclidean proximity to the incident rather than on whether the
vehicle’s remaining path actually traverses the disrupted corridor.

Relevant code:

- `src/suma/core/accident_manager.py`

Implication:

- some nearby vehicles may be rerouted even if they are not operationally
  affected,
- while some farther-upstream vehicles that will meet the blockage later may
  not be rerouted,
- so the route-choice response is still heuristic rather than assignment-based.

Recommendation:

- move to route-aware rerouting,
- or use SUMO rerouters / rerouting devices so the incident response is tied
  to actual path exposure rather than only local geography.

### Finding 3: `lane_capacity_fraction` is still an operational proxy

The hybrid incident model is materially better than the earlier speed-only
logic, but severity is still translated into:

- a discrete blocked-lane count, and
- a speed factor on the remaining lanes.

Relevant code:

- `src/suma/core/accident_manager.py`

Implication:

- this is reasonable for controlled scenario analysis,
- but it is not yet a calibrated lane-drop discharge model or empirically
  grounded lane-blockage model.

Recommendation:

- calibrate severity tiers against observed blocked-lane patterns,
- observed clearance durations,
- and observed capacity reduction on the remaining open lanes.

### Finding 4: demand realism remains the largest external-validity gap

The simulator still depends on `randomTrips.py` demand rather than observed
O-D demand, TAZ-based demand, or corridor-specific temporal demand profiles.

Relevant code:

- `src/suma/generators/generate_thessaloniki.py`
- `src/suma/analysis/scenario_generator.py`

Implication:

- the platform is currently stronger for comparative resilience experiments
  than for predictive operational analysis of a specific city-day condition.

Recommendation:

- treat the current workflow as a controlled network stress-test framework,
- and prioritize O-D / TAZ demand integration if the objective becomes
  predictive policy analysis.

### Finding 5: the risk model is still a risk proxy, not a safety model

The current trigger mechanism is coherent, but scientifically it remains a
proxy exposure model:

- “speed variance” is actually absolute deviation from local mean speed,
- road type is inferred from speed limit rather than network functional class,
- density uses a stylized Gaussian-shaped risk curve,
- and incidents remain exogenous stochastic events rather than endogenous SUMO
  collisions.

Relevant code:

- `src/suma/core/risk_model.py`

Recommendation:

- rename “variance” to “speed heterogeneity” unless the formula changes,
- calibrate the curve and weights against observed incident rates if data are
  available,
- and describe the model explicitly as an incident propensity model.

### Finding 6: the AI metric is useful, but too narrow by itself

The Antifragility Index is still based on pre/post mean-speed recovery around
resolved events. That is useful for network recovery analysis, but it is not a
complete welfare or resilience metric set.

Relevant code:

- `src/suma/core/metrics.py`

Recommendation:

- keep AI,
- but pair it with throughput recovery, vehicle-hours lost, queue dissipation
  time, and route/travel-time impacts.

### Finding 7: compound incidents are still intentionally suppressed

The manager prevents a second active incident on the same edge, and the runner
triggers at most one new incident per simulation step.

Relevant code:

- `src/suma/core/accident_manager.py`
- `src/suma/simulation/runner.py`

Implication:

- this is operationally clean and avoids conflicting control states,
- but it likely understates pile-ups, corridor cascades, and compound
  disruptions on critical links.

Recommendation:

- keep this as the default stable mode,
- but document it explicitly and consider an optional experimental mode for
  corridor-level compound incidents later.

## 2. Current implemented process

### 2.1 Network and demand

For Thessaloniki, SUMA loads the network and route files from the `.sumocfg`
referenced by the YAML configuration. The standard workflow still uses
`randomTrips.py`-based demand generation, so vehicle insertion is synthetic and
controlled mainly through trip-generation period.

In transport-planning terms, this means demand is synthetic and scenario-based,
not observed O-D demand. The current platform is therefore better suited to
comparative resilience analysis than to predictive replication of a real daily
traffic pattern.

### 2.2 SUMO runtime and control loop

The simulation is launched through TraCI with:

- `--step-length 1`
- `--collision.action none`
- `--ignore-route-errors true`
- `--no-step-log true`
- `--duration-log.disable true`

Each simulation cycle does the following:

1. advance SUMO by one simulated second,
2. subscribe newly departed vehicles,
3. bulk-fetch vehicle states and neighborhood context,
4. compute per-edge density and per-vehicle risk,
5. probabilistically trigger at most one new incident,
6. update all active incidents,
7. reroute nearby traffic around active incidents when enabled,
8. record network performance snapshots at the configured interval.

This is a hybrid microscopic control loop producing macroscopic performance
outputs.

### 2.3 Risk model

The per-vehicle trigger score is a weighted composite of:

- speed risk using the Nilsson-style speed-power formulation,
- local speed heterogeneity relative to nearby vehicles,
- density risk,
- and a road-type multiplier.

The trigger rule is:

```text
effective_probability
  = base_probability
  × (1 + 10 × (risk_score − trigger_threshold))
  × secondary_multiplier
```

where the Bernoulli draw is only sampled once the threshold is exceeded.

Important improvement from the first review:

- density is now computed in vehicles per lane-km rather than vehicles per
  edge-km.

That correction materially improves comparability between one-lane and
multi-lane links.

### 2.4 Incident injection and lifecycle

Once a vehicle is selected:

- the incident vehicle is frozen with `traci.vehicle.setSpeed(..., 0.0)`,
- safety checks are disabled for that vehicle with `speedMode = 0`,
- a severity tier is sampled,
- severity determines duration, response time, secondary-risk footprint, and
  remaining capacity.

The incident can be applied in three modes:

- `speed_limit`
- `lane_closure`
- `hybrid`

In the default `hybrid` mode:

- some lanes are explicitly closed using `traci.lane.setDisallowed`,
- remaining lanes receive reduced `maxSpeed`,
- blocked lanes stay closed through the active phase,
- open-lane speeds recover gradually during clearing,
- and nearby vehicles are rerouted with `traci.vehicle.rerouteTraveltime`.

This is a substantial improvement over the earlier “one stopped vehicle plus
single-lane slowdown” approach because the new treatment creates:

- explicit spatial capacity loss,
- link-level operational degradation,
- and a limited route-choice response.

### 2.5 Output indicators

The main network outputs now include:

- mean network speed,
- throughput,
- speed-ratio-based delay proxy,
- active accident count,
- accident-level queue and affected-vehicle counts,
- blocked-lane and rerouted-vehicle counts,
- Antifragility Index (AI).

In transportation-performance terms, the simulator now captures both
state-based indicators and incident-event indicators, which makes it much more
appropriate for resilience analysis than before.

## 3. What improved materially since the first review

- The incident effect model now has a discrete operational footprint.
- Density normalization is now methodologically sounder on multi-lane links.
- Local rerouting makes the response more network-oriented rather than purely
  queue-oriented.
- The `1 s` runtime is much better for incident onset, queue growth, and
  recovery timing.
- The reporting layer now exposes blocked-lane and rerouting information,
  which improves interpretability.

The earlier simulator was useful mainly as a stochastic disruption toy model.
The current version is closer to a defensible incident-impact simulator for
macroscopic system-performance experiments.

## 4. Specific note on `1 s` step length for macroscopic AI

For your use case, the right interpretation is:

- the AI itself is macroscopic,
- but the disturbance generation process is still microscopic,
- so the quality of the macroscopic metric depends on the quality of the
  microscopic shock formation.

That is why `1 s` still has value even if the final analysis target is
system-wide rather than vehicle-level. It is not being used because the
analysis objective is microscopic. It is being used because the incident
process and queue propagation that feed the macroscopic indicators still occur
at the microscopic level.

Practical recommendation:

- keep `1 s` as the scientific reference configuration,
- then benchmark `2 s` or `5 s` against it for AI, throughput, queue, and
  recovery outputs,
- and only use the coarser setup for high-volume sweeps if the deviation is
  acceptably small.

## 5. Recommended next improvements

### Priority 1: make rerouting path-aware

This is now the most important model-logic improvement after the hybrid
incident upgrade.

Best options:

- reroute only vehicles whose remaining route includes the affected edge or a
  corridor downstream of it,
- or use SUMO rerouters / rerouting devices with explicit adaptation policy.

### Priority 2: calibrate the hybrid severity model

The next step should be calibration rather than adding more complexity.

Key targets:

- blocked-lane distributions by severity,
- clearance-time distributions,
- residual discharge capacity on open lanes,
- corridor-type differences between arterials, highways, and local streets.

### Priority 3: improve demand realism

This remains the single biggest scientific upgrade available.

Best candidates:

- O-D matrix import,
- TAZ-based trip generation,
- time-of-day demand profiles,
- class-specific demand composition.

### Priority 4: broaden the resilience measurement package

Keep AI, but add:

- vehicle-hours lost,
- travel-time increase by route or corridor,
- throughput recovery time,
- queue dissipation time,
- corridor-level and district-level recovery indicators.

### Priority 5: refine the risk-model semantics

Recommended cleanups:

- rename “speed variance” to “speed heterogeneity”,
- use network class where available rather than only speed limit,
- consider explicit conflict exposure near intersections if the network
  supports it,
- calibrate the risk weights to observed incident frequencies if data exist.

### Priority 6: formalize two official operating modes

Document two supported analysis modes:

- `scientific_reference`
  - `step_length: 1`
  - hybrid incident effects
  - rerouting enabled
- `batch_exploration`
  - coarser step only after validation against the reference
  - same incident logic
  - same output metrics

This would make the runtime-versus-fidelity tradeoff explicit and reproducible.

## Bottom line

After the second review, SUMA can now be described as:

- a stochastic incident-generation layer,
- coupled to a SUMO microscopic traffic simulation,
- producing macroscopic resilience and recovery indicators.

That is a defensible architecture for system-wide incident-resilience studies.
Its main remaining scientific limitations are now less about SUMO control and
more about:

- demand realism,
- rerouting realism,
- calibration of capacity-loss effects,
- and the breadth of the resilience metrics.

## Official SUMO references used

- SUMO TraCI vehicle state changes: https://sumo.dlr.de/docs/TraCI/Change_Vehicle_State.html
- SUMO TraCI lane state changes: https://sumo.dlr.de/docs/TraCI/Change_Lane_State.html
- SUMO routing and `rerouteTraveltime`: https://sumo.dlr.de/docs/Simulation/Routing.html
- SUMO automatic rerouting devices: https://sumo.dlr.de/docs/Demand/Automatic_Routing.html
- SUMO safety and collision handling: https://sumo.dlr.de/daily/userdoc/Simulation/Safety.html
- SUMO time-step semantics: https://sumo.dlr.de/docs/Simulation/Basic_Definition.html
- SUMO `randomTrips.py`: https://sumo.dlr.de/docs/Tools/Trip.html
