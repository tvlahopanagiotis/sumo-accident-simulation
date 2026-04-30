# SUMA / WP5 Codex Development Instructions

## Purpose of this file

This file provides persistent development context for Codex or any AI coding assistant working on the AntifragiCity WP5 / SUMA codebase.

It should be treated as the main project-development reference until more detailed WP3, WP4, and WP6 deliverables are available. As those deliverables mature, this file should be extended rather than replaced.

---

## 1. Project context: AntifragiCity

AntifragiCity is a Horizon Europe project focused on antifragile urban mobility systems. The project aims to help cities understand their normal mobility state, detect and characterise disruptions, assess impacts through simulation and prediction, evaluate mobility triage responses, and support better decision-making under stress.

The project is not only about resilience in the sense of returning to baseline after a disruption. It is also about antifragility: the ability of an urban mobility system to learn from shocks, adapt, and potentially improve after disruption.

The project uses four main demonstration contexts:

- Bratislava
- Larissa
- Odessa
- Thessaloniki / AHEPA hospital area

SUMA, developed under WP5, is one of the central technical assets of the project.

---

## 2. WP5 role: SUMA

WP5 is responsible for developing SUMA: the Simulator for Urban Mobility Antifragility.

The current Rhoé understanding is that SUMA should be developed as an integration and orchestration environment, not as a collection of disconnected scripts or one-off simulations.

SUMA should connect:

- WP2 outputs: event taxonomy, urban event catalogue, ontology, equilibrium concepts, antifragility indicators, KPI framework.
- WP3 outputs: mobility triage logic, decision-support requirements, vulnerability and acceptability information, once available.
- WP4 outputs: traffic control methods, algorithms, control strategies, modelling components, and simulation logic, once available.
- WP6 / WP11 pilot inputs: city-specific networks, demand data, disruption scenarios, local parameters, and validation feedback.

The core engineering objective is to build a structured, reusable, API-driven system that can support pilot-specific simulation and analysis workflows.

---

## 3. Scope boundary: what SUMA should and should not do

### SUMA should do

SUMA should:

- Provide a coherent API and software layer for configuring, executing, and evaluating urban mobility antifragility simulations.
- Accept structured disruption events and scenarios.
- Accept city-specific network, demand, and parameter inputs.
- Integrate WP4 methods and algorithms once delivered.
- Use WP2 ontology, event, KPI, and equilibrium concepts as semantic and analytical foundations.
- Produce mobility, resilience, and antifragility metrics.
- Support comparison of baseline, disrupted, response, and improved/adaptive scenarios.
- Be modular enough to work with different cities and different levels of data availability.
- Be extensible as WP3, WP4, and WP6 outputs mature.

### SUMA should not do, unless explicitly required later

SUMA should not assume responsibility for:

- Developing entirely new traffic-control algorithms if WP4 is expected to deliver them.
- Defining pilot-specific policy measures without input from pilot partners and WP6.
- Inventing city-specific congestion zones, restrictions, or control parameters without validated local data.
- Hard-coding pilot-specific logic into the core system.
- Treating preliminary WP2 theoretical models as fully calibrated operational models.
- Depending on unavailable datasets as mandatory for all workflows.

Where inputs are missing, SUMA should expose clear placeholders, validation errors, mock adapters, or degraded-mode functionality.

---

## 4. Relationship between WP4 and WP5

The working boundary is:

- WP4 develops generic methods, algorithms, traffic-control strategies, and modelling components.
- WP5 integrates, adapts, parameterises, and operationalises those components inside SUMA.

In implementation terms:

- WP4 components should ideally be wrapped behind interfaces or adapters.
- SUMA should not depend on a specific WP4 implementation format until WP4 confirms whether outputs will be code, scripts, APIs, models, reports, or datasets.
- Use abstractions such as `ControlStrategy`, `SimulationEngine`, `DemandProvider`, `KpiCalculator`, `EventAdapter`, and `ScenarioRunner`.
- Avoid direct coupling between the core API and a single simulation or optimisation implementation.

---

## 5. WP2 inputs currently available

The following WP2 deliverables are currently relevant for SUMA development.

### 5.1 D2.1 Urban Event Mapping

D2.1 provides the event taxonomy and baseline event catalogue.

Key concepts to encode:

#### Event domains

- `transport_related`
- `environment_weather`
- `utilities_connectivity`
- `public_space_social`

#### Event scales

- `daily_stressor`
- `mid_scale_disruption`
- `large_scale_crisis`

#### Severity levels

Use a 1-5 severity scale:

- 1: very low / informational
- 2: low / background
- 3: moderate / standard
- 4: high / priority
- 5: critical / immediate

#### Mobility impact dimensions

Events may affect:

- Passenger mobility
- Freight mobility
- Infrastructure
- Public transport
- Road network performance
- Accessibility
- Emergency access
- Equity-sensitive groups

#### Development implications

Implement event-related models with:

- A stable event schema.
- Extensible controlled vocabularies.
- Optional provenance/source fields.
- Support for temporal and spatial scope.
- Support for severity, scale, and affected mobility dimensions.
- Compatibility with ontology concepts from D2.5.

Do not hard-code only the 76 known events. The event catalogue should be treated as seed data, not as a closed universe.

---

### 5.2 D2.3 Establishing Urban States of Equilibrium

D2.3 provides a theoretical framework for antifragile urban mobility equilibrium.

Important concepts:

- State representation of the urban mobility system.
- Baseline / equilibrium state.
- Disrupted state.
- Response state.
- Improved / antifragile state.
- Antifragility Factor, abbreviated as `AF`.
- System Responsiveness Index, abbreviated as `SRI`.
- Disturbance mapping, including effective disturbance `D_eff(t)`.
- Performance indicators and target-setting logic.
- Traffic assignment, modal choice, and infrastructure-service response layers.

#### Development implications

SUMA should not treat D2.3 as fully calibrated software. It should treat it as a theoretical model specification.

Recommended implementation approach:

- Create interfaces for equilibrium and antifragility calculations.
- Allow placeholder implementations until real calibration data is available.
- Keep formulas and coefficients configurable.
- Store assumptions and calibration metadata with model runs.
- Distinguish between theoretical, mock, calibrated, and validated model modes.
- Never silently return an antifragility score without documenting input quality and calculation mode.

Potential objects:

- `SystemState`
- `EquilibriumState`
- `DisruptionState`
- `ScenarioState`
- `AntifragilityAssessment`
- `AntifragilityFactor`
- `SystemResponsivenessIndex`
- `DisturbanceProfile`
- `PerformanceVector`

---

### 5.3 D2.5 Ontology

D2.5 provides the semantic backbone for AntifragiCity.

It defines an ontology / knowledge graph approach using concepts such as:

- Transport elements
- Mobility services
- Disruption events
- System states
- Scenarios
- Environment factors
- Actors
- Governance instruments
- Spatial units
- Temporal units
- Observations
- KPIs
- Data provenance
- Response actions
- Learning processes

#### Development implications

SUMA should align internal data models with ontology concepts where practical.

The codebase should:

- Use ontology-compatible naming where possible.
- Support future RDF / JSON-LD / knowledge graph export.
- Preserve identifiers for events, scenarios, KPIs, actors, and spatial units.
- Include provenance and uncertainty fields in major data structures.
- Keep semantic mapping logic separate from core simulation logic.

Potential modules:

- `suma/ontology/`
- `suma/semantic_mapping/`
- `suma/kg_export/`
- `suma/schemas/ontology_aligned/`

Recommended formats:

- JSON Schema or Pydantic models for API validation.
- Optional JSON-LD context files for semantic interoperability.
- RDF export later, if required by the project architecture.

---

### 5.4 D2.7 AntifragiCity KPIs and Framework Development

D2.7 provides the KPI framework and confirms that KPI selection must be adaptive, modular, and sensitive to data availability.

Important conclusions:

- The framework should be adaptive and dynamic, not a fixed list of mandatory indicators.
- Cities differ significantly in the KPIs they can provide.
- Practitioners selected only part of the theoretical indicator pool.
- Complex composite antifragility indicators may be difficult to implement in practice due to data constraints.
- The KPI framework is connected to the AntifragiCity ontology and SUMA API.

#### Development implications

SUMA must implement KPIs as configurable, data-aware components.

KPI logic should support:

- Core vs extended KPIs.
- Availability checks.
- Data-quality metadata.
- Quantitative and qualitative inputs.
- Scenario comparison.
- KPI grouping by theme.
- Future lifecycle / LCA integration if required.

Potential KPI groups:

- Network / system performance
- Travel time and reliability
- Accessibility
- Safety
- Equity
- Environmental impact
- Resilience
- Antifragility
- Governance / institutional readiness, if later needed

Recommended objects:

- `KpiDefinition`
- `KpiInputRequirement`
- `KpiObservation`
- `KpiResult`
- `KpiSet`
- `KpiAvailabilityReport`
- `ScenarioKpiComparison`

Avoid forcing all pilots to provide the same data. Implement graceful degradation and clear reporting of missing inputs.

---

## 6. Expected future inputs from WP3, WP4, and WP6

### WP3 expected inputs

WP3 is expected to contribute mobility triage and decision-support logic.

Future additions may include:

- Triage categories.
- Response prioritisation logic.
- Equity and vulnerability assessment rules.
- Public acceptability constraints.
- Governance and justice considerations.
- DSS integration requirements.

Placeholder interfaces should be created where useful, but do not invent final WP3 logic prematurely.

Possible future modules:

- `suma/triage/`
- `suma/vulnerability/`
- `suma/decision_support/`
- `suma/acceptability/`

### WP4 expected inputs

WP4 is expected to provide traffic control, modelling, algorithmic, and strategy components.

Future additions may include:

- Control strategies.
- Traffic light control methods.
- Routing or re-routing algorithms.
- Simulation logic.
- Demand and OD matrix methods.
- Learning algorithms.
- Pre-assessment outputs.
- Operational scenarios.

Design all WP4 integration through adapters.

Possible future modules:

- `suma/control/`
- `suma/wp4_adapters/`
- `suma/traffic_assignment/`
- `suma/routing/`
- `suma/signal_control/`

### WP6 / pilot expected inputs

WP6 and pilot partners are expected to provide or validate city-specific data and scenarios.

Expected input types:

- Road network files.
- Public transport data.
- OD matrices or demand proxies.
- Traffic counts / sensor data.
- Disruption scenarios.
- Local constraints.
- Emergency routes.
- Policy restrictions.
- Validation feedback.

SUMA should support inconsistent data maturity across pilots.

---

## 7. Core architectural principles

### 7.1 Modularity

Each major function should be isolated:

- Events
- Scenarios
- Simulation
- KPIs
- Ontology mapping
- WP3 triage
- WP4 strategy integration
- Pilot configuration
- API layer

### 7.2 API-first design

SUMA should expose functionality through clear APIs.

Use REST-style endpoints unless the project later selects another interface.

Potential endpoints:

- `POST /events/validate`
- `POST /scenarios`
- `GET /scenarios/{id}`
- `POST /simulations/run`
- `GET /simulations/{id}`
- `GET /simulations/{id}/status`
- `GET /simulations/{id}/results`
- `GET /simulations/{id}/kpis`
- `POST /kpis/calculate`
- `POST /ontology/export`
- `GET /health`

### 7.3 Adapter-based integration

External tools should be wrapped:

- SUMO or other simulation engines.
- WP4 algorithms.
- Knowledge graph / ontology tools.
- Data ingestion pipelines.
- Pilot data providers.

Do not mix third-party tool calls directly into API route handlers.

### 7.4 Configuration over hard-coding

All pilot-specific details should be configuration-driven.

Examples:

- City name and IDs.
- Network file paths.
- Simulation time windows.
- Demand assumptions.
- KPI sets.
- Event profiles.
- Control strategies.
- Data availability.

### 7.5 Traceability

Every simulation result should be traceable to:

- Input event(s).
- Scenario definition.
- Network and demand version.
- Model versions.
- KPI definitions.
- Calculation mode.
- Timestamp.
- Data quality assumptions.

### 7.6 Data quality awareness

Because pilot data may be incomplete, data quality must be explicit.

Use fields such as:

- `source`
- `provenance`
- `confidence`
- `resolution_spatial`
- `resolution_temporal`
- `is_observed`
- `is_synthetic`
- `is_estimated`
- `validation_status`

---

## 8. Suggested repository structure

```text
suma/
  api/
    routes/
    dependencies/
    schemas/
  core/
    config.py
    errors.py
    logging.py
  events/
    models.py
    taxonomy.py
    validation.py
    catalogue.py
  scenarios/
    models.py
    builder.py
    validation.py
  simulation/
    engine.py
    runner.py
    adapters/
      sumo_adapter.py
      mock_adapter.py
  kpis/
    definitions.py
    calculators.py
    availability.py
    comparison.py
  ontology/
    mapping.py
    jsonld.py
    export.py
  equilibrium/
    models.py
    antifragility.py
    sri.py
    disturbance.py
  triage/
    interfaces.py
    placeholder.py
  control/
    interfaces.py
    wp4_placeholder.py
  pilots/
    bratislava/
    larissa/
    odessa/
    thessaloniki_ahepa/
  data/
    seed/
    schemas/
  tests/
    unit/
    integration/
    fixtures/
  docs/
    architecture.md
    api.md
    data_models.md
    assumptions.md
```

This is a suggested structure. Follow the actual repository if it already exists, but preserve these conceptual separations.

---

## 9. Data model guidance

### 9.1 Event model

Minimum fields:

```json
{
  "id": "event_001",
  "name": "Localised flooding on corridor",
  "domain": "environment_weather",
  "scale": "mid_scale_disruption",
  "severity": 4,
  "start_time": "2026-01-01T08:00:00Z",
  "end_time": "2026-01-01T12:00:00Z",
  "spatial_scope": {},
  "affected_modes": ["road", "public_transport"],
  "affected_mobility": {
    "passenger": true,
    "freight": false,
    "infrastructure": true
  },
  "source": "manual_or_catalogue_or_sensor",
  "provenance": {},
  "confidence": 0.8
}
```

### 9.2 Scenario model

A scenario should combine:

- Baseline configuration.
- One or more events.
- Network and demand inputs.
- Response strategy.
- Simulation engine.
- KPI set.
- Assumptions.

### 9.3 Simulation run model

A simulation run should include:

- `run_id`
- `scenario_id`
- `status`
- `created_at`
- `started_at`
- `finished_at`
- `engine`
- `engine_version`
- `input_hash`
- `results_uri`
- `logs_uri`
- `errors`

### 9.4 KPI result model

A KPI result should include:

- `kpi_id`
- `name`
- `value`
- `unit`
- `scenario_id`
- `run_id`
- `spatial_scope`
- `temporal_scope`
- `calculation_method`
- `required_inputs`
- `missing_inputs`
- `data_quality`
- `confidence`

---

## 10. Scenario types SUMA should support

### Baseline scenario

Represents normal operating conditions.

### Disruption scenario

Represents the system under one or more disruptions.

### Response scenario

Represents the system under a mitigation or triage response.

### Adaptation / antifragility scenario

Represents a post-learning or improved system state.

### Stress-test scenario

Represents intentionally generated events used to test limits of the system.

This is important because SUMA may need to support both:

- Bottom-up workflows, where real or externally defined events are ingested.
- Top-down workflows, where synthetic or exploratory stress-test scenarios are generated.

The bottom-up workflow is currently safer as the primary assumption. The top-down workflow should be supported as an extensible capability but marked as needing consortium validation.

---

## 11. Coding standards

### General

- Prefer explicit types.
- Keep functions small and testable.
- Avoid hidden global state.
- Validate all external inputs.
- Make assumptions visible in code and documentation.
- Do not silently ignore missing data.
- Use meaningful domain names from the project.

### Python recommendations, if Python is used

- Use Pydantic for data validation if available.
- Use FastAPI for the API layer if the repo has no framework yet.
- Use pytest for testing.
- Use structured logging.
- Use enums for controlled vocabularies.
- Use dataclasses or Pydantic models for internal domain objects.

### Error handling

Create explicit domain errors, for example:

- `InvalidEventTaxonomyError`
- `MissingPilotInputError`
- `UnsupportedSimulationEngineError`
- `KpiInputUnavailableError`
- `OntologyMappingError`
- `ScenarioValidationError`

---

## 12. Testing strategy

### Unit tests

Cover:

- Event taxonomy validation.
- Scenario validation.
- KPI availability checks.
- KPI calculations using small fixtures.
- Antifragility placeholder calculations.
- Ontology mapping functions.

### Integration tests

Cover:

- Creating a scenario.
- Running a mock simulation.
- Calculating KPIs from mock outputs.
- Exporting results.

### Fixture strategy

Use small synthetic test data for:

- A tiny network.
- A simple OD matrix.
- One disruption event.
- One response action.
- Baseline vs disrupted KPI outputs.

Do not require full pilot datasets for normal test runs.

---

## 13. Documentation requirements

Maintain documentation for:

- API endpoints.
- Data schemas.
- Assumptions.
- Known limitations.
- Input requirements.
- KPI definitions.
- Scenario examples.
- Pilot configuration examples.

Whenever Codex creates a new module, it should also update relevant documentation or add docstrings.

---

## 14. Current limitations and uncertainties

The following must remain explicit in the codebase:

1. WP4 outputs are not yet fixed.
   - Unknown whether they will be executable code, APIs, scripts, models, reports, or datasets.

2. WP3 triage logic is not yet available.
   - Do not invent final triage rules.

3. Pilot data availability is uncertain.
   - Support partial data and validation reports.

4. D2.3 models are theoretical.
   - Calibration and empirical validation are deferred.

5. D2.7 shows KPI implementation constraints.
   - Do not assume all theoretical KPIs are computable in every city.

6. D2.5 ontology is broad.
   - SUMA should align with it pragmatically, without blocking development on full semantic infrastructure.

7. Bottom-up vs top-down scenario generation requires clarification.
   - Implement extensible architecture, but label synthetic scenario generation as provisional.

---

## 15. Development priorities

### Priority 1: Foundation

- Establish data models for events, scenarios, simulation runs, KPIs, and pilot configuration.
- Encode D2.1 taxonomy.
- Create API skeleton.
- Create mock simulation workflow.
- Create KPI availability and result structures.

### Priority 2: SUMO / simulation integration

- Add simulation adapter interface.
- Add SUMO adapter if SUMO is selected.
- Keep mock adapter for tests.
- Support baseline and disrupted scenario execution.

### Priority 3: KPI and antifragility evaluation

- Implement simple KPI calculators.
- Implement scenario comparison.
- Add placeholder AF/SRI calculations with explicit mode flags.
- Add data quality reporting.

### Priority 4: Ontology and interoperability

- Add ontology-aligned identifiers.
- Add JSON-LD export if useful.
- Add mapping tables between SUMA objects and ontology classes.

### Priority 5: WP3/WP4 integration

- Add triage and control strategy adapters once deliverables are available.
- Replace placeholders with validated components.

---

## 16. Codex behaviour instructions

When modifying this repository, Codex should:

1. Read this file first.
2. Preserve the WP5 scope boundary.
3. Avoid implementing speculative final logic for WP3 or WP4.
4. Prefer modular abstractions over hard-coded workflows.
5. Keep mock implementations where real components are unavailable.
6. Add tests for new behaviour.
7. Update documentation when adding public interfaces.
8. Keep pilot-specific configuration separate from core logic.
9. Make data-quality and missing-input behaviour explicit.
10. Avoid pretending that theoretical models are empirically validated.

---

## 17. Extension placeholders

### To be extended after WP3 deliverables

Add:

- Mobility triage taxonomy.
- Response prioritisation rules.
- Vulnerability methodology.
- DSS interface requirements.
- Public acceptability constraints.
- Governance / justice constraints.

### To be extended after WP4 deliverables

Add:

- Concrete control strategy interfaces.
- Algorithm integration details.
- Traffic control inputs and outputs.
- OD matrix handling.
- Learning algorithm integration.
- Strategy evaluation logic.

### To be extended after WP6 / pilot data maturity

Add:

- Pilot-specific data schemas.
- Network file formats.
- Validated scenario catalogues.
- Local constraints.
- Calibration notes.
- Demonstration-specific KPIs.

---

## 18. Working definition of success

A successful early SUMA implementation is not a fully validated digital twin. It is a robust, extensible, API-driven integration environment that can:

1. Represent AntifragiCity disruption events.
2. Configure baseline and disruption scenarios.
3. Run simulations through mock or real adapters.
4. Compute or report availability of KPIs.
5. Compare baseline, disrupted, and response scenarios.
6. Preserve traceability, assumptions, and data quality.
7. Incorporate WP3, WP4, and pilot inputs as they become available.

