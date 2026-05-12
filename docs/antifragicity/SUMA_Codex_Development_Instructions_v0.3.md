# SUMA Codex Development Instructions v0.3

Status: internal development context  
Date context: 12 May 2026  
Primary audience: Rhoe/SUMA developers and researchers preparing D5.1, T5.2, and Mini-GA follow-up.  
Supersedes: practical working interpretation of `SUMA_Codex_Development_Instructions_v0.1.md` and `SUMA_Codex_Development_Instructions_v0.2.md`, updated with the v0.3 deliverable analyses, Mini-GA working guide, and SUMA object/API catalogue.

## 1. Purpose

This document is the master internal development context for SUMA. It tells Codex and Rhoe developers how to interpret AntifragiCity material when converting scientific deliverables, meeting outputs, partner handoffs, and pilot evidence into software architecture, API contracts, schemas, implementation backlog, tests, and documentation.

The document is not a consortium-approved deliverable. It is an internal operational guide. It should be used to keep D5.1 and T5.2 scientifically defensible, technically buildable, and explicit about assumptions.

## 2. Core Interpretation Of SUMA

SUMA is the WP5 API-driven orchestration layer for AntifragiCity. It connects:

- WP2 event mapping, social acceptability, ontology, KPI framework, and requirements.
- WP3 triage, response actions, FMEA/risk logic, and decision-support outputs.
- WP4 traffic-control, routing, control-zone, and method outputs.
- WP6 and WP11 pilot validation, living lab evidence, and user feedback.
- DMO role-based UI, dashboards, exports, and user-facing workflows.
- SUMO and current SUMA simulation workflows, with external simulators staged through adapters.

SUMA should not be treated as:

- the owner of final WP3 triage science,
- the owner of WP4 traffic-control algorithms,
- a fully validated antifragility engine before pilot calibration,
- a complete ontology/knowledge-graph rule engine in D5.1,
- a universal simulator platform before vendor access/licences/support are confirmed,
- the owner of city data acquisition,
- the certifier of public acceptability or policy legitimacy.

## 3. Translation Layers

Every source finding must be classified before it becomes development work.

| Layer | Meaning | Implementation form |
|---|---|---|
| `methodology` | Principle, interpretation rule, or scientific framing. | Documentation, validation caveat, UI warning, meeting decision. |
| `evidence` | Deliverable evidence, survey/forum result, caveat, or pilot context. | Provenance, evidence status, confidence, report text. |
| `data_contract` | Object/field that must be represented consistently. | Pydantic schema, JSON Schema, JSON-LD-compatible metadata. |
| `api_contract` | Callable or persisted boundary. | Endpoint, adapter, job lifecycle, request/response payload. |
| `governance` | Owner, permission, acceptance, due date, risk, decision. | Backlog row, decision log, component manifest. |
| `deferred` | Useful but not first-version work. | Roadmap item with owner and trigger condition. |

Do not promote methodology/evidence into API unless a selected use case, partner module, UI/report, simulator adapter, validation workflow, or D5.1 acceptance criterion needs to call, persist, or exchange it.

## 4. Authoritative Working Documents

Use these files as the current v0.3 working set:

| File | Role |
|---|---|
| `docs/antifragicity/SUMA_Codex_Development_Instructions_v0.3.md` | Master developer/researcher context. |
| `docs/antifragicity/Rhoe_MiniGA_Thessaloniki_Working_Guide_v0.3.md` | Rhoe meeting guide and partner question bank. |
| `docs/antifragicity/SUMA_Objects_API_Schemas_v0.3.md` | Technical object/API/schema catalogue. |
| `docs/antifragicity/deliverable_analyses/*.md` | Source-specific D5.1/T5.2/Mini-GA input briefs. |
| `docs/antifragicity/SUMA_Codex_Development_Instructions_v0.1.md` | Historical initial context. |
| `docs/antifragicity/SUMA_Codex_Development_Instructions_v0.2.md` | Historical expanded context. |

The `docs/antifragicity/miniGA/` files are coordinator-preparation material. Use them to seed discussion, not as final commitments. Only rows confirmed during/after the Mini-GA with owner, status/fallback, and due date should become D5.1/T5.2 scope.

## 5. D5.1 Acceptance Definition

For internal planning, D5.1 should be considered acceptable if it includes:

- `/api/v1` OpenAPI specification or equivalent schema snapshot.
- JSON Schema/Pydantic models for `core_d5_1` objects.
- JSON-LD-compatible semantic metadata and an ontology context path.
- Request/response examples for core endpoint families.
- Error model and validation response model.
- Async job lifecycle model for simulation jobs.
- SUMO reference flow or documented stub/reference adapter.
- Adapter contract template for WP3, WP4, CUSP, DMO, and external simulators.
- KPI definition/observation semantics with provenance, confidence, validation status, and data role.
- Explicit implementation status for every endpoint: `implemented`, `mock`, `placeholder`, `external_dependency`, or `deferred`.
- Initial tests or testable examples for schemas and the reference flow.

D5.1 should not claim:

- full pilot validation,
- full WP3/WP4 integration,
- full knowledge graph/rule execution,
- live Vissim/Aimsun/VISUM integration unless confirmed,
- validated real-time antifragility scoring,
- automated social acceptability scoring,
- operational deployment readiness.

## 6. T5.2 Development Boundary

T5.2 should implement the D5.1 contract in stages:

| Stage | Development target |
|---|---|
| Contract baseline | `/api/v1` routers, schemas, examples, validation errors, static registries. |
| SUMO reference flow | `Scenario -> SimulationJob -> current runner -> SimulationRun -> KpiObservation`. |
| Adapter registry | `AdapterContract`, `ComponentManifest`, `SimulatorAdapter`, `SUMOAdapter`, external placeholders. |
| Persistence | File-first artifacts plus SQLite metadata ledger; PostGIS only if spatial/concurrent needs justify it. |
| Ontology | JSON-LD context and class registry first; semantic-lift jobs later. |
| KPI/AF methods | KPI registry/observation store first; D2.3 calculations as maturity-labelled modules. |
| UI bridge | Role payloads, warnings, exports, dashboards, and guided/expert workflow support. |

## 7. Technology Stack Baseline

Keep the current stack and formalize it. Do not pivot platforms before D5.1.

| Layer | Recommendation |
|---|---|
| Backend/API | Python >=3.10, FastAPI, Pydantic v2, Uvicorn, OpenAPI. |
| Simulation | SUMO/TraCI or libsumo first; Vissim/Aimsun/VISUM as adapter contracts until confirmed. |
| Calculations | NumPy, Pandas, SciPy; NetworkX for network metrics; GeoPandas/Shapely/PostGIS only when spatial needs grow. |
| Frontend | React 18, TypeScript, Vite, Leaflet, Recharts. |
| Persistence | File/JSON/YAML plus SQLite metadata ledger first; PostgreSQL/PostGIS only if multi-pilot geospatial/concurrent use requires it. |
| Jobs | Existing local job manager for D5.1; Celery/RQ plus Redis only for durable long-running jobs in T5.2. |
| Ontology | JSON-LD context and mappings first; `rdflib`/`pySHACL` later; triple store only after CU confirms query/use cases. |
| Deployment | Docker Compose for RP1; optional `worker`, `redis`, and `postgis` profile later. |
| Testing | pytest, ruff, mypy; add API schema tests and example payload tests. |

## 8. Current Repository Baseline

The current app already has:

- FastAPI backend in `src/suma/gui/app.py`.
- Existing GUI/operator endpoints under `/api/*`.
- Workflow/job management via `src/suma/gui/jobs.py`.
- SUMO runner and simulation tooling under `src/suma/simulation/`.
- Analysis tooling under `src/suma/analysis/`.
- Data/config/result file layout under `configs/`, `data/`, and `results/`.
- React/Vite frontend under `frontend/`.
- Docker Compose with `api` and `frontend`.

D5.1 `/api/v1` endpoints should live beside the existing GUI/operator API. Do not break current endpoints used by the web app. Preserve `/api/jobs`, `/api/results/*`, `/api/files/text`, `/api/configs`, `/api/docs`, and current workflow endpoints unless there is an explicit migration plan.

## 9. Recommended Package Evolution

Add new modules incrementally rather than rewriting existing code.

| Proposed module | Responsibility |
|---|---|
| `suma.api.v1` | D5.1 routers, OpenAPI tags, response envelopes, examples, errors. |
| `suma.contracts` | Pydantic schemas/enums for D5.1 objects. |
| `suma.storage` | File/SQLite repositories for scenarios, jobs, KPIs, provenance, adapters. |
| `suma.jobs` | Abstract job lifecycle over current `JobManager`; durable backend later. |
| `suma.adapters` | `SimulatorAdapter`, `SUMOAdapter`, placeholder external adapters. |
| `suma.ontology` | Ontology class registry, JSON-LD context, ID/IRI resolver, mapping status. |
| `suma.validation` | Schema rules, data-role rules, AF gates, recommendation gates. |
| `suma.kpis` | KPI registry, selection, observations, derived calculations. |
| `suma.methods` | D2.3 calculators and WP3/WP4 adapter wrappers. |

## 10. Core Object Staging

| Stage | Objects |
|---|---|
| `core_d5_1` | `DisruptionEvent`, `Scenario`, `SimulationJob`, `SimulationRun`, `KpiDefinition`, `KpiObservation`, `PilotConfig`, `DataInventoryItem`, `AdapterContract`. |
| `prototype_stub` | `ResponseAction`, `AcceptabilityConstraint`, `EquityImpact`, `TraceabilityChain`, `ControlZone`, `PriorityObjective`, `DataSourceContract`. |
| `integration_d5_2` | `SimulatorAdapter`, `ComponentManifest`, `SemanticLiftJob`, `SystemState`, `KpiVector`, `TargetPerformance`, `SRIResult`. |
| `ui_d5_3` | `RolePermission`, `DashboardPayload`, `ReportExport`, `RecommendationExplanation`. |
| `validation_wp6_wp10` | `EvaluationSession`, `StakeholderInput`, `LearningArtifact`, `AFValidationRecord`. |
| `deferred` | Automated acceptability scoring, full KG reasoning, live vendor-simulator integrations. |

## 11. Source-Specific Development Rules

### 11.1 Grant Agreement

Use it for contractual scope, risks, deliverable boundaries, KER/component integration, and acceptance staging. Do not turn all GA text into software features.

Development rule: every broad GA promise must be translated into a staged status: `implemented`, `stub`, `dependency`, or `deferred`.

Example:

```yaml
ComponentManifest:
  component_id: wp4_eth_control
  component_name: ETH traffic-control module
  provider: ETH
  interface_status: draft
  maturity_status: prototype
  d5_1_status: dependency
  fallback: manual_import_or_mock_strategy_output
```

### 11.2 D2.1 Urban Event Mapping

Use D2.1 as the event-normalisation baseline. It supplies event domain, scale, severity/urgency separation, source crosswalks, event catalogue fixtures, and the distinction between event, stressor, disruption, and impact.

Development rule: `DisruptionEvent` is core. `Stressor`, `DisruptionImpact`, and `EventDataGap` can be nested or separate depending on whether an endpoint or partner adapter needs them.

Example: AHEPA flood/storm scenario trigger, illustrative only.

```yaml
DisruptionEvent:
  id: event_ahepa_flood_access_001
  event_domain: environment_weather
  event_subcategory: intense_rainfall_flooding
  event_scale: mid
  severity_level: 4
  severity_basis: pilot_to_confirm
  spatial_unit_id: ahepa_approach_corridor
  affected_transport_elements:
    - road_link_to_ahepa_emergency_gate
  affected_modes:
    - ambulance
    - private_car
    - bus
  source_type: manual
  source_limitations:
    - illustrative_until_pilot_validated
  taxonomy_version: d2_1_provisional
  validation_status: unvalidated
```

### 11.3 D2.2 Social Acceptability

Use D2.2 as methodology/evidence for acceptability, equity, vulnerable groups, communication, and public rationale. Do not build a universal acceptability score.

Development rule: acceptability metadata becomes software only when a response action or UI/report needs to show readiness, warning, mitigation, or evidence status.

Example: Larissa rerouting action with unvalidated acceptability.

```yaml
AcceptabilityConstraint:
  id: acc_larissa_flood_reroute_001
  pilot_id: larissa
  action_type: routing
  affected_groups:
    - residents_near_reroute
    - pedestrians
  evidence_origin: deliverable
  evidence_status: indicative_only
  representativity_note: local_validation_required
  acceptability_status: mitigation_required
  mitigation_required: true
  validation_status: unvalidated
```

### 11.4 D2.3 Equilibrium And Antifragility

Use D2.3 for calculation contracts and maturity labels. Do not claim validated AF until baseline, disturbance, post-event window, confidence, causal/equity checks, and validation ownership are present.

Development rule: every D2.3-derived output must carry `model_maturity`, `coverage_flag`, `calculation_mode`, and caveats.

Illustrative calculation contracts:

```text
V = u(S), V in [0,1]^m
x_hat_higher_is_better = (x - xmin) / (xmax - xmin)
x_hat_lower_is_better = (xmax - x) / (xmax - xmin)
x_hat_target_band = band_score(x, lower_bound, target_min, target_max, upper_bound)
E+ = a1*M + a2*R + a3*A + a4*Hnorm + w1*Eenergy + w2*EICT + zeta*I + pi*P + mu*B
E- = b1*Deff + b2*S + theta*Q
Ebar = lambda*E+ - (1-lambda)*E-
Etarget = Ebar - lambda + 1
Deff(e,t) = w_src(src) * exp(-(t-te)/tau) * delta_domain * ((L-1)/4)
AF = mu_post / Ebase
```

These expressions are not hard-coded approved equations. Every weight, parameter, severity mapping, source reliability value, validation window, and decision threshold must carry `parameter_status: default | assumed | partner_confirmed | pilot_calibrated | missing`.

Example: degraded Larissa flood calculation.

```yaml
KpiVector:
  id: kv_larissa_flood_proxy_001
  state_id: state_larissa_flood_proxy_001
  M:
    calculation_mode: simulated
    confidence: 0.45
  S:
    calculation_mode: proxy
    confidence: 0.35
  Deff:
    calculation_mode: theoretical
    confidence: 0.30
  coverage_flag: minimal_3_kpi
  missing_indicators:
    - R
    - C
    - I
  normalization_version: provisional_v0
```

### 11.5 D2.4 Initial Analysis

Use D2.4 as provisional quantitative evidence for emergency priority, equity mitigation, explanation/reporting, communication channels, and trusted data sources. Do not hard-code survey percentages as universal thresholds.

Development rule: emergency access can be a configurable default priority objective, but the evidence must retain `sample_n`, `evidence_scope`, and `threshold_status`.

Example: AHEPA priority objective.

```yaml
PriorityObjective:
  id: priority_ahepa_emergency_access
  pilot_id: ahepa
  objective_type: emergency_access
  priority_rank: 1
  protected_assets:
    - ahepa_emergency_department
  protected_routes:
    - ahepa_primary_ambulance_corridor
  evidence_origin: survey
  evidence_scope: d2_4_initial_analysis
  threshold_status: proposed
  override_allowed: true
```

### 11.6 D2.5 Ontology

Use D2.5 for semantic contracts, not for full KG implementation in D5.1. D5.1 should define ontology-aligned DTOs, IDs, JSON-LD path, named graph/city graph metadata, semantic mapping status, and split KPI definitions from observations.

Development rule: use optional `iri` and stable `canonical_id` until CU confirms final IRI conventions.

Example: AHEPA corridor semantic reference.

```yaml
ResourceReference:
  canonical_id: ahepa_corridor_primary_001
  iri: null
  ontology_class: TransportElement
  label: AHEPA primary ambulance corridor
  city_code: thessaloniki
  local_id: ahepa_corridor_primary_001
  access_level: internal
  redaction_status: none
```

### 11.7 D2.7 KPI Framework

Use D2.7 as KPI registry and selection logic. Do not implement all 209 KPIs in D5.1. Protect AF-critical network/system KPIs through `af_required_override`, even if they are not in the practitioner top 22.

Development rule: never request a computed KPI as raw city data. Use `data_role` to separate `input`, `computed_output`, and `parameter`.

Example: AHEPA selected KPI observation.

```yaml
KpiObservation:
  observation_id: obs_ahepa_ambulance_access_time_001
  kpi_id: emergency_access_travel_time
  pilot_id: ahepa
  use_case_id: ahepa_flood_hospital_access
  scenario_id: scenario_ahepa_flood_intervention_001
  value: null
  value_type: scalar
  unit: minutes
  denominator_value: 1
  denominator_unit: simulated_emergency_trip
  exposure_basis: synthetic_route_until_living_lab_permission
  calculation_mode: simulated
  proxy_flag: true
  confidence: 0.50
  validation_status: unvalidated
```

### 11.8 WP5/WP4 Context And Mini-GA Files

Use context/MoM files to clarify boundaries, adapter contracts, control zones, CUSP readiness, and Mini-GA facilitation. Use Mini-GA files cautiously as preparation material.

Development rule: a Mini-GA row becomes scope only if it has owner, status/fallback, and due date.

Example: WP4 adapter dependency row.

```yaml
AdapterContract:
  adapter_id: eth_control_strategy_adapter
  module_name: ETH WP4 control strategy
  provider: ETH
  implementation_status: external_dependency
  handover_format: unknown
  runtime_mode: unknown
  input_schema_ref: tbc_by_eth
  output_schema_ref: tbc_by_eth
  retraining_assumptions:
    - tbc_control_zone_boundary_stability
  owner: ETH
```

## 12. D5.1 API Families

The D5.1 API should use `/api/v1` for partner-facing contracts and keep current `/api/*` GUI endpoints stable.

| Family | Core endpoints |
|---|---|
| System/spec | `GET /api/health`, `GET /api/v1/openapi.json`, `GET /api/v1/component-manifest` |
| Events/taxonomy | `POST /api/v1/disruption-events`, `POST /api/v1/disruption-events/validate`, `GET /api/v1/taxonomies/*` |
| Pilots/data | `GET/POST /api/v1/pilot-configurations`, `GET/POST /api/v1/data-inventory-items`, `GET /api/v1/data-readiness` |
| Scenarios | `POST /api/v1/scenarios`, `GET /api/v1/scenarios/{id}`, `POST /api/v1/scenarios/{id}/validate` |
| Jobs/runs/adapters | `POST /api/v1/simulation-jobs`, `GET /api/v1/simulation-jobs/{id}`, `GET /api/v1/simulation-runs/{id}`, `GET/POST /api/v1/adapter-contracts` |
| KPI/AF | `GET /api/v1/kpi-definitions`, `POST /api/v1/kpi-selections`, `POST /api/v1/kpi-observations`, `POST /api/v1/af-validations` |
| Response/governance | `POST /api/v1/response-actions`, `POST /api/v1/triage-recommendations`, `POST /api/v1/audit-records` |
| Ontology | `GET /api/v1/ontology/classes`, `GET /api/v1/ontology/context.jsonld`, `POST /api/v1/semantic-lift-jobs` |

## 13. Validation Rules

### 13.1 Data Role Rule

- `input`: may be requested from partners/cities.
- `computed_output`: must be produced by SUMA, simulator, WP3, WP4, or another method.
- `parameter`: requires method/calibration owner and should not be requested as raw city data unless explicitly available.

### 13.2 Recommendation Readiness Rule

A response action cannot be labelled deployment-ready unless it has:

- technical priority,
- owner,
- validation or assumption status,
- acceptability status or explicit `not_assessed`,
- equity status or explicit `not_assessed`,
- mitigation status where needed,
- public rationale where user-facing,
- audit/provenance.

### 13.3 Antifragility Claim Rule

An AF output cannot be labelled validated unless it has:

- baseline state,
- disturbance definition,
- post-event or post-intervention observation window,
- method version,
- KPI coverage tier,
- confidence/uncertainty,
- causal/attribution note,
- equity non-worsening status,
- validation owner.

Otherwise label it `theoretical`, `benchmark`, `proxy`, or `pilot_calibration`.

## 14. Mini-GA Conversion Rule

During and after Mini-GA, every retained use case should be converted into:

```text
UseCaseContract
-> RequirementTrace
-> Ontology/Data Map
-> KpiEvidenceLedger
-> SUMA Function
-> Endpoint/Service
-> Architecture Module
-> UI Role
-> Owner/Fallback
```

If any step is missing, classify the gap:

- `assumption`
- `dependency`
- `risk`
- `owner_missing`
- `deferred`

## 15. Example End-To-End Traceability Chain

Illustrative AHEPA hospital-access scenario:

| Step | Example value | Status |
|---|---|---|
| Use case | Flood/storm disrupts AHEPA emergency access. | illustrative |
| Requirement | FR-01 disruption representation, FR-02 what-if, FR-03 resilience/AF assessment, FR-04 intervention exploration. | to confirm |
| Ontology classes | `DisruptionEvent`, `TransportElement`, `Actor`, `SpatialUnit`, `ResponseAction`, `Observation`. | likely |
| Data objects | `DisruptionEvent`, `ControlZone`, `PriorityObjective`, `Scenario`, `SimulationJob`, `KpiObservation`. | schema-ready |
| KPI | ambulance access travel time, blocked link count, recovery time, equity burden if rerouting affects neighbourhoods. | owner needed |
| SUMA function | ingest event, configure scenario, run SUMO, compare KPI outputs, show readiness warnings. | D5.1/T5.2 |
| Endpoint | `/api/v1/disruption-events`, `/api/v1/scenarios`, `/api/v1/simulation-jobs`, `/api/v1/scenarios/{id}/kpis`. | candidate |
| Module | SUMO adapter plus KPI observation store. | SUMO-first |
| UI role | hospital duty manager, emergency responder, traffic operator. | D5.3 |
| Owner/fallback | AUTH/AHEPA for local data; Rhoe for SUMA contract; fallback synthetic demo if permissions unavailable. | to confirm |

## 16. Roadmap Through RP1

| Period | Target |
|---|---|
| Before Mini-GA | Use v0.3 guide/schema to prepare tables and partner questions. |
| Mini-GA Day 1 | Capture use cases, requirements, ontology/data, KPIs, living lab/stakeholder context. |
| Mini-GA Day 2 | Convert into functions, architecture, adapter contracts, triage/action rows, UI roles, backlog owners. |
| Immediately after Mini-GA | Consolidate confirmed rows into D5.1 outline, examples, and backlog. |
| June 2026 | Implement `/api/v1` schema/stub baseline and static registries. |
| July 2026 | Implement SUMO reference flow and KPI observation export. |
| August 2026 | Freeze D5.1 API specification, examples, tests, and known limitations. |
| September-October 2026 | Advance T5.2 integration, pilot fixtures, adapter contracts, and RP1 validation evidence. |

## 17. Coding Rules For Codex

When implementing SUMA changes:

- Inspect existing code before editing.
- Preserve existing GUI/operator API compatibility.
- Add new D5.1 schemas in a separate contract layer rather than overloading GUI schemas.
- Use Pydantic models and typed enums for API payloads.
- Keep examples and fixtures versioned and clearly labelled as illustrative, mock, or pilot-confirmed.
- Add validation logic for data role, event taxonomy, KPI status, and AF claim gates.
- Do not hard-code survey percentages, AF weights, severity mappings, or KPI thresholds unless a partner-approved source exists.
- Do not silently assign ownerless dependencies to Rhoe.
- Do not implement external modules as core logic; use adapter contracts.
- Prefer small, testable modules and explicit statuses over broad implicit claims.

## 18. Documentation Rules

When updating docs:

- Link each claim to a source deliverable, MoM, agenda, or Mini-GA confirmed row where possible.
- Distinguish `source evidence`, `internal interpretation`, and `implementation decision`.
- Add examples only when they are explicitly labelled as deliverable-derived, pilot-confirmed, or illustrative.
- Use `provisional`, `to confirm`, `stub`, `dependency`, or `deferred` when evidence is incomplete.
- Keep D5.1 language precise: specification, contract, schema, adapter, initial testing.
- Keep diplomatic versions separate from internal frank versions when needed.

## 19. Open Decisions

These must be resolved before D5.1 freeze or clearly marked as dependencies:

- Exact D5.1 acceptance level: spec only, spec plus stubs/tests, or executable prototype.
- Mandatory first-prototype ontology class/property subset.
- IRI/local-ID and JSON-LD expectations.
- Mini-GA-confirmed use cases per pilot.
- KPI subset, formula, unit, baseline, threshold, owner, and verification per selected KPI.
- D2.1 severity/source mapping into D2.3 `Deff`.
- D2.3 weights, parameters, maturity labels, and validation windows.
- WP3 response-action handoff format and triage timing.
- WP4 control/routing handoff format, control-zone assumptions, retraining assumptions, and KPI mapping.
- CUSP NDA, I/O, auth, deployment, latency, and fallback.
- DMO UI payloads, roles, warnings, exports, and guided/expert split.
- Pilot data owners, permission constraints, redaction rules, and validation targets.
- D5.4 versus WP10 feedback/maintainability ownership.

## 20. One-Line Development Principle

Build SUMA as a versioned, evidence-aware orchestration platform: schema first, SUMO reference flow second, partner adapters third, validation and UI maturity after confirmed pilot evidence.
