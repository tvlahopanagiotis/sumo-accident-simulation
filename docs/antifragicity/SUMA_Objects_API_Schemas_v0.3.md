# SUMA Objects, API, And Schema Catalogue v0.3

Status: internal technical catalogue for D5.1/T5.2  
Date context: 12 May 2026  
Purpose: formal candidate object model, API contract, schema conventions, calculation modules, adapter strategy, and technology stack for SUMA.

This document is software-facing. The deliverable analyses explain how each source informs WP5. This catalogue lists what SUMA may implement, stub, validate, expose, or defer.

## 1. Contract Principle

Not every scientific concept becomes a software object. Promote a concept only when a selected use case, partner module, UI/report, simulator adapter, validation workflow, or D5.1 acceptance criterion needs to call, persist, or exchange it.

| Layer | Implementation |
|---|---|
| `methodology` | Documentation, UI warning, validation caveat, meeting decision. |
| `evidence` | Provenance, confidence, source reference, evidence status. |
| `data_contract` | Pydantic DTO, JSON Schema, JSON-LD-compatible metadata. |
| `api_contract` | Endpoint, adapter, job lifecycle, request/response payload. |
| `governance` | Owner, due date, permission, risk, acceptance status. |
| `deferred` | Roadmap item with owner/trigger, not D5.1 commitment. |

## 1.1 Source-To-Object Traceability

Objects and endpoints in this catalogue are candidates only when they trace to a source or a confirmed meeting decision.

| Source basis | Object/API family driven | Required caveat |
|---|---|---|
| Grant Agreement | `ComponentManifest`, `AdapterContract`, D5.1/T5.2 staging, role/audit assumptions. | Contractual scope does not prove partner interfaces are ready. |
| D2.1 | `DisruptionEvent`, taxonomy/crosswalk endpoints, scenario event attachment. | Event severity/source mappings stay provisional until pilot/WP6 validation. |
| D2.2/D2.4 | `AcceptabilityConstraint`, `EquityImpact`, `PriorityObjective`, `RecommendationExplanation`, `MonitoringPlan`. | Social evidence gates recommendations; it is not an automated universal score. |
| D2.3 | `SystemState`, `KpiVector`, `TargetPerformance`, `SRIResult`, `AFValidationRecord`. | Calculation examples are not validated AF claims without baselines/windows/owners. |
| D2.5 | `OntologyMetadata`, `ResourceReference`, JSON-LD context, semantic-lift jobs. | Full KG/reasoning is deferred until CU confirms artefacts and use cases. |
| D2.6 | `RequirementDefinition`, `AcceptanceCriterion`, `VerificationEvidence`, `RequirementGate`, requirements endpoints. | Consensus status is not implementation readiness. |
| D2.7 | `KpiDefinition`, `KpiSelection`, `KpiObservation`, `KpiEvidenceLedger`, thresholds/baselines. | KPI rows need formula, unit, denominator, owner, baseline/threshold, and verification. |
| MoM/context/Mini-GA | `TraceabilityChain`, `DecisionRiskLog`, `ControlZone`, adapter/pilot configuration rows. | Promote only after owner, status, due date, and fallback are recorded. |

Any object without a source basis should be removed, relabelled as an internal proposal, or converted into a Mini-GA question.

## 2. Technology Stack

| Layer | Recommended stack |
|---|---|
| Backend/API | Python >=3.10, FastAPI, Pydantic v2, Uvicorn, OpenAPI. |
| Simulation | SUMO/TraCI or libsumo first; Vissim/Aimsun/VISUM as adapter contracts until live access/licence/support are confirmed. |
| Calculation | NumPy, Pandas, SciPy; NetworkX for network metrics; GeoPandas/Shapely only if spatial processing grows. |
| Frontend | React 18, TypeScript, Vite, Leaflet, Recharts; preserve existing GUI conventions. |
| Persistence | File/JSON/YAML plus SQLite metadata ledger first; PostgreSQL/PostGIS only if multi-pilot geospatial/concurrent deployment needs it. |
| Jobs | Current local `JobManager` as D5.1 backend; Celery/RQ plus Redis only for durable long-running jobs in T5.2. |
| Ontology | JSON-LD context and mappings first; `rdflib`/`pySHACL` later; triple store only after CU confirms query/use cases. |
| Deployment | Docker Compose with `api` and `frontend`; optional `worker`, `redis`, `postgis` profile later. |

The `/api/v1` contract should live beside the current GUI/operator API. Do not rewrite existing dashboard endpoints just to satisfy D5.1. Preserve existing `/api/jobs`, `/api/results/*`, `/api/files/text`, `/api/configs`, and related GUI endpoints as compatibility surfaces.

## 3. Architecture Modules

| Module | Responsibility |
|---|---|
| `suma.api.v1` | D5.1 routers, OpenAPI tags, response envelopes, examples, error model. |
| `suma.contracts` | Pydantic schemas/enums for D5.1 objects. |
| `suma.storage` | Repository layer over files/results plus SQLite metadata ledger. |
| `suma.jobs` | Abstract job lifecycle; current local job manager first, durable queue later. |
| `suma.adapters` | `SimulatorAdapter`, `SUMOAdapter`, and placeholder external adapters. |
| `suma.ontology` | Class registry, JSON-LD context, ID/IRI resolution, semantic mapping status. |
| `suma.requirements` | D2.6 requirement registry, acceptance criteria, traceability, verification evidence, requirement gates. |
| `suma.validation` | Schema validation, data-role rules, AF claim gates, deployability checks. |
| `suma.kpis` | KPI registry, selection, observation storage, derived calculations. |
| `suma.methods` | D2.3 calculation modules and WP4/WP3 adapter wrappers. |
| `suma.gui` | Existing operator dashboard API and UI integration. |

## 4. Object Staging

| Stage | Meaning | Objects |
|---|---|---|
| `core_d5_1` | Needed for D5.1 API spec and initial tests. | `RequirementDefinition`, `AcceptanceCriterion`, `DisruptionEvent`, `Scenario`, `SimulationJob`, `SimulationRun`, `KpiDefinition`, `KpiObservation`, `PilotConfig`, `DataInventoryItem`, `AdapterContract` |
| `prototype_stub` | Define schema now; implementation may be mock/placeholder. | `ResponseAction`, `AcceptabilityConstraint`, `EquityImpact`, `TraceabilityChain`, `ControlZone`, `PriorityObjective`, `DataSourceContract` |
| `integration_t5_2` | Needed mainly for implementation/integration. | `SimulatorAdapter`, `ComponentManifest`, `SemanticLiftJob`, `SystemState`, `KpiVector`, `TargetPerformance`, `SRIResult` |
| `ui_d5_3` | Needed mainly for role-based UI/dashboard/export. | `RolePermission`, `DashboardPayload`, `ReportExport`, `RecommendationExplanation` |
| `validation_wp6_wp10` | Needed for pilot validation and feedback. | `VerificationEvidence`, `RequirementGate`, `EvaluationSession`, `StakeholderInput`, `LearningArtifact`, `AFValidationRecord` |
| `deferred` | Useful but not first-version commitment. | automated acceptability scoring, full KG reasoning, live vendor-simulator integrations |

In this catalogue, `core` means "must be specified in D5.1 and should have examples/tests." It does not automatically mean fully implemented before T5.2. Implementation status must remain explicit on each endpoint or module.

## 5. Common Schema Conventions

Use `snake_case`, ISO-8601 UTC datetimes, stable prefixed IDs, GeoJSON geometry, explicit versions, and provenance on all exchanged records.

```yaml
CommonMetadata:
  id: string
  schema_version: string
  external_ids: object
  iri: string | null
  ontology_class: string | null
  semantic_version: string | null
  source_refs: string[]
  evidence_origin: string | null
  evidence_status: string | null
  owner: string | null
  created_at: datetime
  updated_at: datetime
  validation_status: string
  implementation_status: implemented | mock | placeholder | external_dependency | deferred
  data_quality: object | null
  confidence: number | null
  assumptions: string[]
  limitations: string[]
```

Standard response envelope:

```json
{
  "data": {},
  "meta": {
    "schema_version": "0.3",
    "trace_id": "..."
  },
  "links": {}
}
```

Standard error envelope:

```json
{
  "error": {
    "code": "SCHEMA_VALIDATION_FAILED",
    "message": "Payload failed validation",
    "details": [],
    "trace_id": "..."
  }
}
```

## 6. Core Schemas

### 6.1 `DisruptionEvent`

```yaml
DisruptionEvent:
  id: string
  event_domain: transport | environment_weather | utilities_connectivity | public_space_social
  event_subcategory: string
  event_scale: daily | mid | large
  severity_level: integer | null
  severity_basis: string | null
  start_time: datetime | null
  end_time: datetime | null
  spatial_unit_id: string | null
  geometry: GeoJSON | null
  affected_transport_elements: string[]
  affected_modes: string[]
  affected_groups: string[]
  impact_pathway: string[]
  source_type: survey | social_media | em_dat | copernicus | operator_feed | sensor | manual | synthetic | simulation
  source_id: string | null
  source_limitations: string[]
  taxonomy_version: string
  crosswalk_version: string | null
  confidence: number | null
  validation_status: unvalidated | partner_validated | pilot_validated | operationally_validated
```

### 6.2 `Scenario`

```yaml
Scenario:
  id: string
  pilot_id: string
  scenario_type: baseline | disrupted | intervention | recovery | forecast | stress_test
  generation_mode: observed | historical_replay | synthetic_stress_test | forecast | manual
  event_refs: string[]
  baseline_scenario_id: string | null
  intervention_set_id: string | null
  spatial_scope: object
  temporal_scope: object
  assumptions: string[]
  data_requirements: string[]
  kpi_selection_id: string | null
  pilot_config_id: string
  status: draft | validated | executable | executed | archived
```

### 6.3 `SimulationJob` And `SimulationRun`

```yaml
SimulationJob:
  job_id: string
  scenario_id: string
  adapter_id: string
  requested_by: string
  runtime_mode: local | service | external | manual_import
  input_payload_ref: string
  status: queued | running | completed | failed | cancelled
  created_at: datetime
  started_at: datetime | null
  completed_at: datetime | null
  logs_ref: string | null
  error: object | null

SimulationRun:
  run_id: string
  job_id: string | null
  scenario_id: string
  adapter_id: string
  input_refs: string[]
  output_refs: string[]
  kpi_observation_refs: string[]
  system_state_refs: string[]
  provenance: object
  validation_status: string
```

### 6.4 `OntologyMetadata`

```yaml
OntologyMetadata:
  iri: string | null
  ontology_class: string
  city_code: string | null
  city_graph: string | null
  local_id: string | null
  external_ids: object
  semantic_version: string
  schema_version: string
  source_ref: string | null
  provenance: object
  quality: object | null
  access_level: public | internal | restricted | confidential | unknown
  redaction_status: none | redacted | aggregated | synthetic | unknown
  mapping_status: mapped | partially_mapped | unmapped | not_applicable
```

### 6.5 `KpiDefinition`, `KpiSelection`, And `KpiObservation`

```yaml
KpiDefinition:
  kpi_id: string
  version: string
  label: string
  group: string
  subgroup: string | null
  definition: string
  unit: string | null
  directionality: higher_is_better | lower_is_better | target_band | descriptive
  denominator: string | null
  exposure_basis: string | null
  population_basis: string | null
  formula_ref: string | null
  data_role: input | computed_output | parameter
  ipoo_stage: input | process | output | outcome | unknown
  kpi_layer: layer_1_observed_city | layer_2_equilibrium_variable | layer_3_suma_platform
  ontology_refs: string[]
  owner: string | null
  availability_status: available | proxy | missing | deferred | not_applicable | unknown
  maturity_status: defined | method_tbc | target_tbc | baseline_tbc | validation_tbc | deprecated

KpiSelection:
  id: string
  pilot_id: string
  use_case_id: string
  selected_kpi_ids: string[]
  tier: top_22_practitioner | common_64 | full_209_catalogue | af_required_override | pilot_selected | deferred
  selection_reason: string
  coverage_ratio: number | null
  owner: string | null

KpiObservation:
  observation_id: string
  kpi_id: string
  pilot_id: string
  use_case_id: string | null
  scenario_id: string | null
  run_id: string | null
  value: number | string | null
  value_type: scalar | categorical | boolean | object
  value_object: object | null
  normalized_value: number | null
  unit: string | null
  denominator_value: number | null
  denominator_unit: string | null
  exposure_basis: string | null
  timestamp: datetime | null
  temporal_window: object | null
  spatial_scope: object | null
  mode_scope: string[] | null
  population_scope: string[] | null
  baseline_ref: string | null
  provenance: object
  calculation_mode: observed | simulated | derived | survey | sentiment | lca | proxy | synthetic | theoretical
  proxy_flag: boolean
  confidence: number | null
  completeness: number | null
  accuracy: number | null
  timeliness: number | null
  validation_status: string
```

### 6.6 `SystemState`, `KpiVector`, And D2.3 Calculation Records

```yaml
SystemState:
  id: string
  pilot_id: string
  scenario_id: string
  timestamp: datetime
  topology_version: string
  network_ref: string
  demand_ref: string | null
  capacity_vector_ref: string | null
  flow_vector_ref: string | null
  control_state_ref: string | null
  kpi_vector_id: string | null
  model_maturity: theoretical | benchmark | pilot_calibration | validation_ready | operational_advisory | operational_actuating_blocked | proxy

KpiVector:
  id: string
  state_id: string
  M: object | null
  R: object | null
  A: object | null
  C: object | null
  Deff: object | null
  S: object | null
  Q: object | null
  P: object | null
  B: object | null
  Eenergy: object | null
  EICT: object | null
  I: object | null
  coverage_flag: full_12_kpi | core_6_kpi | minimal_3_kpi | proxy_only | missing_required
  missing_indicators: string[]
  normalization_version: string
```

### 6.7 `ResponseAction`, `AcceptabilityConstraint`, And `EquityImpact`

```yaml
ResponseAction:
  action_id: string
  action_type: control | routing | emergency_access | public_transport | information | policy_restriction | repair | learning
  target_disruption_refs: string[]
  target_assets: string[]
  responsible_actor: string | null
  required_inputs: string[]
  expected_outputs: string[]
  technical_priority: low | medium | high | critical | not_assessed
  social_acceptability_status: not_assessed | indicative_only | acceptable | contested | unacceptable
  equity_status: not_assessed | neutral | improves | worsens | mitigation_required
  mitigation_status: none_required | mitigation_required | mitigation_defined
  public_rationale: string | null
  reversibility: reversible | partially_reversible | irreversible | unknown
  approval_status: draft | proposed | approved | rejected | emergency_fast_path

AcceptabilityConstraint:
  id: string
  pilot_id: string
  action_type: string
  intervention_category: string | null
  affected_groups: string[]
  evidence_origin: citizen_forum | survey | pilot_partner | deliverable | social_media | assumption | substituted
  evidence_status: not_assessed | indicative_only | assessed | conflicting_evidence | missing | substituted
  forum_id: string | null
  sample_n: integer | null
  selection_method: string | null
  representativity_note: string | null
  priority_method: ranked | consensus | unranked_convergence | inferred | substituted
  acceptability_status: not_assessed | acceptable | contested | unacceptable | mitigation_required
  mitigation_required: boolean
  validation_status: string

EquityImpact:
  id: string
  scenario_id: string
  response_action_id: string | null
  affected_neighbourhoods: string[]
  zone_ids: string[]
  affected_groups: string[]
  vulnerable_groups_affected: string[]
  baseline_metric: object | null
  post_action_metric: object | null
  burden_delta: object | null
  accessibility_impact: object | null
  safety_impact: object | null
  digital_exclusion_risk: low | medium | high | unknown
  mitigation_required: boolean
  mitigation_actions: string[]
  mitigation_owner: string | null
  monitoring_required: boolean
```

### 6.8 `PilotConfig`, `ControlZone`, And `DataInventoryItem`

```yaml
PilotConfig:
  pilot_id: string
  pilot_name: string
  study_area: GeoJSON | object
  network_refs: string[]
  od_refs: string[]
  simulator_candidates: string[]
  protected_assets: string[]
  critical_routes: string[]
  control_zones: string[]
  data_readiness: object[]
  communication_channels: string[]
  user_roles: string[]
  validation_target: string | null
  permission_constraints: string[]
  degraded_mode_assumptions: string[]
  owners: object

ControlZone:
  id: string
  pilot_id: string
  geometry: GeoJSON | null
  road_refs: string[]
  protected_asset_refs: string[]
  validity_horizon: string | null
  owner: string | null
  confidence: number | null
  retraining_flag: boolean
  boundary_assumptions: string[]

DataInventoryItem:
  variable_id: string
  name: string
  description: string
  role: kpi | data | model | config | unknown
  data_role: input | computed_output | parameter
  priority: high | medium | low
  availability: available | effortful | low_probability | missing | unknown
  owner: string | null
  source_system: string | null
  proxy_option: string | null
  spatial_scale: string | null
  temporal_scale: string | null
  privacy_constraint: string | null
  prototype_relevance: required | useful | later | not_needed
```

### 6.9 Governance And Integration Records

These records are needed because D5.1 must specify interfaces and ownership, while T5.2 implements or integrates them.

```yaml
RequirementDefinition:
  requirement_id: string
  category: FR | DR | IR | UR | NFR | GR
  title: string
  statement: string
  rationale: string
  priority: must | should | could | deferred
  source_delphi_ids: string[]
  consensus_status: high | moderate | low | engineering | unknown
  source_reference: string
  d5_1_status: contract | stub | dependency | deferred
  t5_2_status: not_started | implemented | blocked | deferred | external_dependency
  owner: string | null
  due_date: date | null
  fallback: string | null

AcceptanceCriterion:
  criterion_id: string
  requirement_id: string
  verification_method: test | api_test | demo | inspection | stakeholder_review | performance_test
  given: string
  when: string
  then: string
  evidence_required: string[]
  target_value: string | null
  degraded_mode_allowed: boolean
  acceptance_status: draft | agreed | passed | failed | deferred

VerificationEvidence:
  evidence_id: string
  requirement_id: string
  criterion_id: string
  evidence_type: api_response | schema_validation | run_log | ui_screenshot | export_file | review_minutes | performance_report
  artifact_ref: string
  produced_at: datetime
  produced_by: string | null
  result: pass | fail | partial | not_run
  caveats: string[]

RequirementGate:
  gate_id: string
  requirement_id: string
  gate_type: owner_confirmed | data_available | interface_confirmed | privacy_checked | latency_target_defined | validation_method_agreed
  status: open | satisfied | blocked | deferred
  owner: string | null
  due_date: date | null
  if_unresolved_then: stub | proxy | exclude_from_demo | defer_to_backlog

RequirementTrace:
  requirement_id: string
  source: grant_agreement | d2_6 | miniga | partner_decision
  statement: string
  d5_1_contract_refs: string[]
  t5_2_implementation_refs: string[]
  verification_method: test | demonstration | inspection | stakeholder_review | performance_test
  acceptance_criterion: string
  owner: string | null
  status: essential | deferred | rejected | owner_missing

AdapterContract:
  adapter_id: string
  module_name: string
  provider: string
  module_type: common_core | local_adapter | external_module
  d5_1_status: specified | stubbed | dependency | deferred
  t5_2_status: not_started | implemented | external_dependency | blocked | deferred
  handover_format: code | service | script | api | report | unknown
  runtime_mode: local | service | manual_import | unknown
  input_schema_ref: string | null
  output_schema_ref: string | null
  kpi_mapping: object | null
  licence_or_nda_constraint: string | null
  owner: string | null
  fallback: string | null

ComponentManifest:
  component_id: string
  component_name: string
  provider: string
  owner: string | null
  licence_status: confirmed | pending | restricted | unknown
  interface_status: documented | draft | missing | nda_blocked
  maturity_status: concept | prototype | tested | validated | operational | unknown
  d5_1_status: implement | stub | dependency | defer
  fallback: string | null

TraceabilityChain:
  chain_id: string
  pilot_id: string
  use_case_id: string
  requirement_ids: string[]
  ontology_classes: string[]
  data_objects: string[]
  kpi_ids: string[]
  suma_functions: string[]
  endpoint_or_service: string | null
  architecture_module: string | null
  ui_roles: string[]
  owner: string | null
  status: decision | assumption | dependency | risk | owner_missing | deferred
  fallback: string | null

KpiEvidenceLedger:
  kpi_id: string
  formula: string | null
  dataset: string | null
  denominator: string | null
  baseline_status: defined | proxy | owner_missing | not_defined
  threshold_status: defined | proposed | owner_missing | not_defined
  confidence: number | null
  verification_method: string | null
  owner: string | null

DecisionRiskLog:
  item: string
  classification: decision | assumption | dependency | risk | owner_missing | deferred
  owner: string | null
  next_action: string
  due_date: date | null
  if_unresolved_then: string
  d5_1_implication: string
  t5_2_implication: string

RolePermission:
  role: string
  visible_data: string[]
  allowed_actions: string[]
  export_permissions: string[]
  warnings_required: string[]

EvaluationSession:
  session_id: string
  pilot_id: string
  user_roles: string[]
  validation_targets: string[]
  feedback: object
  owner: string | null

AuditRecord:
  audit_id: string
  actor: string
  action: string
  resource_ref: string
  timestamp: datetime
  rationale: string | null
  assumptions: string[]
  provenance_refs: string[]
```

## 7. D2.6 Requirements Registry And Acceptance Gates

D2.6 makes requirements first-class SUMA contracts. D5.1 should not only describe API endpoints; it should show which requirement each endpoint, object, module, UI role, and test supports.

Baseline D2.6 groups:

| Group | D5.1 object/API implications | T5.2 implementation implications |
|---|---|---|
| FR-01 to FR-05 | Event, scenario, KPI/AF, intervention, and API-first contracts. | SUMO-first workflow plus external API examples. |
| FR-06/FR-07 | Runtime mode, latency target, learning artefact schemas. | Implement only with confirmed feeds/feedback workflows. |
| DR-01/DR-02 | Data inventory, source diversity, provenance, uncertainty, quality flags. | Data readiness and output warnings. |
| IR-01/IR-02 | Adapter contracts, simulator registry, component manifest, versioning. | SUMO connector first; external tools staged. |
| UR-01/UR-02 | Role permissions, dashboard payloads, comparison/export semantics. | Role-based UI workflows and dashboards. |
| NFR-01 to NFR-03 | Documentation, performance target, change/feedback tracking. | Docs page evidence, latency tests, agile feedback records. |
| GR-01 to GR-04 | Privacy, minimisation, audit, transparency, stakeholder input governance. | RBAC, retention, audit logs, governed input workflows. |

Example requirement contract:

```yaml
RequirementDefinition:
  requirement_id: DR-02
  category: DR
  title: Data quality metadata
  priority: must
  source_delphi_ids:
    - DEL-R22
    - DEL-R20
  consensus_status: moderate_used_as_core_due_to_trust_requirement
  d5_1_status: contract
  t5_2_status: not_started
  owner: Rhoe/CU to confirm
  fallback: expose missing quality metadata as warning

AcceptanceCriterion:
  criterion_id: ac_dr_02_quality_flags
  requirement_id: DR-02
  verification_method: test
  given: ingested dataset with complete and missing provenance fields
  when: SUMA returns scenario outputs
  then: provenance, timestamp, source limitations, and missing-quality warnings are included
  evidence_required:
    - stored_metadata_record
    - api_output_payload
    - ui_warning_screenshot
```

## 8. D2.3 Equations And Modules

D5.1 should expose these as illustrative calculation contracts, not hard-coded approved equations and not validated operational claims. Each formula, weight, window, source reliability value, and threshold must carry `parameter_status: default | assumed | partner_confirmed | pilot_calibrated | missing`.

```text
V = u(S), V in [0,1]^m
x_hat_higher_is_better = (x - xmin) / (xmax - xmin)
x_hat_lower_is_better = (xmax - x) / (xmax - xmin)
x_hat_target_band = band_score(x, lower_bound, target_min, target_max, upper_bound)

E+ = a1*M + a2*R + a3*A + a4*Hnorm + w1*Eenergy + w2*EICT + zeta*I + pi*P + mu*B
E- = b1*Deff + b2*S + theta*Q
Ebar = lambda*E+ - (1-lambda)*E-
Etarget = Ebar - lambda + 1

Hnorm = C / ln(NOD)
E[t+1] = E[t] - kappa * (E[t] - Ebar)

Deff(e,t) = w_src(src) * exp(-(t-te)/tau) * delta_domain * ((L-1)/4)
Deff(t) = min(1.0, sum_e Deff(e,t))

mu30 = (1/30) * sum_{t=31..60} E(t)
AF = mu30 / Ebase
AFreq = 1 + 0.2*(1-Ebase)
AFcrit = 1 + k*sigma_resid
AF decision = LCL0.95(AF) >= max(AFreq, AFcrit)
```

Candidate modules:

- `state_snapshot`,
- `kpi_normalizer`,
- `disturbance_mapper`,
- `target_performance_calculator`,
- `entropy_calculator`,
- `equilibrium_run_adapter`,
- `sri_calculator`,
- `af_validation_batch`,
- `feasible_set_validator`,
- `coverage_maturity_labeler`.

Do not claim validated antifragility, real-time AF scoring, automated permanent interventions, causal attribution, or black-swan generalisation unless pilot evidence and governance checks exist.

Example: illustrative AHEPA emergency-access KPI. This is a schema example, not pilot-confirmed evidence.

```yaml
KpiDefinition:
  kpi_id: emergency_access_travel_time
  version: v0_illustrative
  label: Emergency access travel time
  unit: minutes
  directionality: lower_is_better
  denominator: emergency_trip
  exposure_basis: simulated_ambulance_route
  data_role: computed_output
  kpi_layer: layer_1_observed_city
  maturity_status: method_tbc

KpiObservation:
  observation_id: obs_ahepa_access_time_mock_001
  kpi_id: emergency_access_travel_time
  pilot_id: ahepa
  scenario_id: scenario_ahepa_flood_intervention_001
  value: 8.4
  normalized_value: null
  unit: minutes
  denominator_value: 1
  denominator_unit: simulated_emergency_trip
  exposure_basis: synthetic_route_until_living_lab_permission
  calculation_mode: simulated
  proxy_flag: true
  confidence: 0.5
  validation_status: unvalidated
```

## 9. D2.7 KPI Selection And Quality Formulas

These are SUMA operational formulas for selection/data-readiness support, not D2.7-validated scientific claims.

```text
selection_score =
  wr*relevance +
  wf*feasibility +
  wi*integrability +
  wa*actionability +
  waf*af_required

coverage_ratio = available_required_kpis / required_kpis

data_quality = 0.4*completeness + 0.4*accuracy + 0.2*timeliness
```

Weights must be configurable and labelled as `default`, `assumed`, or `pilot_calibrated`.

## 10. API Families

Status terms in the tables below mean:

| Status | Meaning |
|---|---|
| `D5.1 contract` | Specify schema/path/examples/errors; implementation may be stubbed. |
| `D5.1 stub` | Include placeholder endpoint or documented mock behaviour. |
| `T5.2 implementation` | Actual working implementation/integration target. |
| `deferred` | Documented roadmap item, not current commitment. |

### 10.1 System And Specification

| Endpoint | Method | Status | Purpose |
|---|---|---|---|
| `/api/health` | `GET` | existing | Service health. |
| `/api/v1/openapi.json` | `GET` | D5.1 contract | Machine-readable D5.1 contract. |
| `/api/v1/component-manifest` | `GET` | D5.1 contract | KER/module owner, licence, interface, maturity. |

### 10.2 Requirements And Traceability

| Endpoint | Method | Status | Purpose |
|---|---|---|---|
| `/api/v1/requirements` | `GET` | D5.1 contract | Return D2.6 baseline requirement registry. |
| `/api/v1/requirements/{id}` | `GET` | D5.1 contract | Return one requirement with priority, consensus, owner, and status. |
| `/api/v1/requirements/{id}/trace` | `GET` | D5.1 contract | Return linked objects, endpoints, tests, UI roles, and evidence. |
| `/api/v1/requirements/{id}/gates` | `POST` | D5.1 stub | Record owner/data/interface/privacy/performance gates. |
| `/api/v1/acceptance-criteria` | `GET` | D5.1 contract | Return D2.6 verification methods and acceptance criteria. |
| `/api/v1/verification-evidence` | `POST` | D5.1 stub, T5.2 implementation | Register test/demo/review evidence for a requirement. |
| `/api/v1/traceability-chains` | `GET` | D5.1 contract | Query use-case-to-requirement-to-object-to-endpoint chains. |

### 10.3 Events And Taxonomy

| Endpoint | Method | Status | Purpose |
|---|---|---|---|
| `/api/v1/disruption-events` | `POST` | core | Ingest D2.1-aligned event. |
| `/api/v1/disruption-events/validate` | `POST` | core | Validate event taxonomy, source, space/time, severity. |
| `/api/v1/taxonomies/event-domains` | `GET` | core | Return supported domains/scales/severity labels. |
| `/api/v1/taxonomies/crosswalks` | `GET` | core | Return source-category mappings. |
| `/api/v1/scenarios/{id}/events` | `POST` | D5.1 contract, T5.2 implementation | Attach events/stressors to a scenario. |

### 10.4 Pilots And Data Readiness

| Endpoint | Method | Status | Purpose |
|---|---|---|---|
| `/api/v1/pilot-configurations` | `GET/POST` | core | Manage pilot configuration. |
| `/api/v1/pilot-configurations/{pilot_id}` | `GET` | core | Retrieve pilot config. |
| `/api/v1/data-inventory-items` | `GET/POST` | core/internal | Track data variables and roles. |
| `/api/v1/data-readiness` | `GET` | core/internal | Summarise readiness and gaps. |
| `/api/v1/priority-objectives` | `POST` | D5.1 stub | Define pilot/use-case objective priorities. |
| `/api/v1/data-source-contracts` | `GET` | D5.1 contract | Expose source access/privacy/acceptability status. |

### 10.5 Scenarios, Jobs, Runs, And Adapters

| Endpoint | Method | Status | Purpose |
|---|---|---|---|
| `/api/v1/scenarios` | `POST` | core | Create scenario. |
| `/api/v1/scenarios/{id}` | `GET` | core | Retrieve scenario. |
| `/api/v1/scenarios/{id}/validate` | `POST` | core | Validate scenario executability/readiness. |
| `/api/v1/simulation-jobs` | `POST` | core | Dispatch run. |
| `/api/v1/simulation-jobs/{id}` | `GET` | core | Check async status. |
| `/api/v1/simulation-jobs/{id}/cancel` | `POST` | core | Cancel if supported. |
| `/api/v1/simulation-runs/{id}` | `GET` | core | Retrieve run outputs. |
| `/api/v1/simulator-adapters` | `GET/POST` | T5.2 implementation | Registry/capabilities. |
| `/api/v1/adapter-contracts` | `GET/POST` | core/internal | Register external module contracts. |

### 10.6 KPI And Antifragility

| Endpoint | Method | Status | Purpose |
|---|---|---|---|
| `/api/v1/kpi-definitions` | `GET` | core | KPI registry. |
| `/api/v1/kpi-selections` | `POST` | core/prototype | Pilot/use-case KPI subset. |
| `/api/v1/kpi-observations` | `POST` | core | KPI values. |
| `/api/v1/kpi-calculation-jobs` | `POST` | staged | Derived KPI calculation. |
| `/api/v1/scenarios/{id}/kpis` | `GET` | core | Scenario KPI outputs. |
| `/api/v1/thresholds` | `POST` | D5.1 stub | Store threshold metadata/status. |
| `/api/v1/baselines` | `POST` | D5.1 stub | Store baseline metadata/status. |
| `/api/v1/kpi-evidence-ledger` | `GET` | D5.1 contract | Expose formula, dataset, denominator, confidence, verification method. |
| `/api/v1/af-validations` | `POST` | staged | Batch AF validation workflow. |
| `/api/v1/sri/calculate` | `POST` | staged | Advisory SRI calculation. |

### 10.7 Response, Governance, And Acceptability

| Endpoint | Method | Status | Purpose |
|---|---|---|---|
| `/api/v1/response-actions` | `POST` | prototype | Candidate action catalogue. |
| `/api/v1/triage-recommendations` | `POST` | placeholder | WP3 handoff. |
| `/api/v1/acceptability-assessments` | `POST` | optional/stub | Acceptability evidence/status. |
| `/api/v1/equity-impacts` | `POST` | optional/stub | Distributional impacts. |
| `/api/v1/audit-records` | `POST` | prototype | Decision/action audit. |
| `/api/v1/monitoring-plans` | `POST` | D5.1 stub | Define post-action monitoring/reporting requirements. |
| `/api/v1/evaluation-sessions` | `POST` | validation/stub | WP6 feedback. |

### 10.8 Ontology

| Endpoint | Method | Status | Purpose |
|---|---|---|---|
| `/api/v1/ontology/classes` | `GET` | core/spec | Supported D2.5 class subset. |
| `/api/v1/ontology/relations` | `GET` | spec | Supported relation subset. |
| `/api/v1/ontology/context.jsonld` | `GET` | core/spec | JSON-LD context. |
| `/api/v1/resources/{id}` | `GET` | prototype | Resolve local IDs/IRIs. |
| `/api/v1/semantic-lift-jobs` | `POST` | D5.1 stub, T5.2 implementation | Register semantic mapping jobs. |

## 11. Validation Rules

### 11.1 Data Role Rule

- `input`: can be requested from partners/cities.
- `computed_output`: should be produced by SUMA, WP3, WP4, or simulator methods.
- `parameter`: requires method/calibration owner and should not be requested as raw city data unless explicitly available.

### 11.2 Recommendation Rule

A response action cannot be labelled deployment-ready unless it has technical priority, owner, validation/assumption status, acceptability status or explicit `not_assessed`, equity status or explicit `not_assessed`, mitigation status where needed, and public rationale where user-facing.

### 11.3 AF Claim Rule

An antifragility output cannot be labelled validated unless it has baseline state, disturbance definition, post-event/intervention observation window, method version, KPI coverage tier, confidence/uncertainty, causal/attribution note, equity non-worsening status, and validation owner.

### 11.4 Requirement Traceability Rule

Every D5.1 endpoint family and core object should link to at least one D2.6 `RequirementDefinition` or an explicit non-D2.6 source such as Grant Agreement, Mini-GA decision, or partner interface dependency. A requirement cannot be marked `passed` unless its `AcceptanceCriterion` has corresponding `VerificationEvidence` or a documented deferral/fallback.

## 12. Implementation Phases

| Phase | Work |
|---|---|
| D5.1 contract baseline | Add `/api/v1` schemas, examples, OpenAPI, error model, D2.6 requirement registry, static registries, and mock/stub endpoints. |
| D5.1 SUMO reference flow | Map `Scenario -> SimulationJob -> current runner -> SimulationRun -> KpiObservation`. |
| T5.2 integration | Introduce `SimulatorAdapter`, `AdapterContract`, `ComponentManifest`, SUMO adapter, and external placeholders. |
| T5.2 persistence hardening | SQLite ledger first; optional Redis/Celery and PostGIS profile later. |
| Ontology lift | JSON-LD context/class registry first; semantic-lift jobs second; RDF/SHACL only after ontology acceptance. |
| UI/validation | Role payloads, dashboard warnings, exports, evaluation sessions, stakeholder feedback. |
