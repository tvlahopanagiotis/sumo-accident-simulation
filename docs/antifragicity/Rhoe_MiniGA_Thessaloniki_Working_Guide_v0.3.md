# Rhoe Mini-GA Thessaloniki Working Guide v0.3

Status: internal Rhoe working guide  
Date context: 12 May 2026  
Meeting: AntifragiCity Thessaloniki Mini-GA, 18-19 May 2026  
Primary use: prepare Rhoe to convert consortium discussion into D5.1 content, T5.2 development backlog, and owner-assigned Mini-GA decisions.

## 1. Rhoe Objective

Rhoe's role in the Mini-GA is to turn broad project material into buildable WP5 inputs. The meeting should not end with only scientific discussion or generic tables. It should produce traceable rows that can be used directly in D5.1 and T5.2:

```text
use case -> requirement -> ontology/data object -> KPI -> SUMA function -> endpoint/service -> architecture module -> UI role -> owner/fallback
```

The governing rule is simple: if an item does not need to be ingested, stored, computed, exposed, validated, or exchanged by SUMA, do not force it into an API object. Record it as methodology, evidence, validation context, governance, dependency, or deferred backlog.

## 2. Scope Position To Use With Partners

SUMA is the WP5 API-driven orchestration layer for AntifragiCity. It connects WP2 event/ontology/KPI/requirements work, WP3 triage and response actions, WP4 control/routing/method outputs, WP6 validation needs, pilot configurations, and DMO/UI needs.

SUMA is not the owner of final WP3 triage science, WP4 traffic-control algorithms, the complete knowledge graph/rule engine, all vendor simulator integrations, city data acquisition, public acceptability certification, or a validated antifragility engine before pilot calibration.

## 2.1 D5.1 Versus T5.2

Use this distinction consistently in the meeting:

| Item | Meaning for Rhoe |
|---|---|
| D5.1 | API specification, schemas, endpoint families, examples, stubs/placeholders, adapter contracts, error model, initial tests, and known limitations. |
| T5.2 | Actual implementation and integration: SUMO reference flow, persistence, adapters, KPI calculations, semantic-lift jobs, partner module handoffs, and hardening. |
| Mini-GA output | Owner-assigned rows that tell Rhoe what D5.1 must specify and what T5.2 must implement, defer, or treat as dependency. |

Partner-facing phrasing: "SUMA can represent and integrate this once the owner, interface, evidence status, and fallback are confirmed." Avoid saying that Rhoe refuses ownership; say the item needs an agreed owner and interface before it becomes a WP5 commitment.

## 3. Translation Layers

Use these labels while taking notes and closing sessions.

| Layer | Meaning | Example |
|---|---|---|
| `methodology` | Principle or interpretation rule. | Technical optimality does not imply social acceptability. |
| `evidence` | Source evidence, caveat, validation context. | D2.4 survey percentage or D2.2 citizen forum caveat. |
| `data_contract` | Object/field represented consistently. | `DisruptionEvent`, `KpiDefinition`, `PilotConfig`. |
| `api_contract` | Endpoint, adapter, job, request/response. | `POST /api/v1/simulation-jobs`. |
| `governance` | Owner, permission, decision, due date, acceptance. | CUSP NDA status or KPI owner. |
| `deferred` | Useful but not first-version work. | Full KG reasoning or live Vissim integration. |

## 4. Source-To-Goal Map

| Source | D5.1 content contribution | T5.2 development contribution | Mini-GA use |
|---|---|---|---|
| Grant Agreement | Scope, deliverable boundaries, risk-to-test matrix, component manifest, adapter obligations. | SUMO-first implementation, external adapter staging, RBAC/audit baseline. | Clarify D5.1 acceptance and D5.4/WP10 ownership. |
| D2.1 Event Mapping | Event taxonomy, scale/severity separation, event-source provenance, scenario trigger schema. | Event validation service, taxonomy registry, event-to-scenario builder. | Confirm event domain/scale/severity/AOI/validation owner per use case. |
| D2.2 Acceptability | Acceptability/equity/communication caveats, not a universal score. | Response-action warnings, explanation payloads, stakeholder evidence registry. | Confirm action acceptability, mitigation, vulnerable groups, communication channels. |
| D2.3 Equilibrium/AF | State vector, KPI vector, `Deff`, SRI, AF validation gates and maturity labels. | KPI normalizer, disturbance mapper, AF validation batch, feasible-set validator. | Confirm indicators, weights, baseline/recovery windows, constraints. |
| D2.4 Initial Analysis | Emergency priority, equity mitigation, explanation/reporting, trusted data sources. | Priority objective manager, equity burden checker, data-source profiler. | Confirm protected corridors, zones, mitigation rules, channels, report owners. |
| D2.5 Ontology | DTO-to-ontology mapping, IDs/IRIs, JSON-LD compatibility, named graphs, KPI observation split. | JSON-LD context, semantic mapping registry, optional semantic-lift jobs. | Confirm first class subset, IRI rules, mandatory properties, mapping owners. |
| D2.7 KPI Framework | KPI registry, selection, observation, thresholds, baselines, data roles. | KPI observation store, derived KPI calculator, threshold/baseline manager. | Confirm KPI subset, formulas, owners, data roles, thresholds, validation. |
| WP5/WP4 MoM/context | Adapter boundaries, CUSP readiness, control zones, retraining, scenario generation modes. | Adapter registry, control-zone model, role permissions, traceability ledger. | Force owner/fallback for WP4/CUSP/DMO/WP3 interfaces. |
| Mini-GA pack | Seed requirements, use cases, KPI register, pilot briefings, data inventory. | Convert only confirmed rows into fixtures/contracts. | Use cautiously; promote only rows with owner/status/due date. |

## 5. Day 1 Capture Priorities

Rhoe should listen for the items that Day 2 needs.

| Day 1 session | What Rhoe must capture |
|---|---|
| Use cases | use case ID, pilot, trigger, actor, decision question, expected SUMA value, retained/refined/deferred status |
| Requirements | essential vs deferred requirements, acceptance criterion, verification method, owner |
| Ontology/data | ontology class, object/property, data role, source, owner, mapping status, missing fields |
| KPIs | KPI ID, unit, layer, formula, baseline/threshold status, calculation mode, owner |
| Living lab/stakeholders | user role, vulnerable group, validation context, communication channel, feedback owner |

Do not let Day 1 produce a generic data wish list. A data item matters only if it supports a retained use case, KPI, response action, validation target, or interface.

## 5.1 Concise Agenda Backbone

| Time/session | Rhoe posture | Required output |
|---|---|---|
| Day 1, Session 1: use cases | Listen and clarify triggers, actors, assets, decision question. | Draft `UseCaseContract` rows. |
| Day 1, Session 2: requirements | Push for essential versus deferred, not all requirements. | Requirement shortlist and acceptance notes. |
| Day 1, Session 3: ontology/data | Ask for object, data role, owner, mapping status, missing input action. | Ontology/data map. |
| Day 1, Session 4: KPIs | Ask for unit, formula, baseline, threshold, owner, validation method. | `KpiEvidenceLedger` seed rows. |
| Day 2, Session 5: functions | Convert requirements into SUMA functions and API/spec rows. | Function/API map for D5.1. |
| Day 2, Session 6: architecture | Convert functions into common core, local adapter, external module, deferred item. | T5.2 architecture and adapter map. |
| Day 2, Session 7: triage/acceptability | Link actions to technical priority, acceptability, equity, mitigation, communication owner. | `ResponseAction` rows. |
| UI charrette | Convert roles into permissions, warnings, exports, guided/expert flows. | Role/API payload matrix. |
| Closeout | Confirm owner, status, due date, fallback. | Decision/risk log. |

## 6. Day 2 Session 5: SUMA Function Mapping

Purpose: translate Day 1 outputs into SUMA functions and API/service contracts.

Function families to use:

- ingest/validate,
- detect/classify,
- configure,
- simulate/model,
- assess,
- compare,
- recommend,
- explain/audit,
- notify/export,
- learn.

Required table:

| Pilot/use case | Translation layer | Requirement | SUMA function | Input/evidence | Output/decision | KPI | Endpoint/service if needed | Owner | Status/fallback |
|---|---|---|---|---|---|---|---|---|---|
| AHEPA flood access | `api_contract` | FR-01/FR-02/FR-03 | ingest event, define scenario, run simulation, expose KPI comparison | flood/access event fixture, corridor TBC | scenario KPI outputs | emergency access travel time | `/api/v1/disruption-events`, `/api/v1/scenarios`, `/api/v1/simulation-jobs` | AUTH/AHEPA and Rhoe | dependency: corridor/Living Lab permission |
| Larissa flood reroute | `evidence` + `data_contract` | FR-02/GR-03 | compare proxy scenario and show caveats | proxy flood boundary, traffic counts missing | degraded-mode output | travel time delta, equity burden | `/api/v1/scenarios`, `/api/v1/kpi-observations` | Larissa/Rhoe | proxy until counts/zones confirmed |

Closeout rule: every retained use case needs at least one row. Missing inputs must be labelled `acquire`, `proxy`, `defer`, or `drop`.

## 7. Day 2 Session 6: Architecture And Adapter Mapping

Purpose: define common SUMA core, pilot-local modules, external partner modules, and deferred placeholders.

Architecture contract table:

| Module | Type | Owner | Input schema | Output schema | Format/protocol | Runtime | KPI mapping | Maturity | Fallback |
|---|---|---|---|---|---|---|---|---|---|
| SUMO reference adapter | `common_core` | Rhoe | `Scenario`, `PilotConfig` | `SimulationRun`, `KpiObservation` | local file/process | local | core KPI subset | prototype | synthetic fixture |
| ETH control strategy | `external_module` | ETH | TBC | TBC | TBC | TBC | D2.7 mapping TBC | dependency | manual import/mock output |
| CUSP | `external_module` | CU/Optimize AI | TBC | TBC | NDA constrained | TBC | TBC | external dependency | placeholder adapter |

Module types:

- `common_core`
- `local_adapter`
- `external_module`

Implementation statuses:

- `implemented`
- `mock`
- `placeholder`
- `external_dependency`
- `deferred`

Critical asks:

- ETH/WP4: inputs, outputs, parameters, runtime, control zones, retraining assumptions, D2.7 KPI mapping.
- CU/Optimize AI/CUSP: NDA constraints, I/O package, auth/deployment model, latency, fallback.
- AUTH: Smart Mobility Living Lab feeds, SUMO/VISUM handoffs, hospital corridors, signal-priority capabilities, permissions.
- LISER/WP3: response-action/routing schema, acceptability constraints, vulnerable-group fields, mitigation labels.
- DMO: role-specific payloads, guided/expert flows, warnings, reports, exports.

## 8. Day 2 Session 7: Triage And Acceptability

Purpose: turn proposed interventions into structured response actions with technical and social status.

Response-action table:

| Action | Target disruption | Required inputs | Actor | Technical priority | Acceptability status | Equity risk | Mitigation | Communication owner | Deployability |
|---|---|---|---|---|---|---|---|---|---|
| emergency signal priority | AHEPA access disruption | corridor, signal control capability, emergency route | traffic operator/emergency services | high | not assessed | not assessed | review after pilot data | AUTH/AHEPA TBC | candidate only |
| car restriction/reroute | Bratislava public event or Larissa flood | affected zones, diversion routes, public channels | city/traffic operator | medium/high | contested or indicative only | mitigation required | public rationale and monitoring | city/DMO TBC | not deployable until mitigation |

Hard rule: a high-technical / low-acceptability action can remain a candidate, but it cannot be labelled deployment-ready unless mitigation, public rationale, communication owner, and residual-risk status are recorded.

## 9. UI Charrette

Rhoe should steer UI discussion into role-decision contracts.

| Role | Decision supported | Inputs viewed | Actions allowed | Warnings shown | Export/report needed |
|---|---|---|---|---|---|

Candidate roles:

- city planner,
- traffic operator,
- emergency responder,
- civil protection officer,
- hospital duty manager,
- public transport operator,
- policy decision-maker,
- analyst/researcher,
- admin/API operator.

Ask DMO to identify which `/api/v1` payloads each role needs and whether workflows are guided, expert, or both.

## 10. Partner Question Bank

| Partner | Questions |
|---|---|
| Rhoe | Is D5.1 `spec only`, `spec + stubs/tests`, or `prototype`? Which endpoint core, auth model, error model, schema versioning, and SUMO demo are in scope? |
| Cardiff/WP2 | Which requirements, ontology classes, KPI definitions, and `data_role` labels are authoritative after Mini-GA? Who owns Layer 3 KPI definitions and ARP target conflict? |
| CU/Optimize AI/CUSP | What I/O package can SUMA rely on? What is blocked by NDA? What auth/deployment model, data formats, latency, and fallback status apply? |
| ETH/WP4 | What are control-strategy inputs, outputs, parameters, runtime mode, control-zone boundaries, retraining assumptions, and D2.7 KPI mappings? |
| AUTH | What Living Lab feeds, SUMO/VISUM handoffs, signal-priority capabilities, hospital-access corridors, and permissions are available? |
| LISER | What routing/ABR action schema, acceptability constraints, vulnerable-group logic, mitigation labels, and communication channels should SUMA represent? |
| DMO | Which API payloads does the UI need? Which roles, warnings, exports, guided/expert flows, and explanation views are required? |
| WP3 leads | What response-action catalogue, FMEA risk IDs, technical-priority scale, expected AF gain, mitigation requirements, and D3.5 timing can WP5 depend on? |
| WP6/pilots | What validation targets cover usefulness, ease of use, emergency relevance, justice/equity, and stakeholder feedback? |
| Bratislava | Confirm first use case, VISUM/VISSIM handoff format, corridor/asset, PT/roadworks/event/flood data, live vs planning mode, validation target. |
| Larissa | Confirm Papanastasiou/flood boundaries, traffic-count timeline, parking/e-bike proxies, degraded-mode rules, recovery KPIs. |
| Odesa | Confirm shareable abstraction level, redaction/offline rules, power/infrastructure/emergency-lane use case, data owner, substitute acceptability evidence. |
| AHEPA | Confirm approach corridors, Living Lab permissions/frequencies, ambulance/staff routing fields, hospital role owners, signal-priority assumptions. |

## 11. Pilot Watchpoints

| Pilot | Main SUMA watchpoint | Required ask |
|---|---|---|
| Bratislava | Strong planning/model candidate; avoid assuming live operations. | Confirm use case, model handoff, PT/roadworks/event/flood data, validation target. |
| Larissa | Data scarcity and proxy/degraded mode. | Confirm Papanastasiou/flood boundaries, sensor availability, proxy rules, recovery KPIs. |
| Odesa | Fragmentary/security-sensitive data and missing D2.2 forum. | Confirm abstraction/redaction, offline mode, power/infrastructure use case, substitute evidence. |
| AHEPA | High-value critical-access candidate if permissions align. | Confirm corridors, Living Lab access, ambulance/staff routing fields, hospital owners. |

## 12. Required Close-Out Artefacts

| Table | Required columns |
|---|---|
| `UseCaseContract` | use case ID, pilot, trigger, actor, decision question, retained/refined/deferred, expected SUMA value, owner |
| `TraceabilityChain` | use case, requirement, ontology class, data object, `data_role`, KPI, function, endpoint/service, module, UI role, owner, fallback |
| `PilotConfig / ControlZone` | study area, network/OD ref, protected asset, corridor, control-zone refs, boundary stability, retraining flag, permissions, validation target, degraded mode |
| `DataReadiness` | variable, `input/computed_output/parameter`, priority, availability, owner, proxy, spatial/temporal scale, privacy, `acquire/proxy/defer/drop` |
| `KpiEvidenceLedger` | KPI ID, layer, unit, formula, baseline status, target status, dataset, confidence, verification method, owner |
| `AdapterContract` | module, owner, input schema, output schema, format/protocol, runtime, version, KPI mapping, maturity, fallback, licence/NDA |
| `ResponseAction` | action, target disruption, required inputs, actor, technical priority, acceptability status, equity risk, mitigation, communication owner, deployability |
| `Decision/Risk Log` | item, classification, owner, next action, due date, if unresolved then, D5.1/T5.2 implication |

## 13. D5.1/T5.2 Defaults To Defend Internally

| Decision | Recommended default |
|---|---|
| D5.1 | OpenAPI/JSON Schema, JSON-LD-compatible DTOs, examples, error model, auth assumptions, mock/reference adapters, initial tests. |
| T5.2 | SUMO-first end-to-end integration; external simulators as adapter contracts unless confirmed. |
| Ontology | JSON-LD-compatible metadata and class registry first; full KG/rule engine deferred. |
| AF | Calculation scaffolding and maturity labels first; no validated AF claim without baseline/post-event/equity/causal checks. |
| Acceptability | Evidence and gates first; no automated universal acceptability score. |
| CUSP | `external_dependency` until NDA, I/O, auth, deployment, and fallback are resolved. |
| KPI | No KPI claim without formula, unit, dataset, denominator, baseline/target, owner, and verification method. |

Partner-facing wording: describe these as "confirmation gates" rather than objections. For example, say "This can be included in the SUMA contract once the interface and validation owner are confirmed."

## 14. Technology Stack Position

Keep the current SUMA stack and formalize it rather than pivot.

| Layer | Recommendation |
|---|---|
| Backend/API | Python, FastAPI, Pydantic, Uvicorn, OpenAPI. |
| Simulation | SUMO/TraCI first; Vissim/Aimsun/VISUM as adapter contracts until confirmed. |
| Calculations | NumPy, Pandas, SciPy; NetworkX/GeoPandas where needed. |
| Frontend | React, TypeScript, Vite, Leaflet, Recharts. |
| Persistence | File/JSON/YAML plus SQLite metadata first; Postgres/PostGIS only if multi-pilot geospatial/concurrent use requires it. |
| Jobs | Existing local job manager in D5.1; Celery/RQ plus Redis only for durable long-running jobs in T5.2. |
| Ontology | JSON-LD context and mappings first; `rdflib`/`pySHACL` later; triple store only after CU confirms need. |
| Deployment | Docker Compose for RP1; avoid Kubernetes unless required by institutional deployment. |

## 15. Timeline To October 2026

This timeline separates D5.1 specification work from T5.2 implementation/integration work.

| Period | Main objective | Concrete work |
|---|---|---|
| 18-19 May Mini-GA | Convert consortium knowledge into owner-assigned rows. | Use case, requirement, ontology/data, KPI, function, architecture, triage, UI, decision/risk tables. |
| Late May 2026 | Consolidate Mini-GA outputs. | Produce confirmed `UseCaseContract`, `TraceabilityChain`, `KpiEvidenceLedger`, `AdapterContract`, and `PilotConfig` rows. |
| Early June 2026 | Start with ontology and identifiers. | Confirm D2.5 first class/property subset, define ID/IRI conventions, draft `/api/v1/ontology/context.jsonld`, add JSON/JSON-LD examples. |
| Mid June 2026 | Freeze D5.1 domain schemas. | Finalize `DisruptionEvent`, `Scenario`, `KpiDefinition`, `KpiObservation`, `PilotConfig`, `AdapterContract`, `DataInventoryItem`, and validation/error envelopes. |
| Late June 2026 | Build D5.1 examples and fixtures. | Add AHEPA emergency-access fixture, Larissa degraded/proxy flood fixture, Bratislava external-model adapter fixture, Odesa redacted/offline fixture. |
| July 2026 | Specify and stub `/api/v1`. | OpenAPI snapshot, schema examples, mock/stub endpoints, adapter-contract registry, KPI registry, documentation examples. |
| August 2026 | D5.1 freeze. | API specification, example payloads, known limitations, initial schema tests, risk-to-test matrix, D5.1 narrative. |
| September 2026 | T5.2 implementation sprint. | SUMO reference flow, local job facade, KPI observation export, file/SQLite metadata ledger, pilot configuration loading. |
| October 2026 | RP1 consolidation and T5.2 roadmap. | Validate what can be shown, document dependencies, update backlog for WP3/WP4/CUSP/DMO adapters, prepare RP1 evidence package. |

## 16. End-Of-Day Success Criteria

By 15:00 on 19 May 2026, Rhoe should have:

- retained/refined/deferred use cases per pilot,
- essential/deferred SUMA requirements per use case,
- first ontology/data/KPI/function/API/module/UI traceability rows,
- adapter contract rows for WP4, CUSP, AUTH, LISER, DMO, and simulator interfaces where relevant,
- pilot configuration asks and owners,
- Layer 3 KPI issues assigned,
- owner-assigned D5.1/T5.2 backlog,
- unresolved items classified as `assumption`, `dependency`, `risk`, `owner_missing`, or `deferred`.

Anything not owner-assigned is not a Rhoe commitment.
