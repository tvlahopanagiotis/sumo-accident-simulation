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

Closeout rule: every retained use case needs at least one row. Missing inputs must be labelled `acquire`, `proxy`, `defer`, or `drop`.

## 7. Day 2 Session 6: Architecture And Adapter Mapping

Purpose: define common SUMA core, pilot-local modules, external partner modules, and deferred placeholders.

Architecture contract table:

| Module | Type | Owner | Input schema | Output schema | Format/protocol | Runtime | KPI mapping | Maturity | Fallback |
|---|---|---|---|---|---|---|---|---|---|

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

## 13. D5.1/T5.2 Defaults To Defend

| Decision | Recommended default |
|---|---|
| D5.1 | OpenAPI/JSON Schema, JSON-LD-compatible DTOs, examples, error model, auth assumptions, mock/reference adapters, initial tests. |
| T5.2 | SUMO-first end-to-end integration; external simulators as adapter contracts unless confirmed. |
| Ontology | JSON-LD-compatible metadata and class registry first; full KG/rule engine deferred. |
| AF | Calculation scaffolding and maturity labels first; no validated AF claim without baseline/post-event/equity/causal checks. |
| Acceptability | Evidence and gates first; no automated universal acceptability score. |
| CUSP | `external_dependency` until NDA, I/O, auth, deployment, and fallback are resolved. |
| KPI | No KPI claim without formula, unit, dataset, denominator, baseline/target, owner, and verification method. |

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

## 15. End-Of-Day Success Criteria

By 15:00 on 19 May 2026, Rhoe should have:

- retained/refined/deferred use cases per pilot,
- essential/deferred SUMA requirements per use case,
- first ontology/data/KPI/function/API/module/UI traceability rows,
- adapter contract rows for WP4, CUSP, AUTH, LISER, DMO, and simulator interfaces where relevant,
- pilot configuration asks and owners,
- Layer 3 KPI issues assigned,
- owner-assigned D5.1/D5.2 backlog,
- unresolved items classified as `assumption`, `dependency`, `risk`, `owner_missing`, or `deferred`.

Anything not owner-assigned is not a Rhoe commitment.

