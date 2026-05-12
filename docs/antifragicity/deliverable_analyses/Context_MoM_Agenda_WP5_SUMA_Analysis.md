# Context, Meetings, And Agenda: WP5/SUMA Input Brief v0.3

Sources: `docs/antifragicity/context/`  
Relevant files: WP5 MoM 1 Apr 2026, WP5 MoM 22 Apr 2026, Rhoe internal understanding, WP4/WP5 meeting 29 Apr 2026, Mini-GA Thessaloniki Agenda LV1.  
Primary WP5 role: live project intent, open dependencies, partner boundaries, Mini-GA facilitation structure, and owner-assignment discipline.

## 1. What This Source Is For

The context and meeting files clarify how Rhoe should steer WP5: SUMA operationalises partner outputs through APIs, adapters, schemas, jobs, provenance, and role-aware outputs. It does not absorb undefined WP3/WP4 science or city data acquisition into hidden Rhoe assumptions.

The Mini-GA agenda is the key conversion point. Day 1 creates use cases, requirements, ontology/data structures, and KPIs. Day 2 must convert those into functions, architecture modules, adapter contracts, response actions, UI roles, and backlog owners.

## 2. Evidence To Preserve In D5.1 And Mini-GA Preparation

| Evidence | Context reference | WP5/SUMA interpretation |
|---|---|---|
| WP5 integrates/adapts/operationalises WP4 outputs, not create traffic algorithms. | WP5 MoM 1 Apr and 22 Apr 2026 | WP4 enters through adapter contracts. |
| CUSP NDA and technical access remain unclear. | WP5 MoM 22 Apr 2026 | CUSP stays `external_dependency` until I/O, auth, and deployment constraints are known. |
| Protected/congestion zones and retraining implications were discussed. | WP5 MoM 22 Apr 2026 | Add `ControlZone` with boundary stability and retraining fields. |
| Bottom-up versus top-down scenario generation is open. | WP5 MoM 22 Apr; Rhoe internal understanding | Add `Scenario.generation_mode` and evidence status. |
| ETH/WP4 indicators and D2.7 replacement/mapping need resolution. | WP4/WP5 meeting 29 Apr 2026 | Add `ControlKpiMapping`; do not hard-code WP4 indicators as final KPIs. |
| Mini-GA LV1 agenda frames Day 2 as functionality/architecture/UI/backlog conversion. | Mini-GA LV1 agenda, 18-19 May 2026 | Rhoe must close sessions with tables, not open-ended notes. |

## 3. Direct Inputs To D5.1/T5.2

| Topic | Specific content to add |
|---|---|
| WP4 adapter boundary | D5.1 must define adapter fields for input, output, parameters, runtime, retraining, KPI mapping, and fallback. |
| CUSP readiness | Add `nda_status`, `access_constraints`, `adapter_pending`, and `fallback`. |
| Scenario generation | Add `generation_mode`: observed, historical replay, synthetic stress test, forecast, manual. |
| Control zones | Add geometry, road refs, protected corridor status, validity horizon, owner, confidence, retraining flag. |
| Role permissions | Add role-to-parameter permissions and guided/expert workflow split. |
| Decision log | Every unresolved item must have classification, owner, next action, due date, and fallback. |

## 4. Required Mini-GA Table Outputs

These are close-out artefacts, not optional notes.

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

## 5. Partner Questions To Carry Into Mini-GA

| Partner | Questions |
|---|---|
| Rhoe | Is D5.1 `spec only`, `spec + stubs/tests`, or `prototype`? Which endpoint core, auth model, error model, schema versioning, and SUMO demo are in scope? |
| Cardiff/WP2 | Which requirements, ontology classes, KPI definitions, and `data_role` labels are authoritative after Mini-GA? Who owns Layer 3 KPI definitions and the ARP target conflict? |
| CU/Optimize AI/CUSP | What I/O package can SUMA rely on? What is blocked by NDA? What auth/deployment model, data formats, latency, and fallback status apply? |
| ETH/WP4 | What are control-strategy inputs, outputs, parameters, runtime mode, control-zone boundaries, retraining assumptions, and D2.7 KPI mappings? |
| AUTH | What Smart Mobility Living Lab feeds, SUMO/VISUM handoffs, signal-priority capabilities, hospital-access corridors, and permissions are available? |
| LISER | What routing/ABR action schema, acceptability constraints, vulnerable-group logic, mitigation labels, and communication channels should SUMA represent? |
| DMO | Which API payloads does the UI need? Which roles, warnings, exports, guided/expert flows, and explanation views are required? |
| WP3 leads | What response-action catalogue, FMEA risk IDs, technical-priority scale, expected AF gain, mitigation requirements, and D3.5 timing can WP5 depend on? |
| WP6/pilots | What validation targets cover usefulness, ease of use, emergency relevance, justice/equity, and stakeholder feedback? |
| Bratislava | Confirm first use case, VISUM/VISSIM handoff format, corridor/asset, PT/roadworks/event/flood data, live vs planning mode, validation target. |
| Larissa | Confirm Papanastasiou/flood boundaries, traffic-count timeline, parking/e-bike proxies, degraded-mode rules, recovery KPIs. |
| Odesa | Confirm shareable abstraction level, redaction/offline rules, power/infrastructure/emergency-lane use case, data owner, substitute acceptability evidence. |
| AHEPA | Confirm approach corridors, Living Lab permissions/frequencies, ambulance/staff routing fields, hospital role owners, signal-priority assumptions. |

## 6. T5.2 Development Implications

| Module | Context-driven requirement |
|---|---|
| Adapter contract registry | Needed for WP4, CUSP, AUTH, LISER, DMO, and external simulators. |
| Control-zone model | Needed for WP4 control outputs and pilot-specific protected/congestion zones. |
| Scenario classification | Needed to separate bottom-up observed, historical replay, synthetic stress-test, forecast, and manual scenarios. |
| Role/permission model | Needed for DMO UI, pilot roles, exports, and access controls. |
| Traceability ledger | Needed to avoid implicit Rhoe ownership of unresolved dependencies. |

## 7. Technology Stack Note

Use Markdown/CSV tables during Mini-GA capture, then convert confirmed rows into JSON fixtures or SQLite records. Do not build elaborate tooling before the meeting. After Mini-GA, import confirmed `UseCaseContract`, `AdapterContract`, `KpiEvidenceLedger`, and `PilotConfig` rows into the FastAPI/Pydantic contract layer.

