# Rhoe Mini-GA Thessaloniki Working Guide

Status: internal Rhoe working guide.

Audience: Rhoe members supporting WP5/SUMA during the Thessaloniki Mini-GA.

Date context: prepared on 11 May 2026 for the Mini-GA on 18-19 May 2026.

Purpose: provide a practical guide for Rhoe to use during the charrette-style Mini-GA. This document is not a public deliverable. It translates the project evidence, updated agenda, and CU Mini-GA reference pack into facilitation guidance, decision rules, session canvases, and talking points for WP5/SUMA.

## 1. Rhoe Objective

Rhoe's objective is to help the consortium leave the Mini-GA with a defensible first SUMA prototype definition and D5.1 roadmap. The goal is not to solve every scientific or technical dependency in two days. The goal is to convert partner inputs into a traceable and buildable structure:

`use case -> requirement -> ontology/data object -> KPI -> SUMA function -> API/service -> architecture module -> UI role -> backlog owner`

By 15:00 on Tuesday 19 May 2026, Rhoe should have either a confirmed value or an explicit `assumption`, `dependency`, `risk`, or `deferred` label for:

- retained/refined use cases per pilot,
- requirement subsets per pilot,
- ontology/data structures tied to those requirements,
- KPI subsets and availability status,
- SUMA function map,
- first architecture split between common core and local adaptation,
- triage/acceptability constraints,
- initial UI roles and workflows,
- open dependencies with named owners.

## 2. Rhoe Position

SUMA should be positioned as the WP5 integration and orchestration environment for AntifragiCity. It is not only a simulator and not only a UI. It should provide API-driven scenario configuration, simulation/method execution, KPI and antifragility assessment, explainability, export, and feedback/learning loops.

The current SUMA implementation already provides a SUMO-based execution scaffold. The Mini-GA should help determine how that scaffold becomes project-aligned: ontology-aligned objects, D2.6-derived requirements, D2.7 KPI subsets, D2.2/D2.4 acceptability constraints, WP3 triage placeholders, WP4 adapter boundaries, and WP6 / pilot demonstration inputs.

Suggested diplomatic line:

> Rhoe's role on Day 2 is to help translate the Day 1 evidence into implementable SUMA functions, API contracts, architecture modules, and backlog items.

Rhoe should treat Sessions 5-6 as the conversion-owner sessions: timebox reopened debates, record unresolved items, protect WP5 scope boundaries, and make sure abstract partner modules become input/output contracts or explicit dependencies.

## 3. Non-Negotiables

- Do not let SUMA become responsible for inventing final WP3 triage science or final WP4 control algorithms.
- Do not present prototype, proxy, theoretical, or indicative results as validated.
- Do not treat social acceptability as a UI afterthought; it must constrain recommendations and triage outputs.
- Do not ask pilot partners to provide computed outputs or calibration parameters as if they were raw input data.
- Do not make data-rich assumptions from Thessaloniki/AHEPA mandatory for Larissa or Odesa.
- Do not leave Day 2 with abstract module names but no input/output contracts.
- Do not allow missing partner inputs to become implicit Rhoe commitments.

## 4. Day 2 Operating Rule

Use a strict session rule:

`retain / refine / defer`, then `decide / assume / block`

If a use case, requirement, or KPI was not selected on Day 1, it should go to backlog unless it blocks the minimum viable SUMA prototype.

For every proposed SUMA function, ask:

- Who is the actor?
- What is the input?
- What is the output?
- Which KPI or decision does it support?
- Is it common core or pilot-local?
- Which partner owns the dependency?
- Is it implemented, placeholder, mock, external, or deferred?

If no decision is possible, classify the item immediately as `assumption`, `dependency`, `risk`, `deferred`, or `owner_missing`.

## 5. Updated Agenda Backbone

### Day 1, Monday 18 May 2026

Rhoe's Day 1 role is mostly listening, mapping, and capturing implementation implications.

Key sessions:

- 09:45-10:45: Pilot demonstrator briefings.
- 11:20-12:45: Use-case co-creation.
- 13:45-14:45: Requirements selection from D2.6.
- 14:45-15:45: Mapping requirements to D2.5 ontology/data structures.
- 16:05-17:05: Session 4A, KPI selection from D2.7.
- 16:05-17:05: Session 4B, Living Lab framing and stakeholder ecosystem mapping led by LISER.
- 17:05-17:30: Day 1 synthesis.

Rhoe should capture:

- use-case identifiers,
- retained/refined/deferred status,
- requirement codes,
- data objects,
- KPI candidates,
- missing data,
- pilot-specific corridors/assets,
- partner commitments,
- ambiguities for Day 2.

### Day 2, Tuesday 19 May 2026

Rhoe's Day 2 role is active facilitation and consolidation.

Key sessions:

- 09:15-10:15: Session 5, translating requirements into SUMA functionality.
- 10:15-11:15: Session 6, designing SUMA DSS architecture.
- 11:35-12:15: Session 7, triage measures and acceptability milestone.
- 12:15-12:45: Plenary build session, framing SUMA DSS.
- 13:45-14:30: UI charrette.
- 14:30-14:55: Prototype walkthrough and implementation backlog.
- 14:55-15:00: close and owner assignment.

Rhoe outputs by session:

| Session | Rhoe output |
|---|---|
| Session 5 | function table |
| Session 6 | module/interface map |
| Session 7 | triage/acceptability matrix |
| UI charrette | role-flow table |
| Close-out | owner-assigned backlog |

## 6. Session 5 Playbook: Requirements To SUMA Functionality

Purpose: convert retained Day 1 requirements into concrete SUMA functions.

Function families:

- ingest/validate,
- detect/classify,
- model/simulate,
- assess,
- recommend/compare,
- explain/audit,
- notify/export,
- learn.

Session 5 table:

| Use case | Requirement | Decision question | SUMA function | Actor | Input | Output | KPI | Core/local | Data owner | Missing-input behaviour | Calculation mode | Owner | Deadline | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pending | FR-01 | Can this event be accepted by SUMA? | validate event | traffic operator | DisruptionEvent | validation result | n/a | common core | pilot/CU | validation error | n/a | Rhoe/CU | pending | placeholder |

Status values:

- `implemented`
- `placeholder`
- `mock`
- `external_dependency`
- `deferred`

Rhoe should push for at least one concrete row per retained use case.

## 7. Session 6 Playbook: Architecture And Module Boundaries

Purpose: translate function rows into architecture modules, data flows, and common/local split.

Architecture layers:

- semantic input layer,
- pilot data/configuration layer,
- simulation and method adapter layer,
- KPI/antifragility assessment layer,
- API/UI/export layer.

Named modules from the agenda:

| Module | Partner | Rhoe question |
|---|---|---|
| CUSP backbone | CU | What API/data interface can SUMA rely on, and what is blocked by NDA? |
| Short-term DSS / ABM | AUTH | What outputs can become SUMA response actions or recommendations? |
| Traffic control module | ETH | What inputs, outputs, parameters, and retraining assumptions are required? |
| ABR routing | LISER | What route/action format and acceptability constraints are expected? |
| RE Suite / UI layer | DMO | What dashboard and UI workflows consume SUMA outputs? |
| SUMA API | Rhoe | What core services and schemas should D5.1 define first? |

Architecture contract table:

| Module | Owner | Input | Output | Format | Update frequency | Maturity | Fallback |
|---|---|---|---|---|---|---|---|
| pending | pending | pending | pending | pending | pending | pending | pending |

Module classification uses two separate fields.

Module type:

- `common_core`
- `local_adapter`
- `external_module`

Implementation status:

- `implemented`
- `placeholder`
- `mock`
- `external_dependency`
- `deferred`

## 8. Session 7 Playbook: Triage And Acceptability

Purpose: screen recommended measures for technical priority, acceptability evidence, mitigation needs, and evidence gaps.

Rhoe should insist on two independent dimensions:

- technical priority,
- social acceptability.

Do not merge them into a single score too early.

Acceptability checks:

- emergency access,
- public transport continuity,
- vulnerable-user impact,
- neighbourhood burden distribution,
- essential-service accessibility,
- digital exclusion,
- privacy/surveillance sensitivity,
- communication and explanation readiness,
- reversibility and proportionality,
- mitigation needs.

Evidence and validation status:

- `not_assessed`
- `indicative_only`
- `assessed`
- `pilot_validated`
- `conflicting_evidence`

Mitigation status:

- `none_required`
- `mitigation_required`
- `mitigation_defined`

Suggested categories:

| Category | Examples | SUMA representation |
|---|---|---|
| Critical services | ambulance, fire, police, crisis response | priority class / emergency corridor |
| Essential mobility | public transport users, caregivers, vulnerable groups, essential workers | protected access / service continuity |
| Economic continuity | freight, deliveries, travel-dependent jobs | exemption/support logic |
| General mobility | routine commuting, non-essential private vehicle use | lower priority under constraint |

Suggested rule:

- High technical priority and high acceptability: retain.
- High technical priority and low acceptability: retain only if mitigation, communication owner, and residual-risk label are recorded.
- Low technical priority and high acceptability: retain only if it supports trust or enabling conditions.
- Low technical priority and low acceptability: defer or discard.

## 9. UI Charrette Playbook

Purpose: translate architecture and functions into role-based workflows.

Rhoe should ask DMO and pilot partners to define:

- user roles,
- permissions,
- scenario setup flow,
- validation/readiness warnings,
- KPI interpretation view,
- comparison view,
- export/report needs,
- public-facing explanation needs.

UI role-to-decision table:

| Role | Decision supported | Inputs viewed | Actions allowed | Warnings shown | Export/report needed |
|---|---|---|---|---|---|
| pending | pending | pending | pending | pending | pending |

Candidate roles:

- city planner,
- traffic operator,
- emergency responder,
- hospital duty manager,
- policy decision-maker,
- technical analyst,
- administrator,
- community/stakeholder reviewer.

Important Rhoe caution:

The UI should not hide uncertainty. It should display assumptions, data quality, missing inputs, calculation mode, and validation status.

## 10. Pilot Watchpoints

### Bratislava

Current framing:

- daily and mid-scale stressors,
- roadworks and corridor planning,
- public-transport disruption,
- public events/protests,
- Danube flooding and underpasses.

Data posture:

- VISUM/VISSIM maturity,
- partial key-intersection traffic counts.

Rhoe watchpoint:

- likely strong candidate for planning/scenario and external-model integration discussions.

Required ask:

- confirm priority corridor/asset, data owner, validation target, protected users/assets, and first acceptable degraded-mode assumption.

### Larissa

Current framing:

- flood resilience,
- Papanastasiou priority corridor,
- limited traffic-count data,
- parking/e-bike data as possible proxies.

Rhoe watchpoint:

- use Larissa to test degraded/proxy mode and missing-input reporting.

Required ask:

- confirm Papanastasiou protection logic, available proxy data owner, flood validation target, protected users/assets, and acceptable degraded-mode assumption.

### Odesa

Current framing:

- power outages,
- conflict and infrastructure damage,
- fragmentary data,
- shared bus/emergency lanes under design,
- no D2.2 citizens' forum due to security situation.

Rhoe watchpoint:

- require sensitive-data handling, offline/degraded assumptions, and tailored acceptability baseline.

Required ask:

- confirm critical connection/asset, safe data owner, validation target, protected users/assets, and acceptable offline/degraded-mode assumption.

### AHEPA / Thessaloniki

Current framing:

- hospital-access pilot,
- floods, congestion, earthquakes, regional disruption,
- rich Smart Mobility Living Lab data,
- priority asset is AHEPA Hospital.

Rhoe watchpoint:

- likely best high-fidelity prototype candidate, especially for emergency access and smart-signal/data integration.

Required ask:

- confirm AHEPA approach corridors, data/permission owner, emergency-access validation target, protected users/assets, and first signal/data integration assumption.

## 11. Data Readiness Rule

Use the updated data inventory logic:

- ask partners for `data_role=input`,
- do not ask them to collect computed outputs,
- do not ask them to confirm calibration parameters before calibration runs.

Action labels for data gaps:

- `acquire`: partner can provide or collect,
- `proxy`: indirect input can support prototype,
- `defer`: mark out of first prototype,
- `drop`: not needed for selected use case.

Important current fact:

- the real high-priority external acquisition target is 23 input variables, not 85 mixed variables.

## 12. D5.1 Translation Priorities

After the Mini-GA, D5.1 should be able to specify:

- resource model,
- endpoint families,
- request/response schemas,
- async job model,
- validation and error behaviour,
- data quality and provenance fields,
- pilot configuration model,
- simulation adapter contract,
- WP3/WP4 placeholder contracts,
- KPI availability and calculation modes,
- acceptability/equity status,
- roadmap and deferred items.

Minimal protected endpoint core:

- `/api/v1/disruption-events/validate`
- `/api/v1/scenarios`
- `/api/v1/simulation-jobs`
- `/api/v1/simulation-jobs/{job_id}`
- `/api/v1/simulations/{run_id}/status`
- `/api/v1/simulations/{run_id}/results`
- `/api/v1/simulations/{run_id}/kpis`
- `/api/v1/kpis/availability`
- `/api/v1/kpis/calculate`
- `/api/v1/pilots/{pilot_id}/data-readiness`
- `/api/v1/pilots/{pilot_id}/data-inventory`
- `/api/v1/triage-recommendations`
- `/api/v1/acceptability-assessments`
- `/api/v1/equity-impacts`
- `/api/v1/ontology/export`
- `/api/v1/scenarios/{scenario_id}/audit-records`
- `/api/v1/stakeholder-inputs`

## 13. Decision Log

Use this table during the meeting.

| Session | Pilot/use case | Item | Type | Decision / issue | Owner | Next action | Due date | If unresolved then |
|---|---|---|---|---|---|---|---|---|
| pending | pending | pending | decision | pending | pending | pending | pending | pending |

Types:

- `decision`
- `assumption`
- `dependency`
- `risk`
- `deferred`

## 14. Questions Rhoe Should Keep Asking

- What is the minimum viable version of this function for D5.1/D5.2?
- Is this common across all pilots or local to one pilot?
- What data object does this require?
- Which ontology class or relationship represents it?
- Which KPI proves it worked?
- What happens if the data is missing?
- Is the output technically valid, socially acceptable, both, or neither?
- Who owns the module, data, validation, or decision?
- What is the fallback if the partner output is not ready?

## 15. Close-Out Checklist

By the final close, Rhoe must classify:

- each pilot has retained/refined use cases,
- each retained use case has requirements,
- each requirement maps to data structures,
- KPI subsets are not excessive,
- missing data is labelled acquire/proxy/defer/drop,
- SUMA functions have inputs and outputs,
- architecture modules have owners,
- WP4 and WP3 are represented through contracts/placeholders if not final,
- social acceptability and equity constraints are not left vague,
- UI roles are linked to actual functions,
- D5.1 backlog is owner-assigned.
