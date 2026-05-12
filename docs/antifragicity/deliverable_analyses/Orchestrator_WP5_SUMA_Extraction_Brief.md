# Orchestrator WP5/SUMA Extraction Brief v0.3

Purpose: common extraction and revision rules for turning AntifragiCity sources into practical WP5/SUMA inputs for D5.1, T5.2, and Mini-GA preparation.

## 1. WP5/SUMA Lens

SUMA is the WP5 API-driven orchestration layer. It should connect WP2 event/ontology/KPI/requirements work, WP3 triage and response actions, WP4 control/routing/method outputs, WP6 validation, pilot configurations, and DMO/UI needs.

SUMA should not be treated as:

- the owner of final WP3 triage science,
- the owner of WP4 traffic-control algorithms,
- a full knowledge graph/rule-engine implementation in D5.1,
- a fully validated antifragility engine before pilot validation,
- a universal simulator-integration platform before access/licences are confirmed,
- the owner of city data acquisition or public acceptability certification.

## 2. Translation Layers

Every source finding must be classified before it is promoted into D5.1/T5.2.

| Layer | Meaning | Typical implementation |
|---|---|---|
| `methodology` | Principle or interpretation rule guiding SUMA use. | Documentation, UI warning, validation caveat. |
| `evidence` | Evidence, caveat, survey/forum result, pilot context, or validation note. | Provenance, confidence, evidence status, report text. |
| `data_contract` | Object or field that must be represented consistently. | Pydantic schema, JSON Schema, JSON-LD metadata. |
| `api_contract` | Callable or persisted software boundary. | Endpoint, adapter, job, request/response payload. |
| `governance` | Owner, permission, decision, due date, acceptance, or risk. | Decision log, component manifest, backlog. |
| `deferred` | Useful but not first-version work. | Roadmap/backlog with owner and trigger. |

Do not promote a methodology/evidence item to API unless a selected use case, partner module, UI/report, simulator adapter, validation workflow, or D5.1 acceptance criterion needs to call, persist, or exchange it.

## 3. Standard Analysis Structure

Each deliverable/context analysis should include:

- source, status, and primary WP5 role,
- what the source is for and what it is not for,
- evidence table with page/section/date references,
- direct inputs to D5.1,
- T5.2 development implications,
- concrete objects/fields/equations/modules only where justified,
- Mini-GA questions and expected table outputs,
- technology stack note if the source has implementation implications,
- explicit overclaim/deferred warnings.

## 4. SUMA Translation Matrix

Use this matrix when reviewing or extending any analysis:

| Source finding | WP5 meaning | Layer | D5.1 use | T5.2 use | Mini-GA ask | Object/API decision | Status |
|---|---|---|---|---|---|---|---|
| event taxonomy | common scenario trigger language | data_contract | `DisruptionEvent` schema | event validation service | confirm taxonomy version | specify core contract | dependency on WP2 version |
| citizen acceptability | social caveat/gate | methodology/evidence | warnings and evidence fields | explanation/equity module | confirm action status | optional/stub | pilot-specific |
| D2.3 AF formula | calculation contract | methodology/api_contract | maturity-labelled schema | calculation module | confirm weights/baselines | staged | not validated |
| KPI catalogue | registry and observation model | data_contract | KPI schemas/endpoints | KPI store/calculator | confirm selected subset | specify core contract | subset owner needed |

Example complete extraction chain:

| Source | D5.1 contract output | T5.2 implementation output | Mini-GA confirmation |
|---|---|---|---|
| D2.1 flood event plus D2.5 ontology plus D2.7 access KPI | `DisruptionEvent`, `Scenario`, `KpiDefinition`, JSON-LD context example, validation errors. | SUMO scenario fixture, KPI observation export, scenario comparison output. | Pilot confirms study area, flood trigger, route/corridor, KPI unit, baseline/proxy, and owner. |
| D2.2 acceptability plus WP3 triage placeholder | `ResponseAction`, `AcceptabilityAssessment`, equity/mitigation fields. | Response-action catalogue and warning/explanation output. | LISER/WP3 confirms response categories, social variables, and what is evidence versus assumption. |

## 5. D5.1/T5.2 Default Scope

| Item | Default |
|---|---|
| D5.1 | OpenAPI/JSON Schema, JSON-LD-compatible DTOs, examples, error model, auth assumptions, mock/reference adapters, initial tests. |
| T5.2 | SUMO-first integration, adapter registry, scenario/job/run lifecycle, KPI observations, pilot configuration, external module contracts. |
| External simulators | Adapter contracts unless access, licence, support, and handoff format are confirmed. |
| Ontology | JSON-LD-compatible metadata first; full KG/rule execution deferred. |
| Antifragility | Calculation scaffolding and maturity labels first; validation requires baseline/post-event/equity/causal checks. |
| Acceptability | Evidence and gates first; no automated universal acceptability score. |
| Storage | File/JSON/YAML plus SQLite metadata first; PostGIS/Redis/Celery only when needed in T5.2. |

## 6. Hard Questions For Every Source

- What exact object does SUMA need to ingest, store, compute, expose, validate, or defer?
- Is this common core, local pilot configuration, external module, placeholder, or deferred?
- Is the item raw input, computed output, or parameter?
- What provenance, confidence, data quality, validation status, and owner are required?
- Which partner owns the method, data, validation, interface, permission, or explanation?
- What happens if the input is unavailable?
- Can it be represented in JSON/JSON-LD and linked to D2.5 ontology classes?
- Which KPI proves value, and is the KPI computable now?
- Is the recommendation socially acceptable, not assessed, indicative only, contested, or requiring mitigation?

## 7. Technology Stack Baseline

Use the current SUMA stack as the baseline:

- Backend: Python, FastAPI, Pydantic, Uvicorn.
- Simulation: SUMO/TraCI first; Vissim/Aimsun/VISUM as adapter contracts until confirmed.
- Calculation: NumPy, Pandas, SciPy; NetworkX/GeoPandas/PostGIS only where needed.
- Frontend: React, TypeScript, Vite, Leaflet, Recharts.
- Persistence: file-first plus SQLite metadata in D5.1/T5.2 early phase; Postgres/PostGIS only for multi-pilot spatial/concurrent deployment needs.
- Jobs: current local job manager for D5.1; Celery/RQ plus Redis only for durable long-running workloads.
- Ontology: JSON-LD context and mappings first; `rdflib`/`pySHACL` later; triple store only after CU confirms query/use cases.
- Deployment: Docker Compose; avoid Kubernetes for RP1 unless institutional deployment requires it.
