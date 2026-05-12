# Mini-GA Working Pack: WP5/SUMA Input Brief v0.3

Sources: `docs/antifragicity/miniGA/`  
Status: coordinator-preparation material, not a contractual commitment source.  
Primary WP5 role: practical reference pack for requirements, use cases, KPI register, ontology reference, triage measures, data inventory, and pilot briefings.

## 1. Critical Caveat

The Mini-GA working pack is useful because it is concise and operational, but it overlaps with deliverables and contains coordinator-prepared syntheses. Treat it as preparation material. Only Mini-GA rows confirmed with `owner`, `status/fallback`, and `due_date` should become D5.1/T5.2 scope.

## 2. What This Source Is For

Use the pack to pre-fill tables, accelerate discussion, and avoid getting lost in full deliverables during the charrette. Do not use it to override primary deliverables unless partners explicitly confirm the Mini-GA output.

## 3. Evidence To Preserve

| Evidence | Mini-GA pack reference | WP5/SUMA interpretation |
|---|---|---|
| Requirements reference defines 20 SUMA requirements. | `miniga_requirements.txt`, FR-01 to GR-04 | Convert requirements into acceptance criteria and endpoint/module checks. |
| Use-case shortlist gives four-pilot candidates. | `miniga_usecase_shortlist.txt` | Use cases anchor traceability chains. |
| Ontology reference repeats 12 classes/relations. | `miniga_ontology.txt` | Use as working semantic subset, confirm with CU. |
| KPI register separates Layer 1/2/3 and exposes Layer 3 issues. | `miniga_kpi_register.txt` | Add `kpi_layer`, definition status, owner, baseline/target status. |
| Data inventory report corrects acquisition target to high-priority input variables. | `miniga_data_inventory_report.txt` and CSV | Enforce `data_role` and avoid asking for outputs/parameters as raw city data. |
| Triage reference links response actions, technical priority, and acceptability. | `miniga_triage.txt` | Structure response actions, do not leave triage as free text. |
| D2.2/D2.4 input file translates acceptability into constraints. | `miniga_input_suma_d22_d24.txt` | Add fairness, burden, communication, vulnerable-group fields where relevant. |

## 4. Direct Inputs To D5.1/T5.2

| Mini-GA pack item | D5.1/T5.2 use | Promotion rule |
|---|---|---|
| FR-01 to GR-04 requirements | D5.1 acceptance and test matrix | Promote only after Mini-GA essential/deferred decision. |
| Use-case shortlist | `UseCaseContract` seed rows | Promote retained/refined use cases only. |
| Ontology reference | DTO-to-class mapping seed | Confirm mandatory class/property subset with CU. |
| KPI register | KPI registry and evidence ledger seed | Require owner, formula, unit, baseline, target, verification. |
| Data inventory | `DataInventoryItem` and readiness table | Ask only for `input` data; mark outputs/parameters correctly. |
| Triage measures | `ResponseAction` seed catalogue | Require technical priority, acceptability, equity, owner. |
| Pilot briefings | `PilotConfig` seed rows | Confirm study areas, data, validation target, permissions. |

## 5. Pilot-Specific Use-Case Watchpoints

| Pilot | Candidate SUMA value | Main Mini-GA confirmation |
|---|---|---|
| Bratislava | Planning/model path for roadworks, PT disruptions, public events, flood/underpass response. | Confirm VISUM/VISSIM/SUMO handoff, live vs planning mode, corridor/assets, validation target. |
| Larissa | Flood rerouting, Papanastasiou protection, storm forecast, recovery monitoring. | Confirm boundaries, traffic counts, proxy/degraded mode, recovery KPIs. |
| Odesa | Power outage, infrastructure damage, simultaneous disruptions, restoration planning, PT optimisation. | Confirm abstraction/redaction level, offline rules, shareable data, substitute acceptability evidence. |
| AHEPA | Hospital access during flood/congestion/earthquake/regional disruption. | Confirm Living Lab permissions, hospital approach corridors, ambulance/staff routing, signal priority assumptions. |

## 6. Requirement-To-Prototype Mapping

| Cluster | Requirements | First-prototype implication |
|---|---|---|
| Event/scenario | FR-01, FR-02 | Ontology-aligned event ingestion and scenario lifecycle. |
| Assessment | FR-03, DR-02, GR-03 | KPI/AF outputs with provenance, quality, uncertainty, assumptions. |
| Intervention | FR-04, GR-02 | Response-action catalogue, comparison, audit. |
| API/integration | FR-05, IR-01, IR-02 | Documented `/api/v1`, SUMO reference adapter, external adapter contracts. |
| Real-time/data | FR-06, DR-01 | Source-specific modes; offline/degraded/proxy labels if no SLO. |
| Learning | FR-07, NFR-03 | Lightweight learning artefacts and feedback records; automation later. |
| UI | UR-01, UR-02 | Role-based workflows, scenario comparison, warnings, reports, exports. |
| Governance | GR-01 to GR-04 | Privacy, auditability, assumptions, stakeholder input governance. |

## 7. Data And KPI Rules

- The high-priority acquisition target should be input variables, not mixed outputs and parameters.
- Every data row needs `data_role`: `input`, `computed_output`, or `parameter`.
- Layer 3 SUMA KPIs need definitions, baselines, targets, formulas, owners, datasets, and verification methods before becoming acceptance criteria.
- ARP target conflicts must be resolved before D5.1 uses them as claims.
- OD readiness and link-flow outputs are major dependencies for network diversity, redundancy, and resilience metrics.

## 8. Mini-GA Questions

Ask all table leads:

- Which Mini-GA pack rows are confirmed versus preparation-only?
- For each retained use case, which requirements are essential and which are deferred?
- Which data inventory rows are true inputs, computed outputs, or parameters?
- Which KPI rows have formula, unit, baseline, target, owner, and verification method?
- Which response actions have technical priority, acceptability status, equity risk, mitigation, and deployability?
- Which pilot constraints prevent D5.1/T5.2 implementation?

Expected Mini-GA output: confirmed tables with owner, status/fallback, due date, and explicit D5.1/T5.2 implication.

## 9. Technology Stack Note

Use the Mini-GA pack as seed data for Markdown/CSV capture templates. After the meeting, convert confirmed rows into JSON fixtures for `/api/v1` examples and SQLite metadata for requirements, use cases, KPI evidence, adapter contracts, and pilot configuration. Do not import the whole pack as authoritative machine data without partner confirmation.

