# Grant Agreement: WP5/SUMA Input Brief v0.3

Source: `docs/antifragicity/deliverables/AntifragiCity - Grant Agreement - GAP-101203052.pdf`  
Status: contractual project baseline.  
Primary WP5 role: scope boundary, deliverable obligations, risk framing, KER/component integration, WP5/WP10/WP11 dependencies, and acceptance criteria staging.

## 1. What This Source Is For

The Grant Agreement defines SUMA/KER5 as a WP5 integration and orchestration environment. WP5 must assemble WP2 ontology/KPI/requirements, WP3 triage outputs, WP4 control/routing/method outputs, WP6 validation needs, and pilot-demonstrator inputs into an API-driven and user-facing DSS environment.

The Grant Agreement should not be mined mechanically into API objects. It should be used to set scope, delivery stages, acceptance criteria, risks, governance records, and component/interface obligations.

## 2. Evidence To Preserve In D5.1

| Evidence | Reference | D5.1 implication |
|---|---|---|
| WP5 assembles WP2, WP3, and WP4 outputs into SUMA. | GA pp. 75-76 | SUMA core should be thin: API, schema, orchestration, adapters, provenance, audit. |
| D5.1 is Core SUMA API specification due M16. | GA p. 100 | D5.1 should be OpenAPI/schema/examples/stubs/tests, not full validated platform. |
| T5.2 includes SUMO and potentially Vissim/Aimsun integration. | GA pp. 75-76, 100 | SUMO-first; Vissim/Aimsun as adapter contracts unless access/licence/support are confirmed. |
| T5.3 covers role-based UI and dashboards. | GA p. 76 | D5.1 should expose role-aware API payload assumptions, but not complete D5.3 UI. |
| WP6 evaluates usefulness, ease of use, emergency relevance, justice/equity outcomes. | GA pp. 76-77 | Add evaluation hooks and validation metadata. |
| WP10 appears to handle ongoing feedback/agile development, while D5.4 remains ambiguous. | GA pp. 81, 88, 100 | Clarify D5.4/T10.1 ownership and acceptance at Mini-GA. |
| Project risks include requirements, interoperability, simulator results, demos, KPI failure, cybersecurity, and AI bias. | GA pp. 110-113 | Add risk-to-test matrix and explicit status labels. |
| KER5 integrates Technologies 2-5. | GA p. 144 | Add component manifest with owner, licence, maturity, interface, and support contact. |

## 3. Direct Inputs To D5.1

| D5.1 section | Specific content to add |
|---|---|
| Scope and objective | Define D5.1 as API contract plus initial testability, not final production DSS. |
| Architecture | Separate common SUMA core, local pilot configuration, external partner modules, and deferred components. |
| Integration | Require adapter contracts for WP3/WP4/CUSP/simulator integrations. |
| API specification | Include OpenAPI, JSON Schema/Pydantic model mapping, examples, error model, auth assumptions, and async job pattern. |
| Governance | Add `ComponentManifest`, `AdapterContract`, `Requirement`, `RolePermission`, `EvaluationSession`, and `KpiEvidenceLedger`. |
| Risk management | Add risk-to-test matrix for interoperability, KPI, simulator, security, bias, and data-readiness risks. |

## 4. Delivery Staging Defaults

| Item | Recommended v0.3 default |
|---|---|
| D5.1 acceptance | OpenAPI/JSON Schema, JSON-LD-compatible DTOs, examples, error model, auth assumptions, mock/reference adapters, and initial tests. |
| T5.2 simulator scope | SUMO-first end-to-end reference integration. Vissim/Aimsun/VISUM are adapter contracts unless live access and licence/support are confirmed. |
| T5.3 UI scope | Role-based dashboards, scenario comparison, maps/charts, warnings, exports, and guided/expert workflows after API contract stabilises. |
| Near-real-time | Source-specific SLOs only. Otherwise label as offline, degraded, proxy, or manual import. |
| KPI acceptance | No KPI becomes evidence without formula, dataset, denominator, target/baseline, owner, and verification method. |
| Security/RBAC | Minimum baseline: roles, permissions, retention/redaction notes, audit records, export controls. |
| D5.4/WP10 | Clarify ownership before assigning ongoing agile-feedback obligations to Rhoe. |

## 5. Governance And Integration Records

### 5.1 `ComponentManifest`

```yaml
ComponentManifest:
  component_id: string
  component_name: string
  ker_ref: string | null
  provider: string
  owner: string | null
  licence_status: confirmed | pending | restricted | unknown
  deployment_mode: local | service | manual_import | external | unknown
  interface_status: documented | draft | missing | nda_blocked
  maturity_status: concept | prototype | tested | validated | operational | unknown
  support_contact: string | null
  d5_1_status: implement | stub | dependency | defer
```

### 5.2 `AdapterContract`

```yaml
AdapterContract:
  adapter_id: string
  module_name: string
  provider: string
  module_type: common_core | local_adapter | external_module
  implementation_status: implemented | mock | placeholder | external_dependency | deferred
  handover_format: code | service | script | api | report | unknown
  runtime_mode: local | service | manual_import | unknown
  input_schema_ref: string | null
  output_schema_ref: string | null
  parameter_contract: object | null
  result_contract: object | null
  kpi_mapping: object | null
  licence_or_nda_constraint: string | null
  maturity: draft | prototype | tested | validated | unknown
  fallback: string | null
  owner: string | null
```

### 5.3 `RequirementTrace`

```yaml
RequirementTrace:
  requirement_id: string
  source_wp: string
  requirement_text: string
  linked_use_cases: string[]
  linked_endpoints: string[]
  linked_modules: string[]
  verification_method: test | demonstration | inspection | stakeholder_review | performance_test
  acceptance_criterion: string
  owner: string | null
  status: accepted | essential | deferred | rejected | owner_missing
```

## 6. T5.2 Development Implications

| SUMA module | What to build | Notes |
|---|---|---|
| `/api/v1` contract layer | Versioned partner-facing API beside current GUI API. | Preserve existing operator endpoints. |
| Adapter registry | Store and expose adapter contracts for SUMO/WP3/WP4/CUSP/DMO. | Essential to avoid hidden assumptions. |
| Component manifest | Track KER/module owner, licence, interface, maturity, fallback. | Required before claiming integration. |
| Async job lifecycle | Formalize current job manager as `SimulationJob`. | Durable queue can wait until D5.2. |
| Role/audit baseline | Add minimal role permissions, audit records, retention/redaction notes. | T5.2/D5.3 bridge. |
| Evaluation hooks | Capture WP6 usefulness/ease/equity feedback. | Can start as stub/internal endpoint. |

## 7. Mini-GA Questions

Ask Rhoe/CU/all technical partners:

- Is D5.1 accepted as `spec only`, `spec + stubs/tests`, or `executable prototype`?
- Which endpoint families must be core D5.1 and which can be stubbed/deferred?
- Are Vissim/Aimsun/VISUM required as working integrations or documented adapter interfaces only?
- What does near-real-time mean per data source and endpoint?
- Who owns WP2/WP3/WP4/CUSP/DMO handoff formats, versions, and due dates?
- Which KER components are available, licensed, callable, documented, and supported?
- What is the minimum security/RBAC/export-control baseline for pilot demonstrations?
- Who owns D5.4/T10.1 agile-feedback acceptance?

Expected Mini-GA output: D5.1/T5.2 scope table with `item`, `implement/stub/dependency/defer`, `owner`, `interface`, `verification`, `due_date`, and `fallback`.

## 8. Technology Stack Note

The current stack is acceptable for the GA scope: Python/FastAPI/Pydantic/OpenAPI for D5.1, SUMO/TraCI first for T5.2, React/Vite/Leaflet/Recharts for D5.3, and Docker Compose for deployment. Add SQLite for metadata ledgers before adding Postgres/PostGIS; add Redis/Celery only when simulation/KPI jobs need durable queues. Avoid Kubernetes, full microservices, and full KG infrastructure for RP1 unless explicitly required.

