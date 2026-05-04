# SUMA Mini-GA Diplomatic Brief

Purpose: partner-facing talking-point brief derived from the internal SUMA WP5 Development Context and Mini-GA guide. It is suitable as raw material for slides, speaking notes, or consortium discussion.

## Suggested Opening Position

SUMA is being designed as the WP5 integration environment that connects AntifragiCity outputs from WP2, WP3, WP4, and the pilot demonstrations. The current SUMO-based implementation gives the consortium an executable scaffold: it can run simulations, introduce incidents, generate operational metrics, support batch analysis, and expose workflows through a GUI.

The current scaffold is intentionally being staged into an ontology-, KPI-, WP3-, and WP4-aligned integration environment. WP5 can operationalize partner outputs once interfaces, ownership, validation assumptions, and data-readiness constraints are explicit.

## What We Need To Leave With By 15:00 On 19 May

- A short list of priority use cases per pilot.
- A provisional D2.6 requirement shortlist, or an agreed fallback if the shortlist is not final.
- The first D2.5 ontology/data structures SUMA should support.
- Initial D2.7 KPI subsets per pilot, with available, proxy, missing, or deferred status.
- A function-to-API mapping for the first SUMA prototype.
- A common-core versus local-adaptation split for the SUMA DSS architecture.
- Initial WP3 triage and WP4 method interface expectations.
- Initial user roles, interface flows, data owners, validation targets, and open issue owners.

## Architecture Message

SUMA can be described as a five-layer API-driven environment:

1. Semantic input layer: events, scenarios, actors, spatial units, response actions, provenance, and data quality.
2. Pilot data layer: networks, demand, traffic observations, public transport data, local constraints, permissions, and validation data.
3. Simulation and method adapter layer: SUMO first, WP4 methods through adapters, routing interfaces, and other engines only if needed.
4. KPI and antifragility layer: KPI availability, calculation, scenario comparison, AF/SRI/equilibrium outputs, and validation labels.
5. User/API layer: REST endpoints, role-based GUI, reporting, export, and ontology/KG-compatible outputs.

## Decisions To Resolve

- Which pilot or use case should be the first implementation target. AHEPA is a focused emergency-access candidate, while the first implementation pilot should be selected by data readiness.
- What minimum pilot definitions are available: study area, critical assets, critical corridors, local constraints, data owner, and validation owner.
- What CUSP / Optimize AI information can be shared for WP5 architecture planning, including any NDA constraints.
- What ETH/WP4 can provide as a first method package: code, scripts, API, model, data, notebook, or specification.
- How WP3 triage, response actions, acceptability constraints, and vulnerability/equity information should be represented.
- Which user roles DMO and city partners expect for the first DSS/UI prototype.

## Partner Input Requests

- CU / ontology: first class subset, identifiers, JSON/JSON-LD expectations, competency questions, and CUSP/Optimize AI interface summary where possible.
- AUTH / KPI and city partners: pilot KPI subset, units, thresholds if known, input availability, and proxy/deferred labels.
- ETH / WP4: handover format, network and demand assumptions, control-zone definitions, outputs, parameters, and retraining assumptions.
- LISER / WP3 / routing partners: routing or response-action interface, acceptability constraints, and social/vulnerability variables.
- DMO / UI partners: user roles, permissions, dashboards, reports, export expectations, and guided versus expert workflows.
- WP6 / pilot partners: pilot study area, data inventory, critical routes/assets, validation target, local owner, and permission constraints.

## Diplomatic Lines

- "The current SUMO implementation gives us an executable scaffold for testing how the project concepts can connect."
- "For the first prototype, we propose ontology-aligned objects and API contracts, then fuller KG integration as the semantic infrastructure matures."
- "Where data or methods are still maturing, SUMA will support provisional or proxy modes with clear confidence and validation labels."
- "WP5 can operationalize WP4 methods once we agree the adapter contract: inputs, outputs, parameters, dependencies, and validation assumptions."
- "Rhoe is not asking for perfect datasets by August 2026. We need minimum viable definitions, data owners, confidence labels, permission constraints, and validation expectations."
- "The Mini-GA can help the consortium converge on minimum viable decisions and clearly record deferred items."

## Immediate Backlog After The Mini-GA

- Define event, scenario, KPI, provenance, and data-quality schemas.
- Encode the D2.1 event taxonomy as controlled vocabularies and seed catalogue data.
- Add a KPI registry and data-availability report.
- Wrap the current SUMO runner behind a simulation adapter interface.
- Define WP4 control-strategy and WP3 triage adapter placeholders.
- Add ontology mapping and JSON/JSON-LD export path.
- Create pilot configuration templates, with AHEPA emergency access as a focused candidate and final pilot order selected by data readiness.
