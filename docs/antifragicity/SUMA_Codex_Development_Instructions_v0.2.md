# SUMA WP5 Development Context And Mini-GA Internal Guide

Version: v0.2 working version.

Status: internal Rhoe working document.

Audience: Rhoe developers, research leads, WP5 contributors, and AI coding assistants working on SUMA.

Visibility: internal and frank. Do not circulate this version to the full consortium without converting the open risks, partner asks, and implementation caveats into diplomatic wording.

Date context: updated on 11 May 2026 for the Thessaloniki Mini-GA on 18-19 May 2026.

Canonical role: this file is the primary internal development context for SUMA. It replaces the older standalone Codex development instructions and incorporates the Mini-GA internal preparation report. Future WP3, WP4, WP6, and pilot inputs should extend this file rather than create competing internal guidance.

> **v0.2 update:** This version integrates new draft and working inputs received during 5-11 May 2026: D2.2 Social Acceptability Workshops draft, D2.4 initial survey analysis, CU's Mini-GA reference pack in `docs/antifragicity/miniGA/`, and the updated Mini-GA agenda `MiniGA_Thessaloniki_Agenda_LV1.docx`. Additions or altered guidance are marked with `v0.2 addition` or `v0.2 update` so they can be compared against `SUMA_Codex_Development_Instructions_v0.1.md`.

## 1. Executive Position

SUMA is being developed under WP5 as the Simulator for Urban Mobility Antifragility. The current Rhoe position is that SUMA should be an API-driven integration and orchestration environment, not a set of disconnected simulations and not a new WP4 traffic-control research package.

SUMA must eventually connect:

- WP2 outputs: event taxonomy, event catalogue, ontology, equilibrium concepts, antifragility indicators, and KPI framework.
- WP3 outputs: mobility triage, DSS logic, vulnerability/equity/acceptability information, and response-prioritisation logic once available.
- WP4 outputs: traffic-control methods, algorithms, modelling components, control strategies, routing methods, OD/demand methods, and operational scenarios once available.
- WP6 and pilot demonstration inputs: city-specific networks, demand data, event scenarios, study areas, local constraints, validation data, and feedback from living labs.

The current SUMA codebase already provides an executable SUMO-based scaffold with simulation, incident injection, metrics, batch analysis, OSM/gov.gr data tooling, and a GUI. It is useful as a proof of implementation capacity and as the place where future project concepts can be integrated. It is not yet a validated AntifragiCity digital twin, a full ontology/knowledge-graph system, a D2.7 KPI engine, a WP3 triage DSS, or a WP4 traffic-control integration platform.

The key internal rule is:

> Evidence says what the project expects; Rhoe defines an implementation position; current SUMA has a known capability baseline; unresolved dependencies must stay explicit.

> **v0.2 addition:** Social acceptability, equity, vulnerability, and governance are now implementation constraints for SUMA, not only UI or validation considerations. D2.2, D2.4, and the Mini-GA triage/acceptability references imply that SUMA must distinguish technical performance from social acceptability and must expose when acceptability evidence is absent, indicative, or pilot-specific.

## 2. Rhoe Mini-GA Battle Card

### 2.1 Objective

Use the Thessaloniki Mini-GA to move from conceptual alignment to implementable SUMA contracts:

- pilot use cases,
- D2.6 requirements shortlist,
- D2.5 ontology/data structures,
- D2.7 KPI subsets,
- SUMA functions,
- common-core versus local-adaptation modules,
- WP3/WP4 integration expectations,
- user roles,
- data owners,
- validation assumptions,
- open decisions and owners.

### 2.2 Non-negotiables

- SUMA must preserve the WP5 scope boundary: integrate, adapt, parameterise, operationalise, and validate; do not invent final WP3/WP4 science.
- Every antifragility-related output must carry calculation mode: `prototype`, `proxy`, `theoretical`, `calibrated`, or `validated`.
- Pilot-specific zones, critical corridors, policy restrictions, emergency routes, and local parameter choices must be provided or validated by pilot/WP6 partners.
- D2.3 equilibrium and antifragility models must be treated as theoretical until calibrated and validated.
- D2.7 KPIs must be data-aware. Do not make the full theoretical KPI catalogue mandatory for all pilots.
- WP4 components must be integrated through adapter contracts, not hard-wired into SUMA internals.
- Ontology alignment should start with stable schemas and exportable records before full knowledge-graph deployment.

> **v0.2 addition:** SUMA optimisation outputs must be screened through explicit social-acceptability constraints: fairness, accessibility, proportionality, reversibility, transparency, vulnerable-user protection, and distribution of burdens across neighbourhoods. Do not present a technically optimal response as socially acceptable unless these constraints are evaluated or explicitly marked `not_assessed` or `indicative_only`.

### 2.3 Negotiables

- Whether D5.1 exposes new `/api/v1/*` endpoints or first extends existing `/api/*` GUI endpoints.
- Which pilot becomes the first implementation target. AHEPA is a strong focused case for emergency access, but the first pilot should be selected by data readiness.
- Whether JSON-LD export is included in the first D5.1 prototype or staged immediately after.
- How much role-based UI is implemented before T5.3 becomes active.
- Whether top-down stress tests are presented as first-class scenarios or experimental scenario generation.

### 2.4 Decisions needed by 15:00 on 19 May 2026

- For each pilot: one to three priority use cases.
- For each use case: event/disruption type, actor, affected asset/spatial unit, decision question, and intended value of SUMA.
- For each pilot: minimum data inventory and data owner.
- For each pilot: initial KPI subset with `available`, `proxy`, `missing`, or `deferred` status.
- For WP4: handover format, inputs, outputs, parameters, retraining assumptions, and first candidate method for SUMA adapter integration.
- For WP3: expected triage/response-action representation and acceptability constraints.
- For CU/ontology: mandatory first-prototype ontology classes, identifiers, JSON/JSON-LD expectations, and first competency questions.
- For DMO/UI: user roles, role permissions, and first interface flows.
- For WP6/pilot partners: validation target, local owner, and minimum evidence for pilot-readiness.

> **v0.2 update:** The updated Mini-GA agenda defines Day 2 as the WP5 conversion point. By the close of Day 2, Rhoe must classify each retained pilot use case into a traceable implementation state: `use case -> requirement -> ontology/data object -> KPI -> SUMA function -> API/service -> architecture module -> UI role -> backlog owner`.

### 2.5 Fallback if decisions are not ready

If a partner cannot provide a final answer, record the item as one of:

- `decision`: agreed and ready for implementation planning,
- `assumption`: usable temporarily but must be confirmed,
- `dependency`: blocks implementation until provided,
- `risk`: threatens scope, schedule, or validation quality,
- `owner_missing`: technically actionable but no responsible partner or data owner is assigned,
- `deferred`: explicitly out of first prototype scope.

Do not let missing partner inputs become implicit Rhoe commitments.

## 3. Evidence Traceability

### 3.1 Grant Agreement

Key evidence:

- Grant Agreement, p. 75: WP5 aims to assemble WP2 outputs, including the event ontology, WP3, and WP4 into a user-friendly software environment compatible with existing transportation simulation tools and standalone versions.
- Grant Agreement, p. 75: T5.1 includes a core simulation engine, resilience and antifragility metrics, event ontology/data-processing layer, and modular/extensible architecture.
- Grant Agreement, p. 75-76: T5.2 expects integration with SUMO and potentially Vissim/Aimsun, plus data synchronization, integration documentation, guides, and training.
- Grant Agreement, p. 76: T5.3 expects role-based UI, dashboards, maps, network-flow diagrams, heatmaps, time-series analytics, scenario comparison, and data export.
- Grant Agreement, p. 77: WP6 validation evaluates usefulness, ease of use, emergency-management relevance, and justice/equity outcomes.
- Grant Agreement, p. 100: D5.1 is the Core SUMA API specification, due M16, and should support initial testing.
- Grant Agreement, p. 136: SUMA/KER5 should assemble the resilience framework, event ontology, mobility triage, and response logic into a web-based service suite.
- Grant Agreement, p. 144: KER5 is described as a comprehensive software platform integrating GIS visualization, data-driven traffic control, and routing tools.

Implementation position:

- Treat SUMA as integration-heavy DSS/API infrastructure.
- Keep SUMO as the first concrete simulation adapter.
- Keep extension points for other simulation tools only where the architecture needs them.
- Keep role-based UI and validation/equity outcomes in scope, but stage implementation.

### 3.2 D2.1 Urban Event Mapping

Key evidence:

- D2.1, Ch. 2.1, p. 12-14: events are classified by four domains: transport-related, environment/weather, utilities/connectivity, and public space/social.
- D2.1, Ch. 2.1, p. 13: event scale is daily stressor, mid-scale disruption, or large-scale crisis.
- D2.1, Ch. 2.1, p. 13-14: severity uses levels 1-5: very low/informational, low/background, moderate/standard, high/priority, critical/immediate.
- D2.1, Ch. 2.2, p. 14: cross-walk maps survey domains to social-media categories and EM-DAT/Copernicus groups.
- D2.1, Ch. 3.3.1.2, p. 21-24: survey findings cover severity, climate exposure, coping strategies, preparedness, and city-performance perceptions.
- D2.1, Annex 3, p. 99 onward: the Urban Event Catalogue contains 76 seed events from EM-DAT, survey, and social-media analysis.

Implementation position:

- Encode D2.1 as controlled vocabularies and seed catalogue data.
- Treat the 76 events as seed data, not as a closed universe.
- Make accident incidents one disruption subtype, not the whole disruption model.

### 3.2a v0.2 addition: D2.2 Social Acceptability Workshops Draft

Key evidence:

- D2.2 draft, executive summary, p. 13: citizens' forums in Bratislava, Larissa, and Thessaloniki assessed social acceptability of urban mobility interventions under everyday and disruption conditions.
- D2.2 draft, p. 22-23: the Odesa workshop was suspended due to the security situation; Odesa acceptability evidence must therefore be treated as substituted or incomplete until a tailored baseline is available.
- D2.2 draft, p. 203-205: social acceptability is highest for measures that improve predictability, safety, reliability, fairness, accessibility, public transport, and protection of vulnerable users.
- D2.2 draft, p. 205-210: transparent decision-making, trusted communication, non-digital channels, participatory engagement, and inclusive emergency planning are central to legitimacy.
- D2.2 draft, Thessaloniki forum synthesis, p. 176-184: relevant recommendations include real-time data sharing, coordination among authorities, inclusive emergency mobility planning, priority support for vulnerable groups, interoperable digital systems, and multimodal/public-transport improvements.

Implementation position:

- Treat D2.2 as deliberative and context-sensitive evidence, not as a statistically representative population estimate.
- Encode social acceptability as an explicit assessment layer for response actions and triage measures.
- Preserve non-digital communication and feedback channels in the future UI/API story.
- Require acceptability evidence origin and confidence metadata on any recommendation that claims to be socially acceptable.

### 3.2b v0.2 addition: D2.4 Initial Social Acceptability Survey Analysis

Key evidence:

- D2.4 initial analysis reports strong support for better public transport (92.5%), digital traffic management (66.25%), better active mobility infrastructure (67.5%), and restrictions on cars in dense areas (57.5%).
- D2.4 initial analysis: 80% supported implementing a citywide-beneficial policy that worsens one neighbourhood only if mitigation measures are introduced.
- D2.4 initial analysis: 82.5% agreed transport policies should not unfairly disadvantage certain neighbourhoods; 89.74% agreed cities should monitor which groups or neighbourhoods are most affected by disruptions.
- D2.4 initial analysis: 97.5% prioritised emergency services during disruptions, and 50% prioritised public transport users.
- D2.4 initial analysis: 85% agreed real-time travel information would make them more likely to follow transport rules during disruptions.

Implementation position:

- Treat D2.4 as initial, item-level survey evidence with varying sample sizes around N=73-80; percentages should carry sample/context notes.
- Use D2.4 to justify first-class fields for fairness, neighbourhood burden, emergency access, public-transport continuity, mitigation requirements, trust, and communication readiness.
- Do not treat social acceptability as a single score without preserving the underlying dimensions and uncertainty.

### 3.3 D2.3 Establishing Urban States Of Equilibrium

Key evidence:

- D2.3 abstract, p. 2: results are methodological and no empirical pilots are reported.
- D2.3 executive summary, p. 8: the framework includes target-setting, three-layer equilibrium, Antifragility Factor, System Responsiveness Index, and validation gates.
- D2.3 Ch. 2.3, p. 9-10: theoretical constructs, assumptions, and decision rules are defined; no calibration or case studies are included.
- D2.3 Ch. 3.1, p. 12: the urban mobility system is represented as `G = (S,u)`, where `S` is a state vector and `u` maps states to KPI values.
- D2.3 Ch. 4, p. 14-17: the 12 indicators are `M`, `R`, `A`, `C`, `D_eff`, `S`, `Q`, `P`, `B`, `E_energy`, `E_ICT`, and `I`.
- D2.3 Ch. 4.4.1, p. 17: the minimal viable model uses `M`, `R`, `C`, `D_eff`, `S`, and `I`.
- D2.3 Ch. 5, p. 18-23: the target-setting model defines `E`, target performance, entropy normalization, parameters, update rules, disturbance calculation, and `AF`.
- D2.3 Ch. 7, p. 34-35: target-setting and equilibrium models exchange target performance, KPI values, AF validation, realized performance, network state, and adaptation history.
- D2.3 Ch. 8, p. 36-38: validation requires pre-event baseline, disruption monitoring, recovery, stabilization windows, statistical tests, causal validation, equity checks, and long-term confirmation.
- D2.3 Appendix D, p. 58: AHEPA/Thessaloniki should focus on emergency access reliability, critical hospital access links, disability-sensitive equity, medical transport modes, and emergency response time or hospital accessibility.

Implementation position:

- Implement D2.3 initially as transparent theoretical/proxy capability.
- Do not claim validated antifragility without calibration and empirical validation.
- Preserve AF, SRI, `D_eff(t)`, minimal viable indicators, validation gates, and calculation-mode metadata.

### 3.4 D2.5 Ontology

Key evidence:

- D2.5 abstract, p. 1-2: the ontology formalizes entities, processes, events, states, indicators, governance, equity, and learning, and is designed for knowledge-graph (KG) integration with digital twins, MAS, and AI analytics.
- D2.5 Ch. 4.1, p. 39-40: AntifragiCity must represent acute disruptions, chronic stresses, systemic crises, scenario evaluation, equity, governance, heterogeneous data, and cascading effects.
- D2.5 Ch. 4.2, p. 40-42: requirements cover mobility performance, environment/energy, equity/vulnerability, disruptions/hazards, scenarios/interventions, and antifragility assessment.
- D2.5 Ch. 4.4, p. 46: the 12 top-level classes include TransportElement, MobilityService, DisruptionEvent, SystemState, Scenario, EnvironmentFactor, Actor, GovernanceInstrument, SpatialUnit, TemporalUnit, Observation/DataProvenance/KPI, and ResponseAction/LearningProcess.
- D2.5 Ch. 4.6, p. 54: competency questions should drive validation.
- D2.5 Ch. 8.2, p. 112-113: semantic lifting should convert GIS, IoT, mobility APIs, and simulation outputs into ontology-aligned records with uncertainty, provenance, and data quality.
- D2.5 Ch. 8.3.2, p. 120: Thessaloniki/AHEPA use case is emergency access and multi-agency coordination.
- D2.5 Ch. 10, p. 147-153: ontology validation uses competency questions, expert/stakeholder validation, performance/scalability, completeness, and extensibility.
- D2.5 Appendix B/C, p. 198-199: KPI observations link to Scenario and SystemState and include value, unit, time, provenance, data quality, trust, confidence, uncertainty, source, and observed object.
- D2.5 Appendix D, p. 238-241: mapping to standards includes GeoSPARQL, SOSA/SSN, PROV-O, OWL-Time, and novel AntifragiCity concepts.

Implementation position:

- Build ontology-aligned internal schemas before full knowledge-graph deployment.
- Add stable identifiers, provenance, uncertainty, data quality, and semantic mapping from the start.
- Keep full RDF/SPARQL/SHACL integration staged unless project architecture requires it earlier.

### 3.5 D2.7 KPI Framework

Key evidence:

- D2.7 abstract and executive summary, p. 1-10: the framework is adaptive, dynamic, modular, uses IPOO, integrates the ontology and LCA, and supports DSS, mobility triage, and the SUMA API.
- D2.7 Ch. 3.1, p. 28: the KPI framework supports context-specific dashboards and the D2.3 equilibrium approach.
- D2.7 Ch. 3.2, p. 30-31: 2540 initial indicators were refined into 198 groups; 11 D2.3 indicators were added for 209 KPI groups across seven thematic groups.
- D2.7 Ch. 3.4, p. 34-37: city practitioners select KPIs unevenly, with strong city-by-city variation.
- D2.7 Ch. 6.2.1, p. 80-81: AHEPA/Thessaloniki should emphasize emergency vehicle access, hospital operational continuity, and public health outcomes.
- D2.7 Ch. 6.3, p. 81: KPIs support WP3 mobility triage and WP4 traffic control strategy refinement.
- D2.7 Ch. 6.3.1, p. 82: the SUMA API should use KPIs as core metrics, monitoring capability, and analytical foundation.
- D2.7 Appendix C, p. 108-110: context-specific KPI selection should follow strategic vision, thematic prioritization, operational objectives, constraints/scenarios, KPI filtering/scoring, redundancy removal, stakeholder validation, and iteration.

Implementation position:

- Implement a KPI registry and availability engine, not one mandatory fixed KPI list.
- Support KPI availability statuses separately from KPI tiers or scopes.
- Support equity/governance metadata as first-class assessment context, even where advanced calculations are deferred.
- Treat IPOO and LCA as extension points unless immediate data and methodology are available.

### 3.5a v0.2 addition: D2.6-derived SUMA Requirements Reference

Key evidence:

- CU Mini-GA Requirements Reference: D2.6-derived requirements were refined from 48 Delphi-elicited candidates into 20 SUMA-bound requirements across six categories: functional, data, interoperability, usability, non-functional, and governance.
- Functional requirements: FR-01 disruption/stressor representation, FR-02 scenario modelling, FR-03 resilience/antifragility assessment, FR-04 intervention exploration, FR-05 open documented API, FR-06 near-real-time data integration, and FR-07 learning loop.
- Data requirements: DR-01 data source diversity and DR-02 provenance, quality, and uncertainty handling.
- Interoperability requirements: IR-01 external simulator integration and IR-02 modular/extensible architecture.
- Usability requirements: UR-01 role-based access/workflows and UR-02 scenario comparison/dashboard interpretability.
- Governance requirements: GR-01 privacy/access/retention, GR-02 auditability/accountability, GR-03 transparency of assumptions/limitations, and GR-04 secure stakeholder input governance.

Implementation position:

- Use the 20 requirements as the primary bridge from Mini-GA outputs to D5.1 API sections.
- Every retained use case should map to one or more requirement codes.
- D5.1 should include endpoint/resource coverage for all requirements, even where the first implementation uses placeholders.

### 3.5b v0.2 addition: CU Mini-GA KPI, Ontology, Data Inventory, Use-Case, And Triage References

Key evidence:

- KPI Register: indicators are organised into three layers: Layer 1 observed city indicators, Layer 2 D2.3 equilibrium state variables, and Layer 3 SUMA platform/performance indicators.
- KPI Register: 236 distinct indicators are consolidated across project proposal, D2.7, D2.3, D2.6, and D3.3; 209 sit in the urban indicator catalogue, including 11 equilibrium indicators.
- KPI Register: Layer 3 includes SUMA-side targets such as ingestion latency, scenario evaluation time, response deployment time, live traffic update interval, trust score, proactive rerouting accuracy, modal-shift nudging effectiveness, antifragility trajectory, and cross-city benchmarking.
- Data Inventory Report: the consolidated inventory contains 266 variables. Filtering by `data_role=input` shows that the real high-priority external acquisition target is 23 variables, not 85.
- Data Inventory Report: output variables and calibration parameters should not be requested from pilot partners as if they were raw data.
- Ontology Reference: Mini-GA partner-facing ontology work uses 12 top-level classes and principal relationships such as `affects`, `respondsTo`, `updatesState`, `propagatesTo`, `locatedIn`, `dependsOn`, `governedBy`, `implementedBy`, `hasObservation`, and `evaluatedIn`.
- Use-Case Shortlist: pilot candidates now include 5 Bratislava, 4 Larissa, 5 Odesa, and 4 AHEPA use cases, with retain/refine/set-aside expected during Day 1.
- Triage Measures Reference: Session 7 should reconcile expert-derived technical priority with citizen-derived social acceptability; high-priority/low-acceptability measures require communication, mitigation, or design adjustment rather than automatic rejection.

Implementation position:

- Add `Requirement`, `AcceptabilityConstraint`, `EquityImpact`, `DataInventoryVariable`, and `UserRole` concepts to the internal design vocabulary.
- Use `role` and `data_role` metadata in the KPI/data registry so partner data requests do not ask for computed outputs or calibration parameters.
- Treat Layer 3 SUMA KPIs as D5.1/D5.2 specification items, not as pilot acquisition items.
- Use the Mini-GA reference pack as working input, not as final project law; outputs must be confirmed during the charrette and then translated into D5.1.

### 3.6 WP5 Meetings And WP4-WP5 Workshop

Key evidence:

- WP5 Rhoe Internal Understanding, "High-level WP5 role": WP5 is an integration environment connecting WP2, WP3, WP4, and WP6.
- WP5 Rhoe Internal Understanding, "Relationship between WP4 and WP5": WP4 provides generic methods; WP5 performs integration, adaptation, and operationalization.
- WP5 Rhoe Internal Understanding, "Open questions": WP4 output form, D4.1-D4.5 computability, OD data form, stable versus experimental components, orchestration responsibility, parameter configurability, minimum pilot dataset, and top-down scenario generation remain open.
- WP5 MoM, 1 April 2026: WP5 is in early T5.1 specification; T5.2 has not started; T5.3 starts later.
- WP5 MoM, 1 April 2026: WP4 methods may be scalable in principle but are not directly transferable without city-specific adaptation.
- WP5 MoM, 22 April 2026: August/M16 is a critical milestone month due to D5.1, D5.4, and the related milestone.
- WP5 MoM, 22 April 2026: SUMA should accept disruption events, pilot-specific data, and mobility triage inputs; use SUMO and WP4 methods; and produce mobility, resilience, antifragility, and environmental indicators.
- WP5 MoM, 22 April 2026: protected/congestion-zone definitions require WP6/pilot input and should be relatively stable because learning-based methods may require retraining.
- WP5 MoM, 22 April 2026: bottom-up event workflows and top-down exploratory stress tests are both relevant.
- WP4-WP5 Workshop MoM, 29 April 2026: ETH presented D4.1, D4.2 progress, T4.3 relation to T4.1/T4.2, T4.4 workflow, Luxembourg network, conversion from numerical simulation to SUMO, SUMO inputs/outputs, and antifragile perimeter control.
- WP4-WP5 Workshop MoM, 29 April 2026: ETH identified skewness as an indicator and noted a TODO to replace the current efficiency indicator with D2.7 indicators.

Implementation position:

- Treat WP4 outputs as adapter candidates that need input/output contracts.
- Treat protected/congestion zones as pilot-owned configuration, not Rhoe assumptions.
- Support both bottom-up and top-down workflows through one scenario schema with different provenance.

### 3.7 Mini-GA Agenda

Key evidence:

- Mini-GA Agenda, Sec. 1: the charrette should synthesize deliverables, refine pilot interventions, provide early SUMA specification/architecture, and deliver a first prototype definition.
- Mini-GA Agenda, Sec. 2: outputs include use cases, requirements from D2.6, D2.5 data structures, D2.7 KPI subsets, API/functionality mapping, first SUMA DSS architecture, triage measures, user interfaces, and perceived usefulness/ease-of-use discussion.
- Mini-GA Agenda, Day 1 Sessions 1-4: use cases, requirements, ontology/data structures, and KPIs.
- Mini-GA Agenda, Day 2 Sessions 5-6: translate requirements into SUMA functionality and design the SUMA DSS architecture.
- Mini-GA Agenda, Criteria for Success: the workshop succeeds if it produces use cases, requirements, ontology-based data structures, KPIs, functionality mapping, DSS architecture, triage measures, common/local module split, and UI concept.

Implementation position:

- Rhoe should bring canvases that convert meeting discussion into rows that can become D5.1 API and architecture content.
- Missing D2.6 is an active readiness risk. If the D2.6 shortlist is unavailable before Day 1 Session 2, use provisional requirement categories and mark all mappings as pending confirmation.

> **v0.2 update:** `MiniGA_Thessaloniki_Agenda_LV1.docx` resolves much of the earlier D2.6 uncertainty by providing a D2.6-derived requirements reference for the charrette. Rhoe should still treat the Mini-GA output as the confirming source: the relevant artifact is not the full 20-requirement list alone, but the per-pilot retained/refined requirement subset and its mapping to ontology objects, KPIs, SUMA functions, architecture modules, triage measures, UI roles, and backlog owners.

> **v0.2 addition:** The updated agenda places Rhoe in a lead/support role for Session 5, `Translating requirements into SUMA functionality`, and Session 6, `Designing the architecture of the SUMA decision-support system`, on Tuesday 19 May 2026. Rhoe should not let Day 2 reopen broad use-case debates. Use the rule: if a use case or requirement was not selected on Day 1, it goes to the backlog unless it blocks the minimum viable SUMA prototype.

## 4. Current SUMA Baseline

### 4.1 Existing capabilities

Current SUMA provides:

- SUMO execution through TraCI.
- Probabilistic incident triggering based on speed, speed heterogeneity, density, and road type.
- Incident lifecycle management.
- Lane closures, lane speed degradation, and local rerouting.
- Network metrics and accident reports.
- Batch execution, resilience assessment, MFD analysis, and sweep workflows.
- OSM acquisition, city generation, and Thessaloniki-oriented gov.gr traffic-feed tooling.
- FastAPI backend under `src/suma/gui/`.
- React frontend under `frontend/`.
- Documentation view that renders markdown files from `docs/`.

Current documented output artifacts include:

- `network_metrics.csv`
- `accident_reports.json`
- `antifragility_index.json`
- `metadata.json`
- `report.html`
- assessment and visualization outputs from analysis workflows.

### 4.2 Current limitations

The current implementation is a SUMO-based stochastic incident and resilience prototype. It is not yet:

- ontology-integrated,
- D2.1 event-catalogue driven,
- D2.7 KPI-framework driven,
- WP3 triage integrated,
- WP4 control-method integrated,
- validated against pilot observations,
- a full D2.3 equilibrium/AF/SRI implementation.

Known scientific and technical limits:

- Runtime behavior is most coherent at `sumo.step_length: 1`.
- Current rerouting is proximity-based, not full path-exposure based.
- `lane_capacity_fraction` is an operational proxy, not a calibrated lane-drop discharge model.
- Demand realism remains a major external-validity gap when using `randomTrips.py`.
- The risk model is an incident-propensity proxy, not a validated road-safety model.
- The current Antifragility Index is speed-centered and narrower than the D2.3 AF framework.
- Compound incidents are intentionally suppressed by default.

### 4.3 Repository baseline preconditions

Before using current SUMA examples as D5.1 evidence:

- Ensure one canonical default config exists and is runnable.
- Align README, Makefile, GUI workflow defaults, and `DEFAULT_CONFIG_PATH`.
- Do not build D5.1 examples on stale paths such as deleted or renamed Thessaloniki configs.
- Keep large generated city artifacts out of git unless deliberately tracked.
- Keep `src/sas/` as a compatibility shim only; new work belongs under `src/suma/`.
- Keep current package boundaries unless a migration is explicitly planned.

## 5. Scope Boundary

### 5.1 SUMA should do

SUMA should:

- Provide an API and software layer for configuring, executing, and evaluating urban mobility antifragility simulations and assessments.
- Accept structured disruption events and scenarios.
- Accept city-specific network, demand, KPI, and parameter inputs.
- Integrate WP4 methods through adapters.
- Represent WP3 triage outputs and response actions once available.
- Use D2.1, D2.3, D2.5, and D2.7 as semantic and analytical foundations.
- Produce mobility, resilience, antifragility, environmental, equity, and governance-relevant observations where data allow.
- Support baseline, disruption, response, adaptation, and stress-test scenario comparison.
- Support partial-data workflows with explicit missing-input reports.
- Preserve traceability, assumptions, confidence, provenance, and validation status.

### 5.2 SUMA should not do unless explicitly required later

SUMA should not:

- Develop entirely new traffic-control algorithms if WP4 is expected to deliver them.
- Invent pilot-specific zones, restrictions, emergency corridors, policy measures, or control parameters without pilot/WP6 validation.
- Treat D2.3 theoretical models as calibrated software.
- Claim full antifragility validation from proxy metrics.
- Depend on unavailable datasets as mandatory for all workflows.
- Hard-code one pilot's local logic into the core system.
- Couple core API logic directly to one WP4 implementation format.

### 5.3 Missing-input behavior

Where inputs are missing, SUMA should expose one of:

- validation error,
- availability warning,
- proxy calculation,
- mock adapter,
- deferred KPI,
- degraded-mode execution,
- explicit "not computable" result.

Silent omission is not acceptable.

## 6. Core Domain Model

### 6.1 Required objects

The unified domain model should include:

- `DisruptionEvent`
- `Scenario`
- `SimulationRun`
- `SystemState`
- `KpiDefinition`
- `KpiInputRequirement`
- `KpiObservation`
- `KpiAvailabilityReport`
- `ScenarioKpiComparison`
- `ResponseAction`
- `TriageRecommendation`
- `ControlStrategy`
- `PilotConfiguration`
- `DataQuality`
- `ProvenanceRecord`
- `DisturbanceProfile`
- `AntifragilityAssessment`

> **v0.2 addition:** Add these concepts to the design vocabulary as first-class or near-first-class objects: `Requirement`, `AcceptabilityConstraint`, `EquityImpact`, `UserRole`, `DataInventoryVariable`, `CommunicationChannel`, `AuditRecord`, and `StakeholderInput`. These objects are needed to reflect the D2.6-derived requirement reference, D2.2/D2.4 social acceptability evidence, Mini-GA data inventory, and Day 2 UI/architecture sessions.

### 6.2 Event model

Minimum fields:

```json
{
  "id": "event_001",
  "name": "Localised flooding on corridor",
  "domain": "environment_weather",
  "scale": "mid_scale_disruption",
  "severity": 4,
  "start_time": "2026-01-01T08:00:00Z",
  "end_time": "2026-01-01T12:00:00Z",
  "spatial_scope": {},
  "affected_modes": ["road", "public_transport"],
  "affected_mobility": {
    "passenger": true,
    "freight": false,
    "infrastructure": true,
    "emergency_access": true,
    "equity_sensitive_groups": false
  },
  "source": "manual",
  "evidence_origin": "pilot_partner",
  "provenance": {},
  "confidence": 0.8,
  "validation_status": "unvalidated"
}
```

Controlled vocabularies must include:

- domain: `transport_related`, `environment_weather`, `utilities_connectivity`, `public_space_social`
- scale: `daily_stressor`, `mid_scale_disruption`, `large_scale_crisis`
- severity: 1-5
- source type: `catalogue`, `manual`, `sensor`, `operator_log`, `official_log`, `survey`, `social_media`, `synthetic`

### 6.3 Scenario model

A scenario should combine:

- `scenario_id`
- scenario type: `baseline`, `disruption`, `response`, `adaptation`, `stress_test`
- baseline configuration
- one or more events
- network and demand references
- response action or control strategy
- simulation engine
- KPI set
- assumptions
- evidence origin
- acceptability assessment status
- equity context
- provenance
- validation status

> **v0.2 addition:** Requirements, scenarios, response actions, and constraints must record evidence origin: `citizen_forum`, `survey`, `delphi_expert`, `pilot_partner`, `deliverable`, `sensor`, `synthetic_stress_test`, or `developer_assumption`. Bottom-up acceptability evidence and top-down technical requirements must be reconciled, not collapsed into one opaque priority score.

### 6.4 Scenario migration note

The current repo already has `Scenario` in `src/suma/analysis/scenario_generator.py`, where it means demand-period x incident-probability x seed for resilience assessment. That object is not the same as an ontology-aligned D5.1 scenario.

Required migration:

- Keep the current assessment object stable in the short term.
- Rename or document it as `AssessmentScenario` when touching that module.
- Add the richer D5.1 scenario in a new domain module such as `suma.scenarios.models`.
- Provide a translator from YAML/resilience assessment configs into the D5.1 scenario contract.

### 6.5 KPI observation model

Minimum fields:

- `kpi_id`
- `name`
- `value`
- `unit`
- `scenario_id`
- `run_id`
- `system_state_id`
- `spatial_scope`
- `temporal_scope`
- `calculation_method`
- `calculation_mode`
- `required_inputs`
- `missing_inputs`
- `data_quality`
- `confidence`
- `provenance`

### 6.6 Data quality model

Use fields such as:

- `source`
- `provenance`
- `confidence`
- `trust_score`
- `resolution_spatial`
- `resolution_temporal`
- `is_observed`
- `is_synthetic`
- `is_estimated`
- `validation_status`
- `missing_inputs`
- `known_limitations`

### 6.7 v0.2 addition: Acceptability and equity context model

Minimum fields for `AcceptabilityConstraint` or `EquityImpact` records:

- `affected_groups`
- `vulnerable_group_exposure`
- `neighbourhood_burden_delta`
- `essential_service_access_delta`
- `public_transport_dependency`
- `digital_exclusion_risk`
- `privacy_or_surveillance_risk`
- `mitigation_required`
- `mitigation_actions`
- `equity_non_worsening_status`
- `communication_readiness`
- `non_digital_channel_available`
- `acceptability_assessment_status`
- `evidence_origin`
- `confidence`
- `validation_status`

Separate assessment, mitigation, and validation status concepts:

- `acceptability_assessment_status`: `not_assessed`, `indicative_only`, `assessed`, or `conflicting_evidence`
- `mitigation_status`: `none_required`, `mitigation_required`, or `mitigation_defined`
- `validation_status`: `unvalidated`, `pilot_validated`, or `constituency_validated`

## 7. KPI And Antifragility Strategy

### 7.1 KPI statuses

Every KPI should have one of:

- `available`: computable with current data,
- `proxy`: computable through declared approximation,
- `missing`: required inputs unavailable,
- `deferred`: out of first prototype scope,
- `not_applicable`: not relevant to the selected pilot/use case.

Every KPI should also have a separate tier or scope, for example:

- `core`
- `extended`
- `layer_1`
- `layer_2`
- `layer_3`

### 7.2 KPI groups

SUMA should support KPI grouping by:

- network/system performance,
- travel time and reliability,
- accessibility,
- safety,
- equity,
- environmental impact,
- resilience,
- antifragility,
- governance/institutional readiness,
- lifecycle/LCA extension where relevant.

### 7.3 AF/SRI calculation modes

Use explicit calculation modes:

- `prototype`: useful for testing software flow,
- `proxy`: measurable but not project-validated,
- `theoretical`: follows D2.3 specification but lacks calibration,
- `calibrated`: calibrated against accepted data,
- `validated`: empirically validated against agreed criteria.

The current speed-window Antifragility Index belongs in `prototype` or `proxy` mode unless and until the consortium validates it.

### 7.4 Minimal viable D2.3 indicators

D2.3's minimal viable model uses:

- `M(t)`: mobility throughput,
- `R(t)`: redundancy ratio,
- `C(t)`: network information entropy,
- `D_eff(t)`: effective disturbance,
- `S(t)`: stress,
- `I(t)`: intermodal synergy.

SUMA should allow this subset to operate even when extended indicators are unavailable.

### 7.5 v0.2 addition: Social acceptability, equity, and triage constraints

SUMA must evaluate response actions against social acceptability and equity constraints before labelling them socially acceptable or implementation-ready. Minimum checks are:

- emergency access preservation,
- public-transport continuity,
- vulnerable-group impact,
- neighbourhood burden distribution,
- essential-service accessibility,
- digital-exclusion risk,
- privacy and data-minimisation status,
- communication and explainability readiness,
- reversibility and proportionality,
- mitigation requirements.

Each recommendation should expose:

- `technical_priority`,
- `social_acceptability_status`,
- `equity_status`,
- `evidence_origin`,
- `confidence`,
- `validation_status`.

Where evidence is unavailable or non-representative, SUMA must report `not_assessed` or `indicative_only`, not infer public acceptance from technical performance.

First response-action categories should prioritise:

- emergency corridors,
- public-transport priority,
- adaptive multimodal routing,
- accessible routing,
- priority support for vulnerable groups,
- non-digital information channels,
- post-event reporting and learning.

Pricing-heavy or punitive demand-management measures should remain possible but must carry stronger acceptability and mitigation checks.

### 7.6 v0.2 addition: Three-layer KPI and data registry strategy

Use the Mini-GA KPI Register's three-layer structure:

- Layer 1: observed urban/city indicators reported by pilots.
- Layer 2: computed D2.3 equilibrium state variables.
- Layer 3: SUMA platform and analytical performance KPIs.

KPI and data registry entries must distinguish `role` from `data_role`:

- `role`: `KPI`, `input_variable`, `computed_variable`, or `parameter`.
- `data_role`: `input`, `output`, or `parameter`.

Partner data requests must filter to `data_role=input`. Computed outputs and calibration parameters should be handled through software readiness and calibration plans, not external acquisition asks.

Current Mini-GA data inventory implication:

- The consolidated inventory contains 266 variables.
- The true high-priority external acquisition target is 23 input variables, not 85 mixed variables.
- Layer 3 SUMA KPIs are outputs by definition and require formal software/API specification, not pilot data acquisition.

Open KPI issue:

- The Mini-GA KPI Register notes unresolved target/method questions, including an ARP target inconsistency of 90% versus 95%. Do not hard-code unresolved Layer 3 targets into SUMA until confirmed by the consortium.

## 8. Architecture And API

### 8.1 Five-layer architecture

SUMA should be described as five layers:

1. Semantic input layer: events, scenarios, spatial units, actors, response actions, provenance, and data quality.
2. Data and pilot configuration layer: networks, demand, traffic counts, public transport data, OD matrices, event logs, sensor feeds, city constraints, and pilot-specific parameters.
3. Simulation and method adapter layer: existing SUMO runner, mock adapter, WP4 control adapters, routing adapters, and future simulator adapters if required.
4. KPI and antifragility assessment layer: KPI registry, availability reports, calculators, scenario comparison, and D2.3 theoretical/proxy AF/SRI/equilibrium outputs.
5. API and user-facing layer: existing FastAPI GUI backend, future D5.1 API endpoints, role-based GUI, reporting, export, documentation, and knowledge-graph/JSON-LD export.

### 8.2 Additive module migration

Preserve the current package layout:

- `src/suma/app`
- `src/suma/core`
- `src/suma/simulation`
- `src/suma/analysis`
- `src/suma/generators`
- `src/suma/integrations`
- `src/suma/gui`
- `src/suma/tools`
- `src/suma/visualization`

Add domain modules incrementally:

- `src/suma/events`
- `src/suma/scenarios`
- `src/suma/kpis`
- `src/suma/ontology`
- `src/suma/equilibrium`
- `src/suma/control`
- `src/suma/triage`
- `src/suma/pilots`

Do not create parallel repo roots such as nested `tests/` or `docs/` under `src/suma`. Keep tests in the repository-level `tests/`.

### 8.3 Current API and D5.1 migration contract

The current operator API is under `/api/*` in `src/suma/gui/app.py` and is already used by the React frontend. D5.1 endpoints should extend or version this API, not accidentally replace it.

Migration table:

| Current capability | Current endpoint area | D5.1 endpoint candidate | Compatibility position |
|---|---|---|---|
| Health check | `/api/health` | `/api/health` or `/api/v1/health` | keep existing |
| Config load/save/validate | `/api/config*` | `/api/v1/configs*` | keep existing, add schema-aware validation |
| Workflow execution | `/api/jobs`, `/api/workflows` | `/api/v1/simulation-jobs`, `/api/v1/simulation-jobs/{job_id}` | preserve async job semantics; a job may create or reference one or more `SimulationRun` resources |
| Result summary | `/api/results*` | `/api/v1/simulations/{id}/results`, `/api/v1/simulations/{id}/kpis` | map run folders to simulation resources |
| Documentation | `/api/docs`, `/api/files/text` | keep existing | no D5.1 change needed |
| City/network discovery | `/api/cities*` | `/api/v1/pilots`, `/api/v1/pilots/{id}/data-readiness` | add pilot registry layer |
| Traffic feeds | `/api/traffic-feeds*` | `/api/v1/pilots/{id}/data-sources` | keep current provider workflow slots |
| Event validation | none | `/api/v1/disruption-events/validate` | new |
| Scenario management | none | `/api/v1/scenarios` | new |
| KPI availability | none | `/api/v1/kpis/availability` | new |
| Ontology export | none | `/api/v1/ontology/export` | new |

Endpoint candidates:

- `POST /api/v1/disruption-events/validate`
- `POST /api/v1/scenarios`
- `GET /api/v1/scenarios/{scenario_id}`
- `POST /api/v1/simulation-jobs`
- `GET /api/v1/simulation-jobs/{job_id}`
- `GET /api/v1/simulations/{run_id}/status`
- `GET /api/v1/simulations/{run_id}/results`
- `GET /api/v1/simulations/{run_id}/kpis`
- `POST /api/v1/kpis/availability`
- `POST /api/v1/kpis/calculate`
- `POST /api/v1/ontology/export`
- `GET /api/v1/pilots/{pilot_id}/data-readiness`

> **v0.2 addition:** Add endpoint families or internal services for the Mini-GA/D2.2/D2.4/D2.6 concepts:

- `GET /api/v1/requirements`
- `POST /api/v1/requirement-mappings`
- `GET /api/v1/pilots/{pilot_id}/data-inventory`
- `POST /api/v1/triage-recommendations`
- `POST /api/v1/acceptability-assessments`
- `POST /api/v1/equity-impacts`
- `GET /api/v1/scenarios/{scenario_id}/audit-records`
- `POST /api/v1/stakeholder-inputs`

These do not all need to be implemented in the first prototype. D5.1 should still define whether each is `implemented`, `placeholder`, `mock`, `deferred`, or `external_dependency`.

### 8.4 Simulation adapter contract

Wrap the existing SUMO runner as the first simulation adapter.

Interface responsibilities:

- `validate_inputs(scenario, pilot_config)`: verify network, demand, event, KPI, and control inputs.
- `run(scenario, runtime_options)`: start execution and return run/job reference.
- `status(run_id)`: report queued/running/succeeded/failed/cancelled.
- `cancel(run_id)`: cancel if backend supports cancellation.
- `read_results(run_id)`: load output artifacts into typed result objects.
- `artifact_manifest(run_id)`: list files, paths, types, hashes, and provenance.
- `errors(run_id)`: expose structured failure details.

Folder-backed run outputs may remain the source of truth for the first API iteration, but they should be wrapped as `SimulationRun` resources.

### 8.5 Current artifact to ontology/KPI mapping

| Current artifact | Future object(s) | Notes |
|---|---|---|
| `metadata.json` | `SimulationRun`, `Scenario`, `PilotConfiguration`, `ProvenanceRecord` | config snapshot, seed, versions, assumptions |
| `network_metrics.csv` | `SystemState`, `KpiObservation` | speed, throughput, delay proxy, active accidents, speed ratio |
| `accident_reports.json` | `DisruptionEvent`, `ResponseAction`, `TransportElement`, `KpiObservation` | incident timing, location, severity, queue/rerouting impact |
| `antifragility_index.json` | `AntifragilityAssessment`, `KpiObservation` | mark current mode as prototype/proxy unless validated |
| `simulation_summary.json` | `SimulationRun`, `ScenarioKpiComparison` | aggregate result view where present |
| `report.html` | `ReportArtifact` / export artifact | human-readable summary, not canonical data source |
| GUI job logs | `ProvenanceRecord`, `ExecutionLog` | execution trace and errors |

## 9. Integration Contracts

### 9.1 WP3 triage

Do not invent final WP3 triage logic. Prepare placeholders:

- `TriageCategory`
- `ResponseAction`
- `TriageRecommendation`
- `AcceptabilityConstraint`
- `VulnerabilityAssessment`
- `EquityImpact`

Required future WP3 inputs:

- triage categories,
- response prioritisation rules,
- vulnerability/equity methodology,
- public acceptability constraints,
- DSS interface expectations,
- governance/justice constraints.

> **v0.2 addition:** Triage recommendations should carry two independent dimensions: `technical_priority` and `social_acceptability`. High-priority/low-acceptability actions are not discarded automatically, but must require communication, deliberation, design adjustment, or mitigation before being labelled deployable.

> **v0.2 addition:** Candidate first triage categories from D2.2/D2.4 and the Mini-GA triage reference include critical services, essential mobility needs, economic continuity, and general mobility. SUMA should be able to represent emergency vehicles, public transport users, caregivers, vulnerable groups, persons with disabilities, older/younger people, essential workers, freight/delivery, and non-essential private vehicle use as differentiated priority classes where the pilot has relevant data.

### 9.2 WP4 control and modelling

Do not hard-code WP4 methods directly into core simulation code. Use adapters:

- `ControlStrategy`
- `ControlStrategyInput`
- `ControlStrategyOutput`
- `ControlZone`
- `RetrainingRequirement`
- `ControlKpiMapping`

Minimum WP4 adapter questions:

- What format will WP5 receive: code, script, API, notebook, report, dataset, or mixed package?
- What network format is assumed?
- What demand/OD representation is required?
- What parameters are fixed, configurable, or learned?
- What output should SUMA consume?
- What requires retraining when protected/congestion zones change?
- Which D2.7 KPIs replace or supplement current efficiency/skewness indicators?

### 9.3 Ontology/knowledge-graph integration

Near-term:

- ontology-aligned internal schemas,
- stable IDs and naming,
- mapping tables,
- JSON export,
- optional JSON-LD context.

Later:

- RDF export,
- SHACL validation,
- SPARQL templates,
- triple-store integration if required by project architecture.

> **v0.2 update:** When using the abbreviation KG, define it as `knowledge graph`. In D5.1 language, prefer `full ontology/knowledge-graph integration` and treat it as staged beyond the first API specification. D5.1 should still define ontology-aligned schemas, stable identifiers, JSON export, and a JSON-LD path.

### 9.4 Pilot configuration and data readiness

Pilot configuration template:

```json
{
  "pilot_id": "ahepa_thessaloniki",
  "display_name": "AHEPA / Thessaloniki",
  "study_area": {},
  "network_refs": [],
  "data_readiness": [
    {
      "data_object": "demand",
      "status": "proxy_ready",
      "source": "synthetic_or_modelled",
      "owner": "pending"
    }
  ],
  "traffic_data_sources": [],
  "public_transport_sources": [],
  "event_catalogue_refs": [],
  "kpi_subset": [],
  "critical_assets": [],
  "critical_routes": [],
  "local_constraints": [],
  "validation_owner": "pending",
  "validation_data": [],
  "missing_inputs": [],
  "permission_constraints": [],
  "confidence": 0.0
}
```

Readiness statuses:

- `not_started`
- `conceptual`
- `data_inventory_started`
- `proxy_ready`
- `simulation_ready`
- `calibration_ready`
- `validation_ready`

Four-pilot readiness matrix for Mini-GA use:

| Pilot | Candidate use case | Data status | KPI subset | Local owner | Validation target | Unresolved dependency |
|---|---|---|---|---|---|---|
| Bratislava | roadworks, public-transport disruption, public events, Danube/underpass flooding | partial counts plus VISUM/VISSIM context | reliability, PT continuity, corridor stress, event impact | city/WP6 | planning/scenario usefulness and reliability/accessibility under stress | exact corridor/zone, external-model handover |
| Larissa | riverside flood response, Papanastasiou corridor protection, storm anticipation, flood recovery | limited counts; parking/e-bike proxies possible | flood accessibility, corridor continuity, recovery time | city/WP6 | degraded/proxy mode usefulness and flood-response support | data owner, flood extent, validation baseline |
| Odesa | power outage, infrastructure damage near Railway Station/Pryvoz, simultaneous disruptions, restoration, PT optimisation | fragmentary operational data; sensitive context | emergency access, PT continuity, restoration, timeliness | city/WP6 | critical mobility continuity under degraded data | safe data access, substituted acceptability baseline |
| AHEPA / Thessaloniki | hospital access during flood, congestion, earthquake, regional disruption | rich Smart Mobility Living Lab data if permissions align | emergency access, hospital continuity, public health | AUTH/AHEPA/WP6 | emergency access reliability | corridor definition, observed response data, permission constraints |

> **v0.2 update:** CU's Mini-GA Pilot Briefings provide more concrete current pilot framing:

| Pilot | Updated Mini-GA framing | Data posture | Priority locations / assets | SUMA implication |
|---|---|---|---|---|
| Bratislava | daily/mid-scale stressors, roadworks, public-transport perturbations, public events, flooding of underpasses | VISUM/VISSIM plus partial key-intersection counts | Patronka, radials, inner ring, government district, underpasses | good fit for planning/scenario MVP and external-model adapter discussion |
| Larissa | flood resilience and Papanastasiou corridor protection | no comprehensive traffic counts yet; parking/e-bike data; no transport-planning software | riverside areas, Kiprou, Iroon Politechniou, Papanastasiou | requires proxy/degraded-mode workflows and clear missing-input reporting |
| Odesa | conflict-shaped mobility, power outages, infrastructure damage, public-transport continuity | fragmentary field/video data; operational constraints; no D2.2 forum | Khadzhibeysky-Centre link, Railway Station, Pryvoz, principal arterials | requires sensitive-data handling, degraded/offline assumptions, and cautious acceptability baselines |
| AHEPA / Thessaloniki | hospital access under flood, congestion, earthquake, regional disruption | rich Smart Mobility Living Lab data, detectors, cameras, smart signals, FCD, VISUM context | AHEPA Hospital, Inner Ring, Egnatia, Tsimiski, vertical corridors | best candidate for high-fidelity hospital-access prototype path if local permissions align |

> **v0.2 addition:** Pilot selection for implementation should consider both data richness and demonstrator representativeness. AHEPA may be high-fidelity; Bratislava may be strong for model-integration and planning scenarios; Larissa and Odesa are important tests of degraded/proxy modes.

## 10. Mini-GA Working Pack

> **v0.2 update:** The Mini-GA is now explicitly charrette-like and organised around four pilot tables. Rhoe's Day 2 role is to convert Day 1 outputs into buildable SUMA functionality, architecture, API/service boundaries, UI roles, and backlog owners. Use the master chain: `use case -> requirement -> ontology/data object -> KPI -> function -> API/service -> architecture module -> UI role -> backlog owner`.

Rhoe operating rule for Day 2:

- Do not reopen broad use-case selection unless a Day 1 decision blocks the MVP.
- For every proposed function, require actor, input, output, KPI, owner, and pilot applicability.
- Mark every function with `module_type` (`common_core`, `local_adapter`, or `external_module`) and `implementation_status` (`implemented`, `placeholder`, `mock`, `deferred`, or `external_dependency`).
- Use `acquire`, `proxy`, `defer`, or `drop` for missing data.
- End each session with decisions, unresolved blockers, and named owners.

### 10.1 Session 5 canvas: requirement to SUMA function

Use this table during Day 2 Session 5.

| Requirement ID | Pilot/use case | SUMA function | API endpoint | Input object | Output object | Owner | Maturity | Validation evidence |
|---|---|---|---|---|---|---|---|---|
| FR-01 | example | classify | `/api/v1/disruption-events/validate` | `DisruptionEvent` | validation result | Rhoe/CU | placeholder | schema fixture |

SUMA function vocabulary:

- ingest,
- validate,
- classify,
- configure,
- simulate,
- compare,
- calculate,
- recommend,
- explain,
- alert,
- export.

> **v0.2 addition:** Expanded function vocabulary for Day 2: ingest/validate, detect/classify, model/simulate, assess, recommend/compare, explain/audit, notify/export, and learn. These function families map directly to the D2.6-derived requirements and D5.1 API endpoint families.

### 10.2 Session 6 canvas: function to architecture

Use this table during Day 2 Session 6.

| Function | Component/layer | Common core or local adaptation | Partner dependency | Unresolved decision | Fallback |
|---|---|---|---|---|---|
| simulate disruption | simulation adapter layer | common core | Rhoe, WP4 if control active | adapter input format | SUMO-only proxy mode |

> **v0.2 addition:** For Session 6, force each architecture component to state whether it is common SUMA core, pilot-local configuration, external partner module, mock/placeholder, or deferred. The agenda explicitly names the CUSP backbone, AUTH short-term DSS/ABM, ETH traffic-control module, LISER routing, DMO UI/RE Suite, and Rhoe SUMA API as modules whose connection points need to be made concrete.

### 10.3 Named module alignment table

Use this table to keep the Mini-GA discussion tied to the agenda's named partner modules and to prevent vague integration commitments.

| Module or contribution | Expected SUMA connection | Required clarification | First SUMA contract | Fallback if not ready |
|---|---|---|---|---|
| CUSP / CU | semantic project infrastructure and possible Optimize AI interface | public interface summary, NDA limits, data objects, ownership | `OntologyMapping` / external-service placeholder | mark as dependency, keep JSON export |
| ABM or short-term DSS / AUTH | candidate method or decision-support input | method maturity, input/output form, run-time expectations | `ResponseAction` or method-adapter placeholder | scenario annotation only |
| Traffic control / ETH | antifragile control, perimeter control, SUMO-compatible methods | network and OD assumptions, control-zone definitions, retraining needs | `ControlStrategy` adapter | mock WP4 adapter |
| ABR routing / LISER | routing, acceptability, behaviour or response evidence | response constraints, user groups, social acceptability variables | `RoutingStrategy` / `AcceptabilityConstraint` | response-action placeholder |
| RE Suite, UI, and DMO | role-based UI, reporting, DSS user journey | roles, permissions, outputs, export expectations | `UserRole`, `DashboardView`, report/export requirements | guided/expert UI placeholders |
| SUMA API / Rhoe | integration API, simulation orchestration, KPI availability, exports | minimum viable requirements, pilot data, validation criteria | D5.1 API/schema contract | prototype with explicit assumptions |

### 10.4 Partner ask matrix

| Partner/group | Exact ask | Why Rhoe needs it | Minimum acceptable answer | Deadline | Fallback |
|---|---|---|---|---|---|
| CU / ontology | first prototype class subset, IDs/namespaces, JSON/JSON-LD expectations, competency questions | schema and export design | list of mandatory classes and example records | Mini-GA Day 1/2 | use D2.5 top-level classes provisionally |
| CU / Optimize AI | CUSP documentation, NDA status, input/output expectations | architecture placement and integration feasibility | public interface summary if NDA blocks docs | before D5.1 drafting | mark as dependency |
| AUTH / KPI | KPI subset per pilot, units, thresholds, update frequency | KPI registry and availability design | core/proxy/deferred table | Mini-GA Day 1 | use D2.7 groups provisionally |
| ETH / WP4 | handover format, network/demand assumptions, control inputs/outputs, retraining assumptions | adapter contract | one method package outline | Mini-GA Day 2 | mock WP4 adapter |
| LISER / routing / acceptability | ABR/routing interface and social acceptability evidence | response and routing integration | input/output summary and constraints | Mini-GA Day 2 | response-action placeholder |
| DMO / UI | user roles, permissions, UI flows, reporting/export expectations | T5.3 alignment | role-flow sketches | Mini-GA Day 2 | guided/expert mode placeholder |
| WP3 / triage leads | triage categories, response-action representation, acceptability constraints | triage schema | preliminary response-action fields | after Mini-GA | placeholder interface |
| Pilot/WP6 partners | study area, critical assets/routes, data inventory, validation owner | pilot configuration | minimum viable definitions with confidence labels | Mini-GA Day 1/2 | proxy-ready status only |

### 10.5 Decision log template

| Item | Type | Description | Owner | Deadline | Impact if unresolved |
|---|---|---|---|---|---|
| D2.6 requirements availability | dependency | requirements shortlist not in repo context | CU / relevant WP2 owners | before Day 1 Session 2 | use provisional categories |

Types: `decision`, `assumption`, `dependency`, `risk`, `deferred`.

### 10.6 Data readiness line for cities

Rhoe is not asking for perfect datasets by August 2026. Rhoe needs minimum viable definitions, data owners, confidence labels, permission constraints, and validation expectations so SUMA can expose readiness honestly.

## 11. Roadmap

### 11.1 Before Mini-GA

- Prepare architecture diagram.
- Prepare current-capability versus target-capability table.
- Prepare Session 5 and Session 6 canvases.
- Prepare partner ask matrix.
- Prepare D2.6 fallback categories.
- Prepare AHEPA emergency-access example without implying it is the only first pilot.

> **v0.2 update:** Also prepare an Rhoe Day 2 facilitation pack: traceability-chain canvas, function-family list, common-core/local-adapter module canvas, data `acquire/proxy/defer/drop` decision rule, Layer 3 SUMA KPI issue list, and a decision/dependency/owner log.

### 11.2 May-June 2026: foundation

- Add event/scenario/KPI/provenance/data-quality schemas.
- Encode D2.1 controlled vocabularies.
- Create KPI definition and availability registry.
- Add pilot registry/data-readiness template.
- Add current-artifact-to-ontology/KPI mapping.
- Draft D5.1 API migration from existing `/api/*`.
- Preserve existing workflows while adding domain wrappers.

> **v0.2 addition:** Add requirement mapping, acceptability/equity assessment status, data inventory `role/data_role`, and audit/provenance fields to the D5.1 schema work. Treat the CU Mini-GA requirement, ontology, KPI, data inventory, use-case, triage, and pilot briefing references as the current working inputs to consolidate after the charrette.

### 11.3 July-August 2026: D5.1 and early implementation

- Wrap existing SUMO runner as the first `SimulationEngine` adapter.
- Add mock simulation adapter for tests.
- Add first schema-aware API endpoints.
- Add scenario translator from current YAML/resilience scenarios.
- Add calculation-mode and data-quality flags.
- Add first ontology export draft if mapping has stabilized.
- Add D5.1 documentation and OpenAPI/schema snapshots.

### 11.4 September-December 2026: pilot and partner integration

- Integrate first WP4 method in adapter mode.
- Add first WP3 triage placeholder or real adapter depending on maturity.
- Implement per-pilot configurations.
- Select first implementation pilot based on data readiness.
- Include AHEPA emergency access as a focused candidate case.
- Add KPI scenario comparison reports.
- Add competency-question tests.

## 12. Validation And Acceptance Gates

### 12.1 Development checks

New modules should pass:

- `pytest tests/ -m "not integration"`
- `ruff check .`
- `ruff format --check .`
- `mypy src/suma`
- schema fixture tests,
- mock adapter contract tests,
- generated OpenAPI snapshot or equivalent API contract review when API endpoints change.

If CI does not yet run these checks against `src/suma`, update CI before relying on it as a D5.1 quality gate.

### 12.2 Scientific validation gates

Every assessment output must state:

- input data status,
- model/calculation mode,
- required inputs,
- missing inputs,
- assumptions,
- confidence,
- validation status.

Validation tracks:

- benchmark network validation,
- synthetic disruption tests,
- pilot-specific historical comparison where data allow,
- AHEPA emergency-access validation if selected,
- sensitivity analysis for demand, incident, and control parameters.

> **v0.2 addition:** Social acceptability evidence is indicative and context-specific unless validated for the selected pilot and constituency. D2.2 forum outputs are deliberative evidence, not statistically representative population estimates; D2.4 survey percentages must carry sample-size/context notes; Odesa and AHEPA require tailored acceptability baselines before strong claims.

## 13. Risks

Scientific risk: SUMA could produce antifragility-looking numbers before methodology is calibrated.

Mitigation: calculation mode, confidence, data quality, and validation status are required fields.

Integration risk: WP3, WP4, ontology, KPI, and pilot outputs may arrive in incompatible formats.

Mitigation: adapter contracts, schema-first design, and mock interfaces.

Pilot risk: study areas, corridors, local constraints, or validation targets may remain under-defined.

Mitigation: pilot readiness matrix and decision/dependency log.

Scope risk: the grant can pull SUMA toward full digital twin, DSS, simulation engine, KPI platform, ontology layer, and UI at once.

Mitigation: staged D5.1 minimum viable API and explicit deferred items.

Usability risk: expert configurability can overwhelm non-technical users.

Mitigation: guided mode plus expert mode.

Repository drift risk: stale default configs or docs can undermine examples.

Mitigation: repository baseline preconditions before D5.1 examples.

> **v0.2 addition:** Acceptability risk: technically efficient recommendations may be socially unacceptable, inequitable, privacy-sensitive, or digitally exclusionary.

Mitigation: explicit acceptability/equity constraints, evidence-origin metadata, non-digital communication assumptions, privacy/access-control fields, and `not_assessed` status where evidence is not available.

> **v0.2 addition:** Charrette translation risk: the Mini-GA may produce rich discussion but not a buildable API/architecture backlog.

Mitigation: use the traceability chain and close every Day 2 session with decisions, blockers, owners, and retained/refined/deferred items.

## 14. Codex And Developer Instructions

When modifying this repository:

1. Read this file first.
2. Preserve the WP5 scope boundary.
3. Avoid implementing speculative final WP3 or WP4 logic.
4. Prefer modular abstractions over hard-coded workflows.
5. Keep mock implementations where real components are unavailable.
6. Add tests for new behavior.
7. Update documentation when adding public interfaces.
8. Keep pilot-specific configuration separate from core logic.
9. Make data-quality and missing-input behavior explicit.
10. Do not present theoretical or proxy calculations as validated science.
11. Preserve existing CLI/GUI workflows unless deliberately migrating them.
12. Work additively from current `suma` package structure.

## 15. Working Definition Of Early Success

A successful early SUMA implementation is not a fully validated digital twin. It is a robust, extensible, API-driven integration environment that can:

1. Represent AntifragiCity disruption events.
2. Configure baseline, disruption, response, adaptation, and stress-test scenarios.
3. Run simulations through mock or SUMO adapters.
4. Compute or report KPI availability.
5. Compare baseline, disrupted, and response scenarios.
6. Preserve traceability, assumptions, provenance, confidence, and data quality.
7. Export ontology-aligned records.
8. Incorporate WP3, WP4, and pilot inputs as they become available.

## Appendix A. D5.1 Recommended Structure

Recommended D5.1 sections:

1. SUMA role and scope boundary.
2. Current implementation baseline.
3. System architecture.
4. Core domain model.
5. API migration and endpoint contracts.
6. Simulation engine and adapter contracts.
7. KPI and antifragility calculation modes.
8. Ontology mapping and export strategy.
9. WP3 triage interface placeholder.
10. WP4 control strategy adapter interface.
11. Pilot configuration and data-readiness model.
12. Provenance, data-quality, and validation model.
13. Implementation roadmap.
14. Open decisions and partner dependencies.

## Appendix B. Schema Examples

### B.1 Simulation run

```json
{
  "run_id": "run_001",
  "scenario_id": "scenario_001",
  "status": "succeeded",
  "created_at": "2026-05-19T12:00:00Z",
  "started_at": "2026-05-19T12:01:00Z",
  "finished_at": "2026-05-19T12:30:00Z",
  "engine": "sumo",
  "engine_version": "unknown",
  "input_hash": "sha256:...",
  "results_uri": "results/run_001",
  "logs_uri": "results/run_001/gui_job.log",
  "calculation_mode": "prototype",
  "errors": []
}
```

### B.2 KPI availability result

```json
{
  "pilot_id": "ahepa_thessaloniki",
  "scenario_id": "scenario_001",
  "kpis": [
    {
      "kpi_id": "emergency_access_time",
      "status": "proxy",
      "required_inputs": ["network", "emergency_route", "observed_response_time"],
      "missing_inputs": ["observed_response_time"],
      "proxy_method": "simulated_travel_time",
      "confidence": 0.45
    }
  ]
}
```

## Appendix C. Terminology

- Use `Odesa` for consistency with the Mini-GA agenda, unless quoting a source that uses another spelling.
- Use `Rhoe` in ASCII filenames and code. Use `Rhoe` or official branding consistently in text where needed.
- Use `Antifragility Index` instead of ambiguous `AI` unless the abbreviation is defined locally.
- Use `WP6 / pilot demonstration inputs`, not `WP6 / WP11`, unless referring explicitly to the Grant Agreement split.
