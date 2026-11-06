# PRD compliance matrix

Status key: **Done** = implemented and read in code; **Partial** = some of the requirement; **Missing** = not built.
"Tests pass" means covered by the automated suite, which was run in the build sandbox: 121 tests under `tests/` plus 21 browser-level checks in `tests/ui/` (the real page driven in jsdom against a live server). Run `pytest -q tests` and `sh tests/ui/run.sh` to confirm. Things the sandbox could not reach (live Wikimedia, any quantum or annealing hardware) are marked "not run live".

## Trend Portfolio Optimizer
| Requirement | Status | Evidence and gap |
|---|---|---|
| 6A Trend forecasting engine: ranked top 10 with break-out probability, intervals, lead time; connectors; simulator; backtest | **Partial** | `app/forecast.py`, `app/feeds.py`, `POST /api/v1/forecast`: simulator, past-only features, boosted-tree ensemble with isotonic calibration, 10-90 percent ensemble intervals, top 10 with drivers, rolling-origin backtest against growth, volume and random baselines, hand-off to the optimizer. **New:** feed connectors for CSV upload (wide or long), HTTPS URL (SSRF-guarded, scheduled refresh) and Wikipedia page views; alignment, gap handling and freshness warnings; configurable break-out threshold; refusal to train on too few break-out examples. Tests pass, with the Wikipedia connector tested against a recorded response shape, **not run live**. Missing: further connectors (search trends, social platforms), trend entity detection from text, expected peak date and half-life per trend, conformal intervals, real categories for feed trends (candidates get placeholder categories by rank, so overlap penalties on feed data are not meaningful). Forecasts from user feeds have no backtest; the backtest below is simulated data only. Mean lead time on the simulator is 1.2 to 1.4 days against the 3-day target. |
| 6B QUBO + QAOA with classical reference | **Partial** | **New:** tier and channel variables (one option per trend, tier scales cost and value, channel scales cost and reach, same-channel overlap); an explicit QUBO builder checked against the vectorised energy; three backends behind one interface: statevector QAOA (up to 16 qubits, skipped with a message beyond that), software simulated annealing, and an external-solver adapter with a stated cost, a confirmation and a spending cap required, no scheduled use, and nothing faked when unconfigured. Exact reference is now branch and bound with a linear-relaxation bound: proven optimal unless a 2,000,000-node cap is hit (then stated), checked against brute force on 40 randomised instances. Every backend result is repaired and re-verified classically. Single-option requests return exactly the numbers of the earlier engine (golden test). The hardware adapter was tested against a local stand-in service, **not run on any real hardware or vendor**; a vendor SDK needs a small proxy speaking the documented contract. Missing: situation encoder and business profile, MIP/CP-SAT reference, hybrid decomposition for larger problems. Measured: the annealer reached the exact optimum on some tier-and-channel instances and fell short on others (ratios about 0.96 to 0.98 on a 10-trend, 9-option example), so it should be read as a heuristic. |
| FR-14, FR-15 Every result verified, top 3 portfolios | Done (tests pass) | `ok(m)` check on each portfolio; three distinct portfolios. |
| 8 Neural estimators (value, decay, overlap, graph proposer) | **Missing** | Value, decay and overlap are fixed formulas and a category rule. |
| 8 Problem classifier at least 90 percent on held-out problems | **Partial, target not met** | See "Classifier accuracy" below. |
| FR-23/24 Problem lab and formulation families | Partial | Classifier names the family and a method sentence; no formulation proposal and solve for families other than the budgeted selection. |
| FR-25 PDF and CSV reports | Partial | CSV and JSON only. |
| Pages: dashboard, business profile, forecast, what-if, backtest, models and quantum jobs | Not verified | Read from `static/index.html` only in part; business profile, backtest and forecast pages depend on the missing engines. |

## Scenario Reasoning Engine
| Requirement | Status | Evidence and gap |
|---|---|---|
| Never-blank six-tier fallback ladder, tier shown | Done (tests pass) | `ask()` walks tiers 1,2,4,3,5 then 6; tests cover blank, gibberish, injection and chaos cases. |
| Universal scenario representation and gap handler | Partial | Parser reads a budget and lines like "- Name: cost X value Y". Goal, actors, uncertainties, horizon are not extracted. |
| Solver pool | Partial | Exact knapsack, relaxation, analogical, built-in playbooks. No simulation, forecasting or decision-tree solvers. |
| Retrieval with citations, no invented citations | Done (tests pass) | Private documents (TF-IDF). New: shared open-licence corpus with title and licence in each citation (`app/corpus.py`, tested offline). |
| FR-4 contradictions, FR-8 edit assumptions and re-run, FR-9 follow-ups, FR-17 case library | **Missing** | |
| FR-19 health page | Partial | `/api/v1/health` with tier counts; no fallback causes, latency or error rates. |
| Latency targets | Not measured | |

## Advanced Reasoning Engine
| Requirement | Status | Evidence and gap |
|---|---|---|
| Adaptive depth, escalation when methods disagree, code-checked arithmetic | Partial (tests pass) | `/reason`: exhaustive search, DP cross-check, heuristic critic, 300-run stability test, escalation. Only for the knapsack form. The harness found and fixed two silent failures, below. The "medium" confidence label at Deep depth reflects sensitivity of the choice to value uncertainty, not likelihood of error: all 46 such answers on the development suite were exactly optimal. |
| Evaluation harness, baselines, private test sets, blinded expert preference, calibration vs baselines | **Partial** | `app/evalharness.py`, report in `docs/EVAL_REPORT.md`. Built: independent ground truth, 150 development cases in nine families, four baselines with paired bootstrap intervals, recomputation of stated numbers, malformed-input and silent-input-loss checks, reliability by confidence label, stability calibration against fresh perturbations, private-suite generator (gitignored, not shipped), blinded A/B packet builder and scorer (win rate, sign test, agreement), and a claims gate. The harness is itself tested, including that it fails on a deliberately broken engine. Not done: no private suite has been generated or run; no blinded study with real raters has been run; no external systems are compared; ground truth exists only for the budgeted-selection form. On the development suite the engine reached the optimum on 140 of 140 cases (95% CI 97.3 to 100) against 60.7% for the best baseline; this suite is public, so it proves little about unseen cases. **Every public superiority claim remains unsupported**, as the gate reports. |
| Memory, debate, process reward model, RL training | **Missing** | Out of scope for a local app without the training infrastructure the PRD describes. |

## Terms, Conduct, and Response Etiquette
| Requirement | Status | Evidence and gap |
|---|---|---|
| Nine policy documents reachable in the UI | Done (UI read) | `DOCS` list in `static/index.html`. Text is summary copy; counsel must supply legal text. |
| FR-1..3 Explicit unticked acceptance, stored version, forced re-consent | **Done (tests pass)** | Registration refuses without explicit `accept_terms` and `age_confirmed` and the versions shown; acceptance is stored server-side with versions and time; every feature route returns 403 `terms_required` until the current versions are accepted; changing the version constants forces re-acceptance; withdrawal is available; the UI shows an unticked box at sign-up and a re-consent view. Full legal text is still summary copy and needs counsel. |
| Notices: automated system, decision support, high-stakes, fairness, injection | Done (tests pass) | Added to every `/ask` response. |
| Crisis, minors, unlawful-request refusals | Partial | Regex-based; English only. |
| FR-7 Response checker rejecting untraceable numbers and invented citations | **Partial** | Only replaces the words guaranteed, risk-free, certain. No number tracing. |
| FR-11 Language detection and right-to-left layout | **Missing** | |
| FR-17..19 Export and deletion of all data, report control, appeals, public changelog | **Partial** | **New:** account export (JSON), account deletion (password and typed phrase; removes all linked rows; retrains the model without the person's corrections), appeals for answers, refusals, account decisions and other (refusals are now logged so they can be appealed; reviewers via DE_ADMINS, no self-review). All three stay available without current consent. Caveats: library rows a person added are not linked to them and stay; operator backups are outside deletion; the reviewer side is API-only; there is no service-level promise or timer on appeals. Missing: report control (flagging an answer for abuse or error as a distinct flow) and a public changelog (the UI shows one static entry). |
| No emoji, no placeholder text | Done (UI scanned) | Zero emoji found in `static/index.html`. |

## Classifier accuracy (the only 90 percent target in the PRDs)
Cross-validation holds out whole real problems (117). Authored rows (99, project-written, `data/authored_problems.py`) are used for training only and never scored there.

| Measure | Labels as supplied | Reviewed labels (docs/LABEL_RUBRIC.md) |
|---|---|---|
| Nine-way exact type | 65.0% | 59.8% |
| Right type in top three | not recorded | 86.3% |
| Broad kind (six kinds) | 79.5% | 80.3% |
| Broad kind, answered only at confidence 0.8 or higher | see models/card.json | 90.8% on 65.0% of problems (95% CI 82.2-95.5) |
| Calibration error (ECE) | | 0.162 (PRD target under 0.1: not met) |

Fresh test: 45 authored statements held out of training. Before adding the authored training rows the model scored 88.9% broad kind and 77.8% nine-way; after, 95.6% broad kind (95% CI 85.2-98.8) and 88.9% nine-way. These statements are cleaner than real requests and were written by the same author as the rubric, so treat them as a development check.

Reading: about 90% is reached only on the subset the engine is confident about; the interval still includes values below 90%, and calibration misses its target. A defensible claim needs real problems from independent sources, a second labeller, and a fresh held-out score.

## Trend forecast backtest (simulated data)
Rolling origin, cutoffs at days 150, 180 and 210, break-out means 5x the 30-day baseline.

| Horizon | Model precision at 10 | Growth baseline | Volume baseline | Random | Model NDCG at 10 |
|---|---|---|---|---|---|
| 7 days | 0.38 | 0.23 | 0.05 | 0.01 | 0.75 |
| 14 days | 0.39 | 0.24 | 0.05 | 0.01 | 0.59 |
| 30 days | 0.37 | 0.25 | 0.08 | 0.02 | 0.46 |

Only 13 to 15 forecast dates per horizon, so the margins are indicative. Calibration error is 0.003 to 0.004 and Brier beats the base rate. Simulated data cannot show real-world accuracy.

## Defects the evaluation harness found, and what was done
Found on its first run against the existing engine, then fixed and covered by permanent tests:
1. **Option names containing a colon or pipe were silently dropped.** The engine answered at "high" confidence on the remaining options. Before the fix: optimal on 88.6% of the development suite with a worst gap of 100%, 58 options dropped across 18 cases. The reading error also made the stability figure overclaim in 14 of 108 Deep cases.
2. **Numbers with thousands separators were misread.** "Budget of $1,200" became a budget of 1 and "cost 1,100" removed the option. Formatted numbers were read correctly in 0 of 10 cases before the fix.
3. Unreadable option lines are now reported (`warnings` on `/reason`, a notice on `/ask`) and lower confidence to provisional instead of being ignored.
After the fix: 140 of 140 optimal, 0 options dropped, no stability overclaim. The fixes were made while looking at the public development suite, so they are not an independent confirmation; a private suite is the proper check.

Also found and fixed along the way: the analogy tier printed "Treat this as a knapsack problem: , aiming to ." for the 99 catalogue rows without a decision type or objective; a negative budget crashed the optimizer instead of reporting no feasible portfolio; tests shared a persistent database and failed on a second run (each run now uses its own temporary database); and two UI faults in new code found by the browser-level test (a re-render loop while consent was pending, and a wrong field name in the forecast table).

## Known limits of this release
- The feed categories are placeholders (see 6A). Set real categories before trusting overlap penalties on feed data.
- The problem-type classifier is unchanged and still below its 90 percent target (see below).
- Concurrency: SQLite with one process. Not load tested.
- The quantum simulator is limited to 16 qubits, so tier-and-channel requests with more than about 12 options (after budget bits) skip it.


## Combined build additions
| Requirement | Status | Evidence and gap |
|---|---|---|
| Trend PRD FR-23/24: problem lab proposes a formulation, then solves it, across the nine families | **Done for single-objective forms (tests pass)** | `POST /api/v1/solve`; each family has a reference solver, an independent second method and a verifier; 540 generated instances verified; brute-force agreement tests. Variants with stochastic data, multiple objectives or time windows are formulated but not solved. |
| Scenario PRD: solver pool, router, analogical tier | Partial, improved | Router is the trained classifier plus catalogue retrieval; analogical tier now draws on 1M catalogue problems and 108 playbooks. Simulation, forecasting and decision-tree solvers are still absent from the pool. |
| Scenario PRD 10.2: seed catalogue of at least 10,000 scenarios in 30 domains | **Met in size** | 1,000,000 rows, 15 domains (not 30), indexed. Rows are template-generated, so volume overstates diversity. |
| Advanced PRD: verification suite (numerical, constraint, formal, independent re-derivation) | Partial | Every solve recomputes numbers from the raw instance, checks every constraint, and re-derives the answer a second way. No simulation or adversarial verifier. |
| Optimizer PRD 6B: classical reference on every job; QAOA | Partial | Small integer knapsacks are cross-checked on the QAOA simulator. MIP/CP-SAT reference now exists for the nine families (SciPy HiGHS). Hybrid decomposition and business-profile encoder still missing. |
| Problem classifier at least 90 percent | **Not met** | 88.9% on 45 held-out authored statements (interval 76.5-95.2); catalogue retrieval 46.7%; see README. |
| Conduct PRD notices on solve responses | Done (tests pass) | Automated-system, high-stakes, fairness, injection and illustrative-data notices; refusals are logged. |
