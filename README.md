# Quant Insight

Describe a business decision in plain words. The engine works out what kind of problem it is, finds the matching method guide, solves it if you give numbers, and checks the answer a second way before showing it.

## What is new in Quant Insight
| Change | What it does | Where |
|---|---|---|
| Detailed answers | Every verified solve now also returns `detail` (plain-language sections: why this answer, how firm it is, closest calls, what would change it, before you act) and `analysis` (the numbers behind them). Send `"detail": false` to skip it. | `app/insight.py`, `POST /api/v1/solve` |
| Stability test | Your estimates (values, costs, durations, returns...) are nudged by up to 10% in 40 seeded re-solves. You get how often the same decision returns, which choices never move, which are toss-ups, the likely range of the result, and what acting on the original choice would have cost. | `insight._stability` |
| Closest calls | Knapsack and multi-resource knapsack (up to 40 items): the exact cost of reversing each choice, found by re-solving, checked against brute force in the tests. | `insight._closest_calls` |
| What-if table | One change at a time (a bigger budget, one more machine, one more site, a different target return...) with the new result and which choices move. | `insight._levers` |
| Large knapsacks | Above 8 million table cells the exact table method gave way to a greedy answer with a half-optimum guarantee. It now uses the mixed-integer solver (10 s limit), keeps the better answer and states the proven gap. A 3,000-item test instance is solved to proven optimality in about 1.3 s. The table method also no longer allocates up to 480 MB. | `pipeline.s_knapsack` |
| Faster what-ifs | Re-solves skip the second-method cross-checks (the headline answer always gets them) and run on four threads from one seed, so results do not depend on timing. A 2.5 s wall-clock cap applies and anything unfinished is listed under `skipped`. | `pipeline._fast_solve` |

Measured on the illustrative instances (10 options, no QAOA step): the whole analysis takes 10 to 1,300 ms depending on the family. Environment variables keep their `DE_` prefix so existing set-ups keep working.

What was tried and not changed: seven classifier set-ups (regularisation 3 to 100, balanced class weights, wider word n-grams, calibrated linear SVM) on 5-fold cross-validation over the 216 labelled problems. None beat the current model by more than the noise (about 5 points on 117 held-out problems), so the classifier is unchanged and the 90% target in the PRD is still not met. The limit is the amount of labelled data, not the model type; ingesting more reviewed books and open data (see below) is the lever.

Limits of the new analysis: the nudge is independent, uniform and 10% by default, so it shows sensitivity, not your real uncertainty. For scheduling, many splits tie for the best result, so only the range of the result is reported. For budget splits and portfolios, "same decision" means no share moved by more than 5 percentage points.

## Start in three steps
1. Install Python 3.10 or newer from python.org (tick "Add Python to PATH" on Windows).
2. Double-click `start.bat` (Windows) or `start.command` (Mac), or run `python start.py` in a terminal. The first start installs what it needs and may take a minute.
3. Your browser opens. Press **Start here**, or try one of the ready-made examples on **Find and solve**.

Optional: to search the 1,000,000-problem catalogue, put `business_problems_1M.csv` and `business_problem_solutions_1M.csv` in the `catalogue` folder before starting. The app builds its index once (about 20 seconds). Everything else works without them.

## What each page is for
| Page | Use it to |
|---|---|
| Start here | Pick a path in plain language; read how to interpret an answer and a glossary |
| Find and solve | Name the problem type, get the method guide, and solve your numbers with a second-method check |
| Describe a situation | Get a recommendation, assumptions and a confidence level from a few sentences |
| Double-check a choice | List options with costs and values; several methods solve it and report whether they agree |
| Trend planner | See forecast trends and split a budget across them without waste |
| Past accuracy | See how the forecaster performed on earlier weeks |
| Library, Models, Privacy and data | Manage problem files, see model scores, and control your data and consent |

## Look and accessibility
Beige background with black text. Black on the beige is 16.5:1 contrast and every other text pair is at least 7.2:1 (WCAG AAA). Large type (18 px), 44 px or larger controls, visible keyboard focus, skip link, labelled forms, tables with headers, and reduced-motion support. I could not take screenshots in the build environment, so the look is checked by DOM tests and contrast calculation, not by eye. Please look at it once in your browser.

## Manual start (developers)
    pip install -r requirements.txt
    python -m app.bigdata build business_problems_1M.csv business_problem_solutions_1M.csv   # optional, about 20 s
    uvicorn app.main:app --port 8000

## What was combined
| Part | Where |
|---|---|
| 1M-problem catalogue (full-text index, lookup by id), 108 playbooks, 44 theorems, 156 references | `app/bigdata.py`, `data/knowledge/` |
| Nine reference solvers (your `solvers.py`, unchanged) | `app/ref_solvers.py` |
| Classify, retrieve, formulate, solve, verify in one call | `app/pipeline.py`, `POST /api/v1/solve` |
| QAOA simulator as a third check on small knapsacks | `pipeline.quantum_check` using `app/optimizer.py` |
| Everything from the previous build (trend forecaster, feeds, tier/channel planner, evaluation harness, consent, appeals, ask ladder) | unchanged |

## How one solve works
`POST /api/v1/solve {"text", "catalogue_id", "family", "instance", "use_demo"}`.
1. In parallel: the trained classifier names the family, the catalogue returns the nearest problems, and the text is parsed for options and a budget.
2. Family is chosen in this order: supplied, shape of the supplied numbers, catalogue row, options read from text, classifier.
3. The matching playbook (family, size bucket, sensitivity) is fetched with its theorems and references written out.
4. With numbers: the reference solver runs, a second independent method re-derives it (MIP, exhaustive search, random restarts, depending on family), and a verifier recomputes feasibility and the objective from the raw instance. A failed check means the answer is not shown as verified. Small integer knapsacks also run on the QAOA simulator.
5. Without numbers: you get the formulation, the playbook and the data to collect. `use_demo` solves a generated instance, labelled illustrative everywhere. The catalogue solutions file has no numeric answers (every row says needs_instance_data), so no number comes from it.

Instance formats: `GET /api/v1/solve/formats` returns one worked example per family. Server limits (for example 5,000 knapsack items, 120 items for the MIP-based multi-resource knapsack, 400 stops) are stated in the error when exceeded.

Other endpoints: `GET /catalogue/stats`, `/catalogue/search?q=`, `/catalogue/{id}`, `/knowledge/playbooks|theorems|references`.

## Measured, and what it changed
Problem-family accuracy on the 45 authored statements held out of all training (`python tests/measure_family.py`): trained classifier 88.9% (95% interval 76.5-95.2), top three 100%, catalogue retrieval alone 46.7%, and an equal-weight blend 66.7%. The blend was my first design; it was worse than the classifier, so the classifier decides and retrieval is shown as supporting context. That choice was made after seeing this result on this set, so treat 88.9% as a development figure. The catalogue labels come from templated rows, which is likely why they transfer poorly to short free text.

## Verification
    pytest -q tests          # 297 tests: 258 before the Quant Insight changes, 39 new (tests/test_insight.py)
    sh tests/ui/run.sh       # browser-level checks (needs node and `npm install jsdom` in tests/ui)
New tests compare each solver with brute force on random instances, confirm that bad input gets a plain-language 422, and confirm a deliberately broken solver is caught by the verifier. 540 generated instances (60 seeds across 9 families) all verified.

## Limits
- Routing and large set-cover or portfolio-with-cardinality instances use heuristics, and say so in the guarantee field. Routing is proven optimal only for small single-route cases.
- The quantum step is a software simulator limited to 16 qubits (items plus budget bits), and only for integer knapsacks. No advantage is claimed. No real hardware was used.
- The solvers cover the nine single-objective families. Stochastic, multi-objective and time-window variants named in the catalogue are not solved; the playbook describes the method only.
- Solves run in-process with a 30 s budget; an over-long solve is reported but its thread is not killed. Not load tested.
- The previous build's open gaps remain (see docs/PRD_COMPLIANCE.md): neural estimators, language detection, memory and debate, PDF reports, business-profile page.

---
# Earlier README follows


Run (one command after installing):

    pip install -r requirements.txt && uvicorn app.main:app --port 8000

Open http://localhost:8000 . API docs (OpenAPI): http://localhost:8000/docs

Endpoints under /api/v1. Public: health, terms, auth/register, auth/login. Signed in, any consent state: consent (GET, POST, revoke), account/export, account/delete, appeals. Signed in with current consent (everything else): ask, reason, optimize, optimize/presets, forecast, feeds, schedules, problems/classify, library, documents, models, reports.json, reports.csv.

Tier 4 is a built-in reasoner (app/reasoner.py): playbooks for sales decline, cash and stock, supply disruption, capacity and backlog, prioritisation, marketing spend, churn and cost cuts. It needs no outside service or key. Optionally set OLLAMA_MODEL to use a model running locally through Ollama instead.
Scheduled re-runs execute in a background task inside the server process. Sessions last 7 days.
Data is stored in data/app.db (SQLite); set DE_DB to use another path. Uploaded documents are private to this server and never used for training.

## Training and tests
    python -m app.train      # trains the problem-type model, writes models/ and prints measured accuracy
    pip install -r requirements-dev.txt && pytest -q tests
    sh tests/ui/run.sh      # optional: drives the real page in jsdom against a live server (needs node and `npm install jsdom` in tests/ui)

The model is a TF-IDF plus logistic-regression classifier, retrained from the dataset and from corrections that users give on answers (only users who opted in). Folds hold out whole problems. Retrain from the Models tab or POST /api/v1/models/retrain.

## Trend forecast and data feeds
    python -m app.forecast            # top 10 on simulated data
    python -m app.forecast backtest   # rolling-origin backtest vs growth, volume and random baselines
POST /api/v1/forecast {"horizon": 7|14|30, "feed_id": null, "threshold": 5}. Without `feed_id` it runs on the built-in simulator. With a feed it learns from that feed's own history. `threshold` is what counts as a break-out (multiple of the 30-day baseline, 1.2 to 10). Real feeds rarely reach 5x, so lower it (1.5 to 3) if the response says there are too few examples to learn from; the server refuses rather than train on fewer than 20 break-out and 20 non-break-out examples. The response includes `candidates` that can be sent to /api/v1/optimize as `trends`.

Feeds (POST /api/v1/feeds/csv | /url | /wikipedia, GET /feeds, POST /feeds/{id}/refresh, DELETE /feeds/{id}; at most 10 per account):
- **CSV upload**: wide (a `date` column, then one column per trend) or long (`date`, `trend`, `value`).
- **URL**: a CSV or JSON file at an HTTPS address you control. Private and local addresses, redirects, files over 20 MB and non-HTTPS addresses are refused. Optional automatic refresh hourly or daily by the server's background task.
- **Wikipedia**: daily page views for 8 to 100 article titles from the public Wikimedia REST API (no key). Set DE_CONTACT in the environment to put a contact address in the request's User-Agent, as Wikimedia asks.
A feed needs at least 8 trends and 150 days. Gaps of up to 3 days are carried forward; more than 10 percent missing is refused. Values below 1 are raised to 1 so growth ratios stay defined. Every forecast reports the feed's latest day and warns when it is more than 7 days old.

Limits to know: forecasts learned from your own feed have not been backtested; the Wikipedia connector was tested against a recorded API shape, not the live service; and categories on forecast candidates are placeholders (assigned by rank), so overlap penalties between feed-derived trends are not meaningful until you set real categories.

## Trend planner: tiers, channels and solver backends
POST /api/v1/optimize accepts `tiers` and `channels` (GET /optimize/presets returns examples). Each trend is funded at one tier through one channel, or not at all. A tier scales cost and value; a channel scales cost and reach, and two trends in the same channel overlap by `lam_channel`. Option costs are rounded to whole units when either is used. With neither, the model is the original one-option-per-trend model and returns the same numbers as before.

`backend` chooses the solver whose answer is reported as `quantum`; `compare_backends` runs the simulator and annealer side by side.
- `simulator`: statevector QAOA, p=2, CVaR, at most 16 qubits (options plus budget bits). Larger requests are skipped with a message, not failed.
- `anneal`: software simulated annealing. Classical; it is the stand-in behind the same interface a quantum annealer would use.
- `hardware`: sends the QUBO to an external solver service. Nothing is faked when none is configured. The operator sets QUBO_SOLVER_URL, QUBO_SOLVER_COST_PER_RUN (required: the stated cost) and optionally QUBO_SOLVER_TOKEN and QUBO_SOLVER_TIMEOUT. The request must set `hardware_confirm` true and a `hardware_max_cost` at or above the stated cost. Scheduled runs cannot use hardware. The service contract is `POST {"num_vars", "offset", "qubo": [[i, j, value], ...], "num_reads"}` returning `{"samples": [{"bits": "0101..."}], "backend": "name"}`; variables are the options, then the budget slack bits. A vendor SDK (D-Wave, IBM, Braket) needs a small proxy service that speaks this contract. This adapter was tested against a local stand-in service only, never against real hardware.

Whatever the backend, samples are decoded, repaired and re-verified by the classical checker, and compared with an exact branch-and-bound reference (a proven optimum unless the node cap of 2,000,000 is hit, which is then stated). Measured on software; no quantum advantage is claimed.

## Terms, consent and account rights
The server blocks every feature route until the signed-in account has accepted the current Terms and Privacy versions (403 with code `terms_required`). Registration needs an explicit `accept_terms` and `age_confirmed` plus the versions the person was shown (a stale version gets 409). Changing TERMS_VERSION or PRIVACY_VERSION in app/main.py forces everyone to accept again. `POST /consent/revoke` withdraws acceptance and the shared-learning opt-in.
Available without current consent: `GET /account/export` (everything stored about the account as JSON), `POST /account/delete` (password plus the exact phrase DELETE MY ACCOUNT; removes the account and all linked rows, and retrains the problem-type model if the person had contributed corrections), and appeals.
Appeals (`POST /appeals`, `GET /appeals`): kinds answer, refusal, account, other. Refused requests are logged as a run (the refusal text, not the request) so they can be appealed. Reviewers are the emails in DE_ADMINS (comma separated); they use `GET /admin/appeals` and `POST /admin/appeals/{id}/decide` with upheld, rejected or needs_info, and cannot decide their own appeal. A decision is a record and a message to the person; it does not rewrite the stored answer.
Not covered: rows a person added to the shared problem library are not linked to their account and are not removed; backups made by whoever runs the server are outside deletion; there is no reviewer interface beyond the API.

## Evaluating the reasoning engine
    python -m app.evalharness make-dev                 # regenerate the public development suite (data/eval/dev_cases.jsonl)
    python -m app.evalharness make-private --seed N    # operator only: a larger held-out suite in data/eval/private/ (gitignored)
    python -m app.evalharness run --suite dev --out docs/EVAL_REPORT.md
    python -m app.evalharness collect --server URL --token T --system engine --out engine.jsonl   # also framework_only, analogy_only
    python -m app.evalharness blind-pack --responses engine.jsonl framework.jsonl --raters 2 --out packet.csv --key key.json
    python -m app.evalharness blind-score --ratings ratings.csv --key key.json
The harness scores the budgeted-selection reasoner against ground truth from its own dynamic program, against four baselines with paired bootstrap intervals, and checks that the stated numbers match a recomputation, that bad input is rejected, that no option is silently dropped, how reliable each confidence label is, and whether the stability figure overclaims against fresh perturbations. It also builds blinded A/B packets for human raters and scores them (win rate with interval, sign test, agreement). A claims gate prints which public statements the evidence supports; at present every superiority claim is unsupported. Scope: only the budgeted-selection form has ground truth; free-text scenarios can only be judged by blinded human preference, and no such study has been run. Do not tune against the private suite. Current results: docs/EVAL_REPORT.md.
## Ingesting books and open data
    python -m app.ingest build ./my_sources      # my_sources/ holds the files and a sources.csv manifest
    python -m app.ingest merge data/review_queue.csv   # after a person fills reviewed_family for rows they approve
    python -m app.labels apply && python -m app.train

sources.csv columns: file,title,license,allow_training,source_url. Every readable file is indexed for retrieval with its title and licence in each citation. Only files marked allow_training=yes with a licence that permits it (CC0, public domain, CC BY, CC BY-SA, MIT, Apache, BSD) produce candidate training problems, and no candidate becomes training data until a person labels it in the queue.

## Model accuracy (5-fold, whole problems held out)
Exact figures, both label sets, intervals and the operating point are in models/card.json and docs/PRD_COMPLIANCE.md. Labels were reviewed against docs/LABEL_RUBRIC.md; originals are kept in the formulation_family_original column. Broad kind: 77.8% with the labels as supplied, 80.3% with reviewed labels. The engine states a problem kind only when its confidence reaches the operating point (0.8 on the current model: 91.7% right on 61.5% of problems, 95% interval 83-96%). The PRD target of 90% on all held-out problems is not yet met. See docs/PRD_COMPLIANCE.md for what remains.
