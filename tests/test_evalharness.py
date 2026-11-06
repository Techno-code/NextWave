"""The harness is itself tested: ground truth against brute force, statistics against known values, blinding, and above all that it can fail."""
import csv, itertools, json, random
from pathlib import Path
import numpy as np, pytest
from fastapi.testclient import TestClient
from app import evalharness as E, solvers, main

def brute(options, B):
    return max(sum(o["v"] for o, k in zip(options, m) if k) for m in itertools.product([0, 1], repeat=len(options)) if sum(o["c"] for o, k in zip(options, m) if k) <= B + 1e-9)

def test_truth_matches_brute_force():
    R = random.Random(5)
    for _ in range(60):
        o = [{"n": f"o{i}", "c": R.choice([1, 2, 2.5, 3, 4.25, 7]), "v": R.randint(1, 30)} for i in range(R.randint(1, 9))]; B = R.choice([0, 1, 4, 6.5, 10, 20])
        assert abs(E.truth(o, B) - brute(o, B)) < 1e-9

def test_dev_cases_are_reproducible_and_the_shipped_file_is_current():
    assert E.make_cases(1000) == E.read_cases(E.EVAL / "dev_cases.jsonl")
    fam = {c["family"] for c in E.make_cases(1000)}; assert {"random", "correlated", "ties", "fractional_costs", "tight_budget", "large", "awkward_names", "formatted_numbers", "malformed"} <= fam

def test_every_case_text_states_exactly_its_options():
    for c in E.make_cases(7, .3):
        if c["expect"] == "answer": assert [o["n"] for o in c["options"]] and E.truth(c["options"], c["budget"]) == c["optimum"]

def test_statistics_against_known_values():
    assert E.wilson(0, 0) is None and E.wilson(50, 100) == [0.4038, 0.5962] and E.wilson(100, 100)[1] == 1.0
    assert abs(E.sign_test_p(9, 1) - 0.021484375) < 1e-12 and E.sign_test_p(5, 5) == 1.0 and E.sign_test_p(0, 0) == 1.0
    assert E.paired_bootstrap([1.0] * 20, seed=1) == [1.0, 1.0] and E.paired_bootstrap([], seed=1) is None and E.paired_bootstrap(list(range(30)), seed=3) == E.paired_bootstrap(list(range(30)), seed=3)
    assert E.kappa(list("AABB"), list("AABB")) == 1.0 and E.kappa(list("AB" * 10), list("AAAA" + "BB" * 8)) < .5

def test_engine_is_exact_on_the_suite_and_baselines_are_not():
    r = E.evaluate(E.make_cases(31, .2), fresh_draws=40)
    assert r["systems"]["engine_standard"]["optimal_rate"] == 1.0 and r["systems"]["engine_standard"]["infeasible"] == 0 and r["systems"]["engine_standard"]["claim_matches_recomputation"].split("/")[0] == r["systems"]["engine_standard"]["claim_matches_recomputation"].split("/")[1]
    assert r["systems"]["greedy_value_per_cost"]["optimal_rate"] < 1.0 and r["systems"]["random_feasible"]["mean_gap_pct"] > r["systems"]["greedy_value_per_cost"]["mean_gap_pct"]
    assert r["malformed_input"]["rejected"] == r["malformed_input"]["cases"] and r["parse_fidelity"]["options_dropped_silently"] == 0

def test_harness_catches_a_broken_engine(monkeypatch):
    """A harness that cannot fail proves nothing: swap in a solver that returns the greedy answer while claiming it is exact."""
    real = solvers.reason_knapsack
    def broken(text, depth="Standard"):
        r = real(text, depth); p = solvers.parse(text); m, v = solvers.greedy(p["opts"], p["budget"]); c = solvers.tot(p["opts"], m, "c")
        return {**r, "chosen_mask": m, "answer": f"Choose {solvers.names(p['opts'], m)}. Cost {c:g} of {p['budget']:g}; value {v:g}."}
    monkeypatch.setattr(solvers, "reason_knapsack", broken); r = E.evaluate(E.make_cases(31, .2), fresh_draws=20)
    assert r["systems"]["engine_standard"]["optimal_rate"] < .9
    assert E.claims(r, "private")["more accurate than simple baselines on unseen cases"].startswith("not supported")
    def lying(text, depth="Standard"):
        r = real(text, depth); return {**r, "answer": "Choose everything. Cost 1 of 1; value 999."}
    monkeypatch.setattr(solvers, "reason_knapsack", lying); assert E.evaluate(E.make_cases(31, .15), fresh_draws=10)["systems"]["engine_standard"]["claim_matches_recomputation"].startswith("0/")

def test_harness_catches_overspending(monkeypatch):
    real = solvers.reason_knapsack
    monkeypatch.setattr(solvers, "reason_knapsack", lambda t, d="Standard": {**real(t, d), "chosen_mask": (1 << len(solvers.parse(t)["opts"])) - 1})
    assert E.evaluate(E.make_cases(31, .15), fresh_draws=10)["systems"]["engine_standard"]["infeasible"] > 0

def test_silent_input_loss_is_detected(monkeypatch):
    old = solvers.parse
    def lossy(t): p = old(t); return {**p, "opts": p["opts"][1:], "unread": []}
    monkeypatch.setattr(solvers, "parse", lossy); r = E.evaluate(E.make_cases(31, .15), fresh_draws=10)
    assert r["parse_fidelity"]["options_dropped_silently"] > 0 and E.claims(r, "dev")["no silent loss of user input"].startswith("not supported")

# ---- parser defects the harness found, kept as permanent regression tests ----
def test_names_with_colons_pipes_commas_are_read():
    p = solvers.parse("Budget of 20\n- Phase 1: pilot: cost 3 value 4\n- Brand | partner: cost 2, value 3\n- R&D, core: cost 5 value 9")
    assert [(o["n"], o["c"], o["v"]) for o in p["opts"]] == [("Phase 1: pilot", 3, 4), ("Brand | partner", 2, 3), ("R&D, core", 5, 9)] and p["unread"] == []
def test_thousands_separators_and_currency():
    p = solvers.parse("Budget of $1,200.50\n- A: cost $700 value 5,000\n- B: cost 1,100 value 9"); assert p["budget"] == 1200.5 and [(o["c"], o["v"]) for o in p["opts"]] == [(700, 5000), (1100, 9)]
def test_a_comma_separator_is_not_swallowed_into_the_number():
    assert [(o["c"], o["v"]) for o in solvers.parse("- A: cost 4, value 7")["opts"]] == [(4, 7)]
def test_unreadable_option_lines_are_reported_and_lower_confidence():
    r = solvers.reason_knapsack("Budget of 10\n- A: cost 4 value 7\n- B: cost 5 value 9\n- C: cost four value 3"); assert r["confidence"] == "provisional" and "C: cost four" in r["warnings"][0]
    assert solvers.reason_knapsack("Budget of 10\n- A: cost 4 value 7\n- B: cost 5 value 9")["warnings"] == []

c = TestClient(main.app)
def _user(email):
    v = c.get("/api/v1/terms").json()
    return {"Authorization": "Bearer " + c.post("/api/v1/auth/register", json={"email": email, "password": "password1", "accept_terms": True, "age_confirmed": True, "terms_version": v["terms_version"], "privacy_version": v["privacy_version"]}).json()["token"]}
def test_ask_tells_the_user_when_a_line_was_left_out():
    h = _user("ev1@example.com"); r = c.post("/api/v1/ask", headers=h, json={"text": "Budget of 10\n- A: cost 4 value 7\n- B: cost 5 value 9\n- C: cost four value 3"}).json()
    assert r["tier"] == 1 and r["confidence"] == "provisional" and any("could not be read" in n for n in r["notices"])
    r = c.post("/api/v1/reason", headers=h, json={"text": "Budget of 10\n- A: cost 4 value 7\n- B: cost 5 value 9\n- C: cost four value 3"}).json(); assert r["warnings"] and r["confidence"] == "provisional"

# ---- calibration ----
def test_stability_calibration_is_reported_and_not_overclaiming_on_clean_input():
    r = E.evaluate([x for x in E.make_cases(11, .6) if x["family"] in ("random", "correlated")], fresh_draws=60)["stability_calibration"]
    assert r and r["cases"] >= 5 and r["mean_signed_error_ci95"][1] <= .05

# ---- claims gate ----
def _report(**kw):
    base = {"systems": {"engine_standard": {"cases": 250}}, "comparisons": {f"x{i}": {"ci95_pct": [.5, 2.0], "engine_worse": 0} for i in range(4)}, "reliability": {"high": {"n": 250, "ci95": [.97, 1.0]}},
            "stability_calibration": {"cases": 80, "mean_signed_error_ci95": [-.02, .03]}, "parse_fidelity": {"options_dropped_silently": 0, "cases_affected": []}}
    base.update(kw); return base
def test_claims_gate_requires_private_suite_and_evidence():
    assert E.claims(_report(), "dev")["more accurate than simple baselines on unseen cases"].startswith("not supported: run on the private suite")
    assert E.claims(_report(), "private")["more accurate than simple baselines on unseen cases"] == "supported"
    assert E.claims(_report(comparisons={"x": {"ci95_pct": [-.1, 2.0], "engine_worse": 0}}), "private")["more accurate than simple baselines on unseen cases"].startswith("not supported")
    assert E.claims(_report(systems={"engine_standard": {"cases": 120}}), "private")["more accurate than simple baselines on unseen cases"].startswith("not supported")
    assert E.claims(_report(), "private")["stated stability does not overclaim"] == "supported"
    assert E.claims(_report(stability_calibration={"cases": 80, "mean_signed_error_ci95": [.04, .2]}), "private")["stated stability does not overclaim"].startswith("not supported")
    assert E.claims(_report(), "private")["better than other AI systems in general"].startswith("not supported")
    assert E.claims(_report(), "private")["preferred by independent experts over other systems"].startswith("not supported: no blinded study")
    good = {"raters": 3, "items": 80, "win_rate_ci95": [.6, .8], "sign_test_p": .01, "kappa": .55}; assert E.claims(_report(), "private", good)["preferred by independent experts over other systems"] == "supported"
    assert E.claims(_report(), "private", {**good, "kappa": .1})["preferred by independent experts over other systems"].startswith("not supported")
    assert E.claims(_report(), "private", {**good, "raters": 1})["preferred by independent experts over other systems"].startswith("not supported")

def test_private_suite_is_not_shipped_and_cannot_run_without_its_file(tmp_path):
    assert not (E.EVAL / "private").exists()
    with pytest.raises(SystemExit, match="not shipped"): E.run_suite("private", path=tmp_path / "none.jsonl")

def test_make_private_writes_a_gitignored_set_that_differs_from_dev(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "EVAL", tmp_path); E.main(["make-private", "--seed", "99", "--scale", "0.2"]); p = tmp_path / "private" / "cases.jsonl"
    assert p.exists() and (tmp_path / "private" / ".gitignore").read_text() == "*\n"
    dev = {c["text"] for c in E.make_cases(1000)}; assert len([c for c in E.read_cases(p) if c["text"] in dev and c["family"] != "malformed"]) < 3
    rep, cl, md = E.run_suite("private", path=p, out=tmp_path / "r.md"); assert (tmp_path / "r.md").exists() and (tmp_path / "r.json").exists() and "Private suite" in md and cl["more accurate than simple baselines on unseen cases"].startswith("not supported: ")
    assert sum(1 for c in E.make_cases(99, 2.5) if c["expect"] == "answer") >= 200      # the default private suite is large enough for the claims gate

# ---- blinded preference ----
def _resp(system, n=12, text=None): return [{"scenario_id": f"s{i:02d}", "system": system, "recommendation": text or f"{system} says do {i}", "why": "because", "alternatives": ["a"], "risks": ["r"], "questions": ["q"]} for i in range(1, n + 1)]
SC = [{"id": f"s{i:02d}", "text": f"scenario {i}"} for i in range(1, 13)]
def test_blind_pack_hides_system_names_and_balances_positions():
    rows, key = E.blind_pack({"engine": _resp("engine"), "framework_only": _resp("framework_only", text="generic")}, SC, raters=2, seed=1)
    assert len(rows) == 24 and len(key) == 12 and all(set(k) == {"A", "B", "scenario"} for k in key.values())
    blob = json.dumps(rows); assert "framework_only" not in blob and "Tier" not in blob and "Built-in" not in blob
    a_engine = sum(k["A"] == "engine" for k in key.values()); assert 2 <= a_engine <= 10                                        # not always in the same position
    assert {r["rater_id"] for r in rows} == {"R1", "R2"} and all(r["winner"] == "" for r in rows)

def _ratings(rows, key, prefer="engine", flip=0.0, seed=0, raters=("R1", "R2")):
    R = random.Random(seed); out = []
    for r in rows:
        if r["rater_id"] not in raters: continue
        k = key[r["item_id"]]; want = "A" if k["A"] == prefer else "B"
        if R.random() < flip: want = "B" if want == "A" else "A"
        out.append({**r, "winner": want})
    return out
def test_blind_scoring_recovers_the_preference_and_agreement():
    rows, key = E.blind_pack({"engine": _resp("engine", 12), "framework_only": _resp("framework_only", 12)}, SC, raters=2, seed=2)
    s = E.blind_score(_ratings(rows, key), key); assert s["focus_wins"] == 12 and s["focus_losses"] == 0 and s["win_rate"] == 1.0 and s["kappa"] == 1.0 and s["sign_test_p"] < .001 and s["per_opponent"]["framework_only"]["wins"] == 12
    s = E.blind_score(_ratings(rows, key, prefer="framework_only"), key); assert s["focus_wins"] == 0 and s["win_rate"] == 0.0
    s = E.blind_score(_ratings(rows, key, flip=.5, seed=4), key); assert s["kappa"] is not None and s["kappa"] < .6 and s["win_rate_ci95"][0] < .5
def test_blind_scoring_ignores_bad_rows_and_counts_ties():
    rows, key = E.blind_pack({"engine": _resp("engine", 4), "framework_only": _resp("framework_only", 4)}, SC[:4], raters=1, seed=3)
    rt = _ratings(rows, key); rt[0]["winner"] = "tie"; rt[1]["winner"] = "maybe"; rt.append({"item_id": "nope", "rater_id": "R1", "winner": "A"}); s = E.blind_score(rt, key)
    assert s["ties"] == 1 and s["items"] == 3 and s["focus_wins"] == 2
def test_blind_study_end_to_end_through_the_real_api(tmp_path):
    h = _user("ev2@example.com")
    post = lambda text, off: c.post("/api/v1/ask", headers=h, json={"text": text, "disable_tiers": off}).json()
    real = E.read_cases(E.EVAL / "scenarios_dev.jsonl")[1:4]; SCR = real
    eng = E.collect(real, post, "engine"); fw = E.collect(real, post, "framework_only"); assert all(r["recommendation"] for r in eng + fw) and any(e["recommendation"] != f["recommendation"] for e, f in zip(eng, fw))
    rows, key = E.blind_pack({"engine": eng, "framework_only": fw}, real, raters=2, seed=1); assert len(rows) == 6
    f = tmp_path / "p.csv"
    with open(f, "w", newline="", encoding="utf-8") as fh: w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    assert len(list(csv.DictReader(open(f, encoding="utf-8")))) == 6
