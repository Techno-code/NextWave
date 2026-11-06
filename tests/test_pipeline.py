"""Solvers are checked against brute force on random instances; the pipeline and API are checked end to end.
The catalogue tests build a small index from the shipped 117-row dataset, so they need neither the 1M files nor a prebuilt index."""
import csv, itertools, json, math, random, threading
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app import bigdata, main, pipeline as P, ref_solvers as R

c = TestClient(main.app)
def _hdr(email):
    v = c.get("/api/v1/terms").json()
    t = c.post("/api/v1/auth/register", json={"email": email, "password": "password1", "accept_terms": True, "age_confirmed": True, "terms_version": v["terms_version"], "privacy_version": v["privacy_version"]}).json()["token"]
    return {"Authorization": "Bearer " + t}
H = _hdr("pipe1@example.com"); H2 = _hdr("pipe2@example.com")

# ---------------------------------------------------------------- solver correctness against brute force
def brute_knap(v, w, cap):
    best = 0
    for m in range(1 << len(v)):
        if sum(w[i] for i in range(len(v)) if m >> i & 1) <= cap: best = max(best, sum(v[i] for i in range(len(v)) if m >> i & 1))
    return best
@pytest.mark.parametrize("seed", range(25))
def test_knapsack_matches_brute_force(seed):
    r = random.Random(seed); n = r.randint(3, 10); v = [r.randint(1, 50) for _ in range(n)]; w = [r.randint(1, 15) for _ in range(n)]; cap = r.randint(5, sum(w))
    s = P.solve_instance("knapsack", {"values": v, "weights": w, "capacity": cap}, quantum=False)
    assert s["verified"] and s["objective"] == brute_knap(v, w, cap)
@pytest.mark.parametrize("seed", range(15))
def test_assignment_matches_brute_force(seed):
    r = random.Random(seed); k = r.randint(2, 6); C = [[r.randint(1, 40) for _ in range(k)] for _ in range(k)]; mx = seed % 2 == 0
    best = (max if mx else min)(sum(C[i][p[i]] for i in range(k)) for p in itertools.permutations(range(k)))
    s = P.solve_instance("assignment", {"cost": C, "maximize": mx}); assert s["verified"] and s["objective"] == best
@pytest.mark.parametrize("seed", range(15))
def test_set_cover_matches_brute_force(seed):
    r = random.Random(seed); U = list(range(1, 8)); S = [sorted(r.sample(U, r.randint(1, 4))) for _ in range(7)] + [U[:4], U[4:]]; cost = [r.randint(1, 9) for _ in S]
    best = min(sum(cost[j] for j in sub) for k in range(1, len(S) + 1) for sub in itertools.combinations(range(len(S)), k) if set().union(*[set(S[j]) for j in sub]) >= set(U))
    s = P.solve_instance("set_cover", {"universe": U, "sets": S, "costs": cost}); assert s["verified"] and abs(s["objective"] - best) < 1e-9
@pytest.mark.parametrize("seed", range(10))
def test_facility_matches_brute_force(seed):
    r = random.Random(seed); n, m = 5, 6; f = [r.randint(5, 20) for _ in range(n)]; D = [[r.randint(1, 15) for _ in range(n)] for _ in range(m)]; d = [r.randint(1, 4) for _ in range(m)]
    best = min(sum(f[j] for j in S) + sum(d[i] * min(D[i][j] for j in S) for i in range(m)) for k in range(1, n + 1) for S in itertools.combinations(range(n), k))
    s = P.solve_instance("facility_location", {"fixed": f, "dist": D, "demand": d}); assert s["verified"] and abs(s["objective"] - best) < 1e-6
@pytest.mark.parametrize("seed", range(10))
def test_mdk_matches_brute_force(seed):
    r = random.Random(seed); n = 9; v = [r.randint(5, 40) for _ in range(n)]; A = [[r.randint(1, 8) for _ in range(n)] for _ in range(2)]; b = [20, 18]
    best = max(sum(v[i] for i in range(n) if m >> i & 1) for m in range(1 << n) if all(sum(A[k][i] for i in range(n) if m >> i & 1) <= b[k] for k in range(2)))
    s = P.solve_instance("multi_dim_knapsack", {"values": v, "A": A, "b": b}); assert s["verified"] and s["objective"] == best
def test_scheduling_wspt_is_optimal():
    r = random.Random(5); t = [r.randint(1, 9) for _ in range(6)]; w = [r.randint(1, 5) for _ in range(6)]
    s = P.solve_instance("scheduling", {"durations": t, "weights": w, "machines": 1}); assert s["verified"] and "Smith" in s["method"]
def test_scheduling_lpt_bound_holds():
    for seed in range(10):
        r = random.Random(seed); t = [r.randint(1, 20) for _ in range(9)]; s = P.solve_instance("scheduling", {"durations": t, "machines": 3}); assert s["verified"] and s["objective"] >= max(sum(t) / 3, max(t)) - 1e-9
def test_routing_exact_on_small_single_route():
    s = P.solve_instance("routing", {"depot": [0, 0], "points": [[3, 0], [3, 4], [0, 4]], "demand": [1, 1, 1], "capacity": 10}); assert s["verified"] and abs(s["objective"] - 14) < 1e-9
def test_budget_allocation_beats_equal_split():
    inst = {"curves": [{"name": "A", "type": "exp", "a": 60, "b": .3}, {"name": "B", "type": "log", "a": 30, "b": .1}, {"name": "C", "type": "sqrt", "a": 12, "b": 1}], "budget": 30, "step": 1}
    s = P.solve_instance("budget_allocation", inst); assert s["verified"]
    eq = sum(P._curve(cu)(10) for cu in inst["curves"]); assert s["objective"] >= eq - 1e-9
def test_budget_minimum_and_maximum_respected():
    s = P.solve_instance("budget_allocation", {"curves": [{"name": "A", "type": "log", "a": 50, "b": .5, "min": 5, "max": 8}, {"name": "B", "type": "log", "a": 10, "b": .1}], "budget": 20, "step": 1})
    a = {x["option"]: x["spend"] for x in s["decision"]["allocation"]}; assert s["verified"] and 5 <= a["A"] <= 8 + 1e-9 and sum(a.values()) <= 20 + 1e-9
def test_portfolio_cardinality():
    inst = P.demo_instance("portfolio_selection", 6, 3); inst["max_assets"] = 3; inst["max_weight"] = .6; s = P.solve_instance("portfolio_selection", inst)
    assert s["verified"] and len(s["decision"]["weights"]) <= 3

# ---------------------------------------------------------------- bad input is refused in plain words, never crashes
@pytest.mark.parametrize("fam,inst,frag", [
    ("knapsack", {"values": [1, 2], "weights": [1], "capacity": 3}, "expected 2"),
    ("knapsack", {"values": [1, 2], "weights": [-1, 1], "capacity": 3}, "negative"),
    ("knapsack", {"values": [1, 2], "weights": [1, 1]}, "capacity"),
    ("knapsack", {"values": ["a"], "weights": [1], "capacity": 3}, "number"),
    ("set_cover", {"universe": [1, 2, 3], "sets": [[1], [2]]}, "no set covers"),
    ("portfolio_selection", {"mu": [.1, .2], "cov": [[.04, 0], [0, .01]], "target_return": .9}, "above the highest"),
    ("portfolio_selection", {"mu": [.1, .2], "cov": [[.04, 1], [0, .01]], "target_return": .1}, "symmetric"),
    ("assignment", {"cost": [[1, 2], [3]]}, "rectangular"),
    ("routing", {"depot": [0, 0], "points": [[1, 1]], "demand": [9], "capacity": 5}, "larger than the vehicle"),
    ("budget_allocation", {"curves": [{"type": "cubic", "a": 1, "b": 1}], "budget": 5}, "Unknown curve"),
    ("facility_location", {"fixed": [1], "dist": [[1]], "demand": [1], "k": 3}, "between 1 and"),
    ("nonsense", {}, "Unknown family")])
def test_bad_input_is_explained(fam, inst, frag):
    with pytest.raises(ValueError) as e: P.solve_instance(fam, inst)
    assert frag.lower() in str(e.value).lower()
def test_verifier_catches_a_wrong_solver(monkeypatch):
    """If the reference solver returned an over-capacity selection the checks must fail and the result must not count as verified."""
    monkeypatch.setattr(R, "knapsack", lambda v, w, cap: (sum(v), list(range(len(v)))))
    s = P.solve_instance("knapsack", {"values": [5, 6, 7], "weights": [4, 4, 4], "capacity": 5}, quantum=False)
    assert not s["verified"] and any(not x["passed"] for x in s["checks"])
def test_every_family_has_a_format_and_demo():
    for f in P.FAMILIES: assert f in P.FORMATS and P.solve_instance(f, P.demo_instance(f, 6, 2), quantum=False)["verified"]

# ---------------------------------------------------------------- quantum cross-check
def test_quantum_check_matches_exact_on_small_knapsack():
    s = P.solve_instance("knapsack", {"values": [10, 13, 7, 8], "weights": [3, 4, 2, 3], "capacity": 7}); q = s["quantum_check"]
    assert q["ran"] and q["matches_exact"] and "advantage" in q["note"] and "No quantum advantage" in q["note"]
def test_quantum_check_says_why_it_did_not_run():
    q = P.solve_instance("knapsack", {"values": list(range(1, 21)), "weights": [3] * 20, "capacity": 30})["quantum_check"]; assert not q["ran"] and "qubit" in q["reason"]
    q = P.solve_instance("knapsack", {"values": [1, 2], "weights": [1.5, 2], "capacity": 3})["quantum_check"]; assert not q["ran"] and "integer" in q["reason"]

# ---------------------------------------------------------------- catalogue index (small index built from the shipped dataset)
@pytest.fixture(scope="module")
def small_index(tmp_path_factory):
    d = tmp_path_factory.mktemp("idx"); src = list(csv.DictReader(open(bigdata.ROOT / "data" / "business_problems_dataset.csv", encoding="utf-8-sig")))
    pcsv, scsv = d / "p.csv", d / "s.csv"
    with open(pcsv, "w", newline="") as f: w = csv.DictWriter(f, list(src[0])); w.writeheader(); w.writerows(src)
    with open(scsv, "w", newline="") as f:
        w = csv.DictWriter(f, ["id", "playbook_id"]); w.writeheader()
        for r in src:
            pb = bigdata.playbook_for(r["formulation_family"], 20, r["trend_sensitivity"]); w.writerow({"id": r["id"], "playbook_id": pb["playbook_id"]})
    out = d / "idx.sqlite"; n = bigdata.build(pcsv, scsv, out, log=lambda *_: None); assert n == len(src); return out, src
@pytest.fixture
def use_index(small_index, monkeypatch):
    monkeypatch.setattr(bigdata, "INDEX", small_index[0]); monkeypatch.setattr(bigdata, "_local", threading.local()); monkeypatch.setattr(bigdata, "_dimcache", {}); return small_index[1]
def test_index_roundtrip(use_index):
    src = use_index; r = bigdata.get("BP0001"); assert r["problem"] == src[0]["problem"] and r["formulation_family"] == src[0]["formulation_family"] and r["playbook_id"] in bigdata.PLAYBOOKS
    assert bigdata.get("BP9999999") is None and bigdata.get("nonsense") is None
    st = bigdata.stats(); assert st["available"] and st["rows"] == len(src) and sum(st["families"].values()) == len(src)
def test_search_finds_the_row_it_was_given(use_index):
    hits = bigdata.search(use_index[0]["problem"], 3); assert hits and hits[0]["problem"] == use_index[0]["problem"]
    assert bigdata.search("", 3) == [] and bigdata.search("zzzxqv", 3) == []
def test_build_rejects_misaligned_files(tmp_path):
    (tmp_path / "p.csv").write_text("id,domain,problem,decision_type,objective,key_constraints,typical_variables,formulation_family,trend_sensitivity\nBP1,d,p,t,o,k,v,knapsack,low\n")
    (tmp_path / "s.csv").write_text("id,playbook_id\nBP2,PB-KNP-SM-L\n")
    with pytest.raises(ValueError, match="ids differ"): bigdata.build(tmp_path / "p.csv", tmp_path / "s.csv", tmp_path / "o.sqlite", log=lambda *_: None)
def test_playbook_expansion_resolves_every_reference():
    for pb in bigdata.PLAYBOOKS.values():
        e = bigdata.expand(pb); assert e["steps"] and e["theorems"] and e["references"]
        assert len(e["theorems"]) == len(pb["core_theorem_ids"].split(";")) and len(e["references"]) == len(pb["core_ref_ids"].split(";"))
def test_every_family_scale_sensitivity_has_a_playbook():
    for f in P.FAMILIES:
        for n in (5, 50, 1000, 10000):
            for s in ("low", "medium", "high"): assert bigdata.playbook_for(f, n, s)["formulation_family"] == f

# ---------------------------------------------------------------- pipeline
CLS = lambda t: [{"family": "knapsack", "p": .7}, {"family": "portfolio_selection", "p": .2}]
def test_pipeline_text_with_options_is_solved_and_verified():
    r = P.run("Budget of 10\n- A: cost 4 value 7\n- B: cost 5 value 9\n- C: cost 6 value 10", CLS, quantum=False)
    assert r["family"] == "knapsack" and r["tier"] == 1 and r["solution"]["verified"] and r["solution"]["objective"] == 17 and r["confidence"]["level"] in ("high", "medium")
def test_pipeline_without_data_gives_formulation_not_numbers():
    r = P.run("We need to pick warehouse sites to open", lambda t: [{"family": "facility_location", "p": .8}])
    assert r["tier"] == 3 and "solution" not in r and r["confidence"]["level"] == "provisional" and r["playbook"]["steps"] and r["questions"]
def test_pipeline_demo_is_labelled_illustrative():
    r = P.run("Assign nurses to shifts", lambda t: [{"family": "assignment", "p": .9}], use_demo=True)
    assert r["solution"]["verified"] and r["instance_source"].startswith("illustrative") and r["confidence"]["level"] == "provisional" and "illustrative" in r["recommendation"]
def test_pipeline_explicit_family_and_instance_win_over_classifier():
    r = P.run("anything", CLS, family="assignment", instance={"cost": [[4, 1], [2, 0]]}, quantum=False); assert r["family"] == "assignment" and r["solution"]["objective"] == 1 + 2
@pytest.mark.parametrize("fam", P.FAMILIES)
def test_instance_shape_identifies_every_family(fam):
    inst = P.demo_instance(fam, 6, 4); assert P.family_from_instance(inst) == fam
    r = P.run("", None, instance=inst, quantum=False); assert r["family"] == fam and r["family_source"].startswith("shape") and r["solution"]["verified"] and r["confidence"]["level"] == "high"
def test_mismatched_family_and_numbers_are_refused():
    with pytest.raises(ValueError, match="look like a assignment|look like an? assignment|assignment instance"): P.run("", None, family="knapsack", instance={"cost": [[1, 2], [2, 1]]})
def test_pipeline_rejects_unknown_family_and_empty_input():
    with pytest.raises(ValueError): P.run("x", family="nope")
    with pytest.raises(ValueError): P.run("")
def test_low_classifier_confidence_is_not_reported_as_high():
    r = P.run("pick the best things", lambda t: [{"family": "knapsack", "p": .3}], use_demo=True, quantum=False); assert r["family_source"] == "classifier" and r["confidence"]["level"] == "provisional"
    # numbers whose shape names the family are decisive, so a weak classifier no longer matters
    r = P.run("pick the best things", lambda t: [{"family": "assignment", "p": .3}], instance={"values": [3, 4], "weights": [1, 2], "capacity": 2}, quantum=False); assert r["family"] == "knapsack" and r["confidence"]["level"] == "high"
def test_classifier_decides_over_catalogue_vote():
    r = P.run("anything at all", lambda t: [{"family": "scheduling", "p": .9}], use_demo=True, quantum=False); assert r["family"] == "scheduling"
def test_pipeline_unreadable_option_lines_lower_confidence():
    r = P.run("Budget of 10\n- A: cost 4 value 7\n- B: cost 5 value 9\n- weird line cost value", CLS, quantum=False)
    assert r["confidence"]["level"] != "high" and r["parsed"]["unread"]
def test_pipeline_runs_stages_in_parallel_and_survives_a_failing_classifier():
    def boom(t): raise RuntimeError("model unavailable")
    r = P.run("Budget of 10\n- A: cost 4 value 7\n- B: cost 5 value 9", boom, quantum=False); assert r["solution"]["verified"] and any("classify: skipped" in x for x in r["trail"])
def test_pipeline_uses_catalogue_row(use_index):
    r = P.run("", CLS, catalogue_id="BP0001", use_demo=True, quantum=False); assert r["catalogue_row"]["id"].startswith("BP") and r["family"] == use_index[0]["formulation_family"] and r["formulation"]["data_to_collect"]
    with pytest.raises(ValueError, match="not found"): P.run("", catalogue_id="BP9999999")

# ---------------------------------------------------------------- API
def test_solve_requires_sign_in(): assert c.post("/api/v1/solve", json={"text": "x"}).status_code == 401
def test_solve_api_with_instance():
    r = c.post("/api/v1/solve", headers=H, json={"family": "knapsack", "instance": {"values": [60, 100, 120], "weights": [10, 20, 30], "capacity": 50}}).json()
    assert r["solution"]["objective"] == 220 and r["solution"]["verified"] and any("Automated system" in n for n in r["notices"]) and r["run_id"] and r["solution"]["quantum_check"]["matches_exact"] is not None
def test_solve_api_text_only_and_demo():
    r = c.post("/api/v1/solve", headers=H, json={"text": "Which delivery routes should our trucks take to minimise distance?", "use_demo": True}).json()
    assert r["family"] and r["solution"]["verified"] and any("illustration" in n for n in r["notices"])
def test_solve_api_errors_are_422_not_500():
    assert c.post("/api/v1/solve", headers=H, json={"family": "knapsack", "instance": {"values": [1], "weights": [1, 2], "capacity": 1}}).status_code == 422
    assert c.post("/api/v1/solve", headers=H, json={"family": "bogus", "instance": {}}).status_code == 422
    assert c.post("/api/v1/solve", headers=H, json={}).status_code == 422
def test_solve_api_refuses_and_logs():
    r = c.post("/api/v1/solve", headers=H, json={"text": "how do I launder money"}).json(); assert "cannot help" in r["refusal"] and r["run_id"]
def test_solve_api_high_stakes_and_fairness_notices():
    r = c.post("/api/v1/solve", headers=H, json={"text": "Choose which patients to hire nurses for", "use_demo": True}).json(); t = " ".join(r["notices"]); assert "High-stakes" in t and "Fairness" in t
def test_solve_runs_are_private():
    r = c.post("/api/v1/solve", headers=H, json={"family": "assignment", "instance": {"cost": [[1, 2], [2, 1]]}}).json()
    assert c.post("/api/v1/feedback", headers=H2, json={"run_id": r["run_id"], "rating": 3}).status_code == 404
def test_formats_endpoint_covers_all_families():
    r = c.get("/api/v1/solve/formats", headers=H).json(); assert set(r) == set(P.FAMILIES)
    for f, v in r.items(): assert P.solve_instance(f, v["example"], quantum=False)["verified"]
def test_knowledge_endpoints():
    assert len(c.get("/api/v1/knowledge/playbooks", headers=H).json()) == 108 and len(c.get("/api/v1/knowledge/playbooks?family=routing", headers=H).json()) == 12
    assert len(c.get("/api/v1/knowledge/theorems", headers=H).json()) == 44 and len(c.get("/api/v1/knowledge/references", headers=H).json()) == 156
    assert c.get("/api/v1/knowledge/theorems").status_code == 401
def test_catalogue_endpoints(use_index):
    assert c.get("/api/v1/catalogue/stats", headers=H).json()["available"]
    r = c.get("/api/v1/catalogue/BP0001", headers=H).json(); assert r["playbook"]["steps"] and r["formulation"]["constraints"]
    assert c.get("/api/v1/catalogue/BP9999999", headers=H).status_code == 404
    assert c.get("/api/v1/catalogue/search", params={"q": use_index[0]["problem"]}, headers=H).json()["results"]
    assert c.get("/api/v1/catalogue/search", params={"q": "ab"}, headers=H).status_code == 422
    assert c.post("/api/v1/solve", headers=H, json={"catalogue_id": "BP0001", "use_demo": True}).json()["catalogue_row"]
def test_catalogue_absent_is_explained(monkeypatch, tmp_path):
    monkeypatch.setattr(bigdata, "INDEX", tmp_path / "none.sqlite"); assert c.get("/api/v1/catalogue/search", params={"q": "knapsack budget"}, headers=H).status_code == 503
    assert c.get("/api/v1/catalogue/stats", headers=H).json()["available"] is False and c.get("/api/v1/health").json()["catalogue"] is False
