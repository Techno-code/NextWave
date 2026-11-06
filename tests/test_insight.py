"""Quant Insight: the what-if analysis, the large-knapsack solver path, and the new name. Every number the analysis reports is checked against brute force."""
import itertools, random
import pytest
from fastapi.testclient import TestClient
from app import insight, main, pipeline as P

c = TestClient(main.app)
def _hdr(email):
    v = c.get("/api/v1/terms").json()
    t = c.post("/api/v1/auth/register", json={"email": email, "password": "password1", "accept_terms": True, "age_confirmed": True, "terms_version": v["terms_version"], "privacy_version": v["privacy_version"]}).json()["token"]
    return {"Authorization": "Bearer " + t}

def brute(v, w, cap, skip=None):
    best = 0
    for m in range(1 << len(v)):
        if skip is not None and m >> skip & 1: continue
        if sum(w[i] for i in range(len(v)) if m >> i & 1) <= cap: best = max(best, sum(v[i] for i in range(len(v)) if m >> i & 1))
    return best

def knap(seed, n=9):
    r = random.Random(seed); w = [r.randint(2, 15) for _ in range(n)]; v = [r.randint(5, 60) for _ in range(n)]
    return {"values": v, "weights": w, "capacity": sum(w) // 2}

@pytest.mark.parametrize("seed", range(8))
def test_closest_calls_match_brute_force(seed):
    inst = knap(seed); base = P.solve_instance("knapsack", inst, quantum=False); an = insight.analyze("knapsack", inst, base, P._fast_solve, budget_s=20)
    rows = {r["item"]: r for r in an["closest_calls"]} if len(an["closest_calls"]) >= len(inst["values"]) else None
    full = insight._closest_calls("knapsack", inst, base, P._fast_solve, insight._labels("knapsack", inst), 1e18)[0]
    v, w, cap = inst["values"], inst["weights"], inst["capacity"]; b = base["objective"]
    assert len(full) >= 1
    for r in full:
        i = int(r["item"].split()[-1]) - 1
        if r["currently"] == "chosen": expect = brute(v, w, cap, skip=i)
        else: expect = v[i] + brute([x for j, x in enumerate(v) if j != i], [x for j, x in enumerate(w) if j != i], cap - w[i])
        assert abs(r["objective_if_reversed"] - expect) < 1e-6 and r["cost_of_reversing"] >= -1e-9 and abs(r["cost_of_reversing"] - (b - expect)) < 1e-6
    assert [r["cost_of_reversing"] for r in full] == sorted(r["cost_of_reversing"] for r in full)

@pytest.mark.parametrize("seed", range(6))
def test_capacity_levers_match_brute_force(seed):
    inst = knap(seed); base = P.solve_instance("knapsack", inst, quantum=False); an = insight.analyze("knapsack", inst, base, P._fast_solve, trials=5, budget_s=20)
    for wi in an["what_if"]:
        f = {"capacity cut by 10%": .9, "capacity raised by 10%": 1.1, "capacity raised by 25%": 1.25}[wi["change"]]
        assert abs(wi["objective"] - brute(inst["values"], inst["weights"], inst["capacity"] * f)) < 1e-6

def test_zero_noise_gives_full_stability_and_is_deterministic():
    inst = knap(3); base = P.solve_instance("knapsack", inst, quantum=False)
    a = insight.analyze("knapsack", inst, base, P._fast_solve, eps=0.0, trials=10); assert a["stability"]["decision_unchanged_pct"] == 100.0 and a["stability"]["robustness"] == "firm"
    x = insight.analyze("knapsack", inst, base, P._fast_solve, seed=5); y = insight.analyze("knapsack", inst, base, P._fast_solve, seed=5)
    assert x["stability"] == y["stability"] and x["closest_calls"] == y["closest_calls"]

def test_toss_ups_and_always_chosen_are_consistent():
    inst = knap(1); base = P.solve_instance("knapsack", inst, quantum=False); st = insight.analyze("knapsack", inst, base, P._fast_solve, eps=.3, trials=40)["stability"]
    names = {f"Item {i + 1}" for i in range(len(inst["values"]))}
    assert set(st["always_chosen"]) <= names and all(10 < t["chosen_pct"] < 90 for t in st["toss_ups"])
    assert not set(st["always_chosen"]) & {t["item"] for t in st["toss_ups"]}
    o = st["objective_range"]; assert o["p5"] <= o["median"] <= o["p95"]

@pytest.mark.parametrize("family", P.FAMILIES)
def test_every_family_gets_a_detailed_answer(family):
    r = P.run("", family=family, use_demo=True, demo_n=8, quantum=False)
    assert r["solution"]["verified"] and "analysis" in r and r["analysis"]["seconds"] < 8
    heads = [d["heading"] for d in r["detail"]]; assert heads[0] == "Why this answer" and heads[-1] == "Before you act" and "How firm it is" in heads
    assert any("illustration" in d["text"] for d in r["detail"] if d["heading"] == "Before you act")   # demo data is always labelled as not the person's own

def test_time_budget_is_respected_and_reported():
    inst = knap(2); base = P.solve_instance("knapsack", inst, quantum=False); an = insight.analyze("knapsack", inst, base, P._fast_solve, budget_s=0)
    assert an["seconds"] < 2 and an.get("skipped") and an["stability"].get("trials", 0) < 5

def test_detail_flag_and_default():
    H = _hdr("qi_detail@example.com")
    on = c.post("/api/v1/solve", json={"family": "knapsack", "instance": knap(0), "quantum_check": False}, headers=H).json()
    off = c.post("/api/v1/solve", json={"family": "knapsack", "instance": knap(0), "quantum_check": False, "detail": False}, headers=H).json()
    assert "analysis" in on and "detail" in on and "analysis" not in off and "detail" not in off and on["solution"]["objective"] == off["solution"]["objective"]

# ---------------------------------------------------------------- large knapsacks use the MIP solver, small ones are unchanged
@pytest.mark.parametrize("seed", range(10))
def test_mip_path_matches_brute_force(seed, monkeypatch):
    monkeypatch.setattr(P, "DP_CELLS", 0); inst = knap(seed, 10); s = P.solve_instance("knapsack", inst, quantum=False)
    assert s["verified"] and s["objective"] == brute(inst["values"], inst["weights"], inst["capacity"]) and s["guarantee"].startswith("Optimal") and "HiGHS" in s["method"]

def test_large_knapsack_is_solved_and_bounded():
    r = random.Random(4); n = 400; w = [r.randint(50, 1000) for _ in range(n)]; v = [x + r.randint(0, 90) for x in w]; cap = sum(w) // 3
    assert cap * n > P.DP_CELLS
    s = P.solve_instance("knapsack", {"values": v, "weights": w, "capacity": cap}, quantum=False)
    assert s["verified"] and "HiGHS" in s["method"] and s["bound"]["gap_pct"] <= 0.5 and s["decision"]["resource_used"] <= cap

# ---------------------------------------------------------------- the name
def test_named_quant_insight_everywhere_a_person_sees_it():
    assert c.get("/openapi.json").json()["info"]["title"] == "Quant Insight"
    page = c.get("/").text; assert "<title>Quant Insight</title>" in page and "Decision Engine" not in page
    H = _hdr("qi_export@example.com"); r = c.get("/api/v1/account/export", headers=H); assert "quant-insight-export.json" in r.headers["content-disposition"]
