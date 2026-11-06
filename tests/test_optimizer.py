"""Optimiser: no change for single-option requests, tier and channel variables, QUBO correctness, backends, and the paid-hardware guard."""
import json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import numpy as np, pytest
from app import optimizer as O

def T(n=6, seed=0):
    R = np.random.default_rng(seed); return [{"name": f"T{i}", "cat": i % 3, "p": round(float(.3 + R.random() * .6), 2), "E": round(float(20 + R.random() * 40)), "hl": 10, "c": int(2 + i % 4)} for i in range(n)]

def test_single_option_requests_match_the_original_engine_exactly():
    gold = json.loads((Path(__file__).parent / "golden" / "optimizer_v1.json").read_text())
    cases = [O.Opt(), O.Opt(seed=3, budget=9, risk="low"), O.Opt(seed=5, budget=15, risk="high", max_picks=3), O.Opt(seed=9, budget=6, lam_overlap=2, ship_days=5)]
    for c, g in zip(cases, gold):
        r = O.run_opt(c)
        assert [(p["trends"], p["spend"], p["net_value"], p["verified"]) for p in r["portfolios"]] == [(p["trends"], p["spend"], p["net_value"], p["verified"]) for p in g["portfolios"]]
        assert r["recommended"] == g["recommended"] and r["waste_avoided_pct"] == g["waste_avoided_pct"] and r["classical"]["objective"] == g["classical_objective"]
        assert {k: r["quantum"].get(k) for k in g["quantum"]} == g["quantum"]

def test_branch_and_bound_matches_brute_force_with_tiers_and_channels():
    b = O.Opt(trends=T(5), budget=11, max_picks=4, tiers=O.DEFAULT_TIERS, channels=O.DEFAULT_CHANNELS[:2]); M = O.Model(b); res, proven, nodes = O.exact(M)
    import itertools
    best = []
    for A in itertools.product(*[[-1] + M.by_trend[i] for i in range(M.n)]):
        if M.ok(list(A)): best.append((M.f(list(A)), A))
    best.sort(key=lambda x: -x[0])
    assert proven and abs(res[0][0] - best[0][0]) < 1e-9 and [round(v, 6) for v, _ in res] == [round(v, 6) for v, _ in best[:3]]

def test_every_trend_gets_at_most_one_option_and_constraints_hold():
    r = O.run_opt(O.Opt(trends=T(8), budget=14, max_picks=4, tiers=O.DEFAULT_TIERS, channels=O.DEFAULT_CHANNELS, backend="anneal"))
    for p in r["portfolios"]:
        assert p["verified"] and p["spend"] <= 14 and len(p["plan"]) <= 4 and len({x["trend"] for x in p["plan"]}) == len(p["plan"])
        assert all(x["tier"] in {"Test", "Standard", "Full"} and x["channel"] in {"Organic", "Paid social", "Creator partnership"} for x in p["plan"])
    assert r["quantum"]["verified"] and r["quantum"]["ratio"] <= 1.0001 and r["classical"]["proven_optimal"]

def test_tiers_change_the_answer_when_a_cheap_test_is_better_value():
    cheap = [O.Tier(name="Test", cost_mult=.3, value_mult=.8), O.Tier(name="Full", cost_mult=2, value_mult=1.1)]
    r = O.run_opt(O.Opt(trends=T(6), budget=8, max_picks=6, tiers=cheap)); assert any(x["tier"] == "Test" for x in r["portfolios"][0]["plan"])
    r0 = O.run_opt(O.Opt(trends=T(6), budget=8, max_picks=6)); assert r["portfolios"][0]["net_value"] > r0["portfolios"][0]["net_value"]

def test_channel_overlap_penalises_putting_trends_in_the_same_channel():
    one = [O.Channel(name="A", cost_mult=1, reach_mult=1), O.Channel(name="B", cost_mult=1, reach_mult=1)]
    M = O.Model(O.Opt(trends=T(4), budget=20, channels=one, lam_channel=2.0)); a = M.by_trend[0][0]; same = M.by_trend[1][0]; other = M.by_trend[1][1]
    assert M.S[a][same] > M.S[a][other] and M.S[a][other] == O.Model(O.Opt(trends=T(4), budget=20)).S[0][1]
    plan = O.run_opt(O.Opt(trends=T(4), budget=20, channels=one, lam_channel=2.0))["portfolios"][0]["plan"]; assert len({x["channel"] for x in plan}) == len(plan) or len(plan) < 2

def test_qubo_energy_equals_the_vectorised_energy_used_by_qaoa():
    M = O.Model(O.Opt(trends=T(4), budget=9, tiers=O.DEFAULT_TIERS[:2])); Q, off, nv = O.build_qubo(M); nS = O.slack_bits(9); R = np.random.default_rng(0)
    for z in R.integers(0, 1 << nv, 120):
        x = [(int(z) >> i) & 1 for i in range(nv)]; assert abs(O.energies(M, np.array([int(z)]), nS)[0] - O.qubo_energy(Q, off, x)) < 1e-8

def test_qubo_minimum_is_a_feasible_optimum():
    M = O.Model(O.Opt(trends=T(3), budget=6, tiers=O.DEFAULT_TIERS[:2])); Q, off, nv = O.build_qubo(M); best = min(range(1 << nv), key=lambda z: O.qubo_energy(Q, off, [(z >> i) & 1 for i in range(nv)]))
    A = M.from_mask(best & ((1 << M.nO) - 1)); assert M.mask_ok(best & ((1 << M.nO) - 1)) and abs(M.f(A) - O.exact(M)[0][0][0]) < 1e-8

def test_simulator_reports_qubit_cap_instead_of_failing():
    r = O.run_opt(O.Opt(trends=T(8), budget=12, tiers=O.DEFAULT_TIERS, channels=O.DEFAULT_CHANNELS))
    assert "skipped" in r["quantum"] and "anneal" in r["quantum"]["skipped"] and r["recommended"] == "classical exact"

def test_compare_backends_runs_both_and_never_beats_a_proven_optimum():
    r = O.run_opt(O.Opt(trends=T(5), budget=10, tiers=O.DEFAULT_TIERS[:2], compare_backends=True)); assert set(r["backends"]) == {"simulator", "anneal"} and all(b["ratio"] <= 1.0001 for b in r["backends"].values() if "ratio" in b)

def test_annealer_is_deterministic_for_a_seed():
    a = O.run_opt(O.Opt(trends=T(7), budget=12, backend="anneal", seed=4))["quantum"]; b = O.run_opt(O.Opt(trends=T(7), budget=12, backend="anneal", seed=4))["quantum"]
    assert {k: a[k] for k in ("objective", "trends", "reads_reaching_best")} == {k: b[k] for k in ("objective", "trends", "reads_reaching_best")}

def test_node_cap_is_reported_not_hidden(monkeypatch):
    monkeypatch.setattr(O, "NODE_CAP", 30); r = O.run_opt(O.Opt(trends=T(12), budget=15)); assert r["classical"]["proven_optimal"] is False and "not a proof" in r["classical"]["note"]

def test_bad_requests():
    with pytest.raises(ValueError): O.run_opt(O.Opt(backend="gpu"))
    with pytest.raises(ValueError): O.run_opt(O.Opt(trends=T(17)))
    with pytest.raises(ValueError, match="No feasible"): O.run_opt(O.Opt(budget=-1))

# ---- paid hardware guard and adapter, against a local stand-in for a solver service ----
class Solver(BaseHTTPRequestHandler):
    seen = []
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"]))); Solver.seen.append((self.headers.get("authorization"), body))
        nv, Q = body["num_vars"], {(i, j): v for i, j, v in body["qubo"]}
        # brute-force sampler: returns the 5 lowest-energy bitstrings (a stand-in for annealing hardware)
        import itertools
        E = sorted(((body["offset"] + sum(v * x[i] * x[j] for (i, j), v in Q.items()), x) for x in itertools.product([0, 1], repeat=nv)), key=lambda t: t[0])[:5] if nv <= 14 else []
        out = json.dumps({"backend": "test-solver", "samples": [{"bits": "".join(map(str, x)), "energy": e} for e, x in E]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(out)
    def log_message(self, *a): pass
@pytest.fixture
def solver(monkeypatch):
    s = HTTPServer(("127.0.0.1", 0), Solver); threading.Thread(target=s.serve_forever, daemon=True).start(); Solver.seen.clear()
    monkeypatch.setenv("QUBO_SOLVER_URL", f"http://127.0.0.1:{s.server_port}/solve"); monkeypatch.setenv("QUBO_SOLVER_TOKEN", "tok"); monkeypatch.setenv("QUBO_SOLVER_COST_PER_RUN", "2.5"); yield s; s.shutdown()

def test_hardware_refused_when_not_configured(monkeypatch):
    monkeypatch.delenv("QUBO_SOLVER_URL", raising=False)
    with pytest.raises(ValueError, match="No hardware solver is configured"): O.run_opt(O.Opt(backend="hardware", hardware_confirm=True, hardware_max_cost=10))

def test_hardware_needs_a_stated_cost_confirmation_and_cap(solver, monkeypatch):
    with pytest.raises(ValueError, match="confirm"): O.run_opt(O.Opt(trends=T(4), budget=8, backend="hardware", hardware_max_cost=10))
    with pytest.raises(ValueError, match="spending cap"): O.run_opt(O.Opt(trends=T(4), budget=8, backend="hardware", hardware_confirm=True, hardware_max_cost=1))
    monkeypatch.delenv("QUBO_SOLVER_COST_PER_RUN")
    with pytest.raises(ValueError, match="not stated a cost"): O.run_opt(O.Opt(trends=T(4), budget=8, backend="hardware", hardware_confirm=True, hardware_max_cost=10))
    assert Solver.seen == []                                          # nothing was sent while a guard failed

def test_hardware_samples_are_repaired_and_verified_locally(solver):
    r = O.run_opt(O.Opt(trends=T(4), budget=8, backend="hardware", hardware_confirm=True, hardware_max_cost=5)); q = r["quantum"]
    assert q["backend"] == "test-solver" and q["verified"] and q["cost_charged"] == 2.5 and q["ratio"] <= 1.0001 and Solver.seen[0][0] == "Bearer tok" and Solver.seen[0][1]["num_vars"] == 4 + O.slack_bits(8)

def test_hardware_garbage_is_not_trusted(solver, monkeypatch):
    class Bad(Solver):
        def do_POST(self):
            self.rfile.read(int(self.headers["content-length"])); self.send_response(200); self.end_headers(); self.wfile.write(json.dumps({"samples": [{"bits": "1111111111111111"}]}).encode())
    s = HTTPServer(("127.0.0.1", 0), Bad); threading.Thread(target=s.serve_forever, daemon=True).start(); monkeypatch.setenv("QUBO_SOLVER_URL", f"http://127.0.0.1:{s.server_port}/")
    r = O.run_opt(O.Opt(trends=T(4), budget=8, backend="hardware", hardware_confirm=True, hardware_max_cost=5)); s.shutdown()
    assert r["quantum"]["verified"] and r["quantum"]["trends"] is not None            # infeasible all-ones sample was repaired, never reported as is

def test_unreachable_solver_is_reported_not_raised(monkeypatch):
    monkeypatch.setenv("QUBO_SOLVER_URL", "http://127.0.0.1:9/solve"); monkeypatch.setenv("QUBO_SOLVER_COST_PER_RUN", "1")
    r = O.run_opt(O.Opt(trends=T(4), budget=8, backend="hardware", hardware_confirm=True, hardware_max_cost=5)); assert "did not answer" in r["quantum"]["skipped"] and r["recommended"] == "classical exact"

# ---- through the API ----
from fastapi.testclient import TestClient
from app import main
c = TestClient(main.app)
def hd(email):
    v = c.get("/api/v1/terms").json()
    return {"Authorization": "Bearer " + c.post("/api/v1/auth/register", json={"email": email, "password": "password1", "accept_terms": True, "age_confirmed": True, "terms_version": v["terms_version"], "privacy_version": v["privacy_version"]}).json()["token"]}
def test_api_presets_and_tiered_request():
    h = hd("o1@example.com"); p = c.get("/api/v1/optimize/presets", headers=h).json(); assert len(p["tiers"]) == 3 and p["backends"]["hardware"]["available"] is False
    r = c.post("/api/v1/optimize", headers=h, json={"trends": T(6), "budget": 10, "tiers": p["tiers"], "channels": p["channels"][:2], "backend": "anneal"}); assert r.status_code == 200 and r.json()["portfolios"][0]["plan"][0]["tier"]
    assert "Fund" in c.get("/api/v1/reports.json", headers=h).json()[0]["summary"]
def test_api_hardware_errors_are_422_and_schedules_cannot_use_it():
    h = hd("o2@example.com"); r = c.post("/api/v1/optimize", headers=h, json={"backend": "hardware"}); assert r.status_code == 422 and "hardware" in r.text.lower()
    assert c.post("/api/v1/schedules", headers=h, json={"minutes": 60, "params": {"backend": "hardware"}}).status_code == 422
    assert c.post("/api/v1/schedules", headers=h, json={"minutes": 60, "params": {"backend": "anneal"}}).status_code == 200
