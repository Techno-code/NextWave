"""Tests for the Qiskit integration. Run from the project root:  python -m pytest qiskit_integration/tests -q
Tests that need Qiskit are skipped (not failed) when it is not installed."""
import os, sys, random, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
from app import optimizer as O
from app.optimizer import Model, Opt, exact, build_qubo, qubo_energy
import qiskit_integration as qi


@pytest.fixture(autouse=True)
def _fresh_config():
    old = qi.CONFIG; qi.clear_cache(); yield; qi.CONFIG = old; qi.qiskit_qaoa.CONFIG = old; qi.clear_cache()


def small(seed=7, budget=8, n=6):
    T = [{"name": f"t{i}", "cat": i % 3, "p": .4 + .05 * i, "E": 30 + 3 * i, "hl": 12, "c": 2 + (i % 3)} for i in range(n)]
    return Model(Opt(budget=budget, trends=T, seed=seed, max_picks=4))


def test_ising_matches_qubo_on_every_bitstring():
    M = small(); Q, off, nv = build_qubo(M); H, info = qi.build_cost_operator(M, normalise=False)
    for z in random.Random(1).sample(range(1 << nv), 200):
        assert info["diag"][z] == pytest.approx(qubo_energy(Q, off, [(z >> i) & 1 for i in range(nv)]), abs=1e-6)


def test_pauli_operator_diagonal_matches_numpy_diagonal():
    """The SparsePauliOp itself (not just my numpy shortcut) has the same diagonal."""
    M = small(); H, info = qi.build_cost_operator(M, normalise=False)
    diag = np.real(np.diag(H.to_matrix(sparse=True).toarray())) + info["const"]
    assert np.allclose(diag, info["diag"], atol=1e-8)


def test_qiskit_circuit_equals_project_numpy_simulator():
    """Same angles through Qiskit and through optimizer.run_simulator's maths give the same output distribution."""
    M = small(); qc, H, info = qi.build_circuit(M, reps=2, presolved=False)
    x = [.31, .77, 1.9, 4.2]
    eng = qi.qiskit_qaoa._Engine(qc, 1)
    assert np.abs(eng.probs(x) - qi.probabilities_numpy_reference(M, x)).max() < 1e-9


def test_statevector_fallback_agrees_with_aer():
    from qiskit.quantum_info import Statevector
    M = small(); qc, _, _ = qi.build_circuit(M, reps=1, presolved=False); x = [.4, 1.1]
    a = qi.qiskit_qaoa._Engine(qc, 1).probs(x); b = Statevector(qi.qiskit_qaoa._bind(qc, x)).probabilities()
    assert np.abs(a - b).max() < 1e-9


def test_cvar_alpha_one_is_expectation():
    pr = np.array([.5, .3, .2]); E = np.array([3., 1., 2.])
    assert qi.qiskit_qaoa._cvar(pr, E, 1.0) == pytest.approx(pr @ E)
    assert qi.qiskit_qaoa._cvar(pr, E, .3) == pytest.approx(1.0)


def test_run_qiskit_result_is_verified_and_shaped_like_other_backends():
    M = small(); ef = exact(M)[0][0][0]; qi.configure(restarts=1, maxiter=40, shots=4096)
    r = qi.run_qiskit(M, ef)
    assert r["verified"] is True and 0 < r["ratio"] <= 1.0001
    for k in ("objective", "ratio", "trends", "plan", "ms", "qubits", "shots", "depth", "raw_feasibility", "backend"):
        assert k in r
    assert r["qubits"] <= M.nO + O.slack_bits(M.b.budget) and "Qiskit" in r["backend"]


def test_deterministic_with_same_seed():
    M = small(); ef = exact(M)[0][0][0]; qi.configure(restarts=1, maxiter=30, shots=2048)
    a = qi.run_qiskit(M, ef); b = qi.run_qiskit(M, ef)
    assert (a["objective"], a["params"], a["raw_feasibility"]) == (b["objective"], b["params"], b["raw_feasibility"])


def test_too_many_qubits_is_skipped_not_failed():
    M = Model(Opt()); qi.configure(qubit_cap=8)
    r = qi.run_qiskit(M, 1.0)
    assert "skipped" in r and "qubits" in r["skipped"]


def test_non_integer_cost_is_skipped():
    T = [{"name": "a", "cat": 0, "p": .6, "E": 30, "hl": 10, "c": 2.5}, {"name": "b", "cat": 1, "p": .6, "E": 30, "hl": 10, "c": 3}]
    M = Model(Opt(budget=6, trends=T)); assert "skipped" in qi.run_qiskit(M, 1.0)


def test_noisy_sampling_still_returns_verified_plan():
    pytest.importorskip("qiskit_aer"); M = small(n=4, budget=6); ef = exact(M)[0][0][0]
    qi.configure(restarts=1, maxiter=30, shots=2048, noise_model=qi.depolarizing_noise_model(.002, .02, .02))
    r = qi.run_qiskit(M, ef)
    assert r["verified"] and "noisy" in r["backend"]


def test_register_exposes_backends_through_run_opt():
    saved = (dict(O.RUNNERS), O.BACKENDS)
    try:
        qi.register(); qi.register()
        assert O.BACKENDS.count("qiskit") == 1
        qi.configure(restarts=1, maxiter=30, shots=2048)
        T = [{"name": f"t{i}", "cat": i % 2, "p": .5 + .05 * i, "E": 30, "hl": 12, "c": 2 + i % 2} for i in range(5)]
        r = O.run_opt(Opt(budget=7, trends=T, backend="qiskit", compare_backends=True))
        assert r["quantum"]["verified"] and "Qiskit" in r["quantum"]["backend"]
        assert set(r["backends"]) == {"qiskit", "simulator", "anneal"}
        assert r["classical"]["proven_optimal"] is True
    finally:
        O.RUNNERS.clear(); O.RUNNERS.update(saved[0]); O.BACKENDS = saved[1]


def test_ibm_backend_refuses_without_cost_confirmation(monkeypatch):
    pytest.importorskip("qiskit_ibm_runtime"); M = small(n=4, budget=6)
    monkeypatch.delenv("QISKIT_IBM_COST_PER_RUN", raising=False)
    with pytest.raises(O.BackendUnavailable, match="cost per run"): qi.run_ibm(M, 1.0)
    monkeypatch.setenv("QISKIT_IBM_COST_PER_RUN", "5")
    with pytest.raises(O.BackendUnavailable, match="hardware_confirm"): qi.run_ibm(M, 1.0)
    M.b.hardware_confirm = True; M.b.hardware_max_cost = 1
    with pytest.raises(O.BackendUnavailable, match="spending cap"): qi.run_ibm(M, 1.0)


def test_ibm_path_runs_on_a_fake_device_locally():
    """Exercises transpile-to-ISA + SamplerV2 on a fake IBM device (local noisy simulation, no account, no cost)."""
    pytest.importorskip("qiskit_ibm_runtime"); pytest.importorskip("qiskit_aer")
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    M = small(n=3, budget=4); ef = exact(M)[0][0][0]; qi.configure(restarts=0, maxiter=20, ibm_shots=1024, ibm_optimization_level=1)
    r = qi.run_ibm(M, ef, ibm_backend=FakeSherbrooke())
    assert "skipped" not in r, r
    assert r["verified"] and r["device"] == "fake_sherbrooke" and r["two_qubit_gates"] > 0


def test_qasm3_export_is_text():
    M = small(n=3, budget=4); s = qi.export_qasm3(M, params=[.3, .4, 1.0, 2.0])
    assert s.startswith("OPENQASM 3")


# ---------------------------------------------------------------- v2: native circuit, presolve, depth growth, cache
def test_native_circuit_equals_qiskit_library_ansatz():
    M = small(); x = [.31, .77, 1.9, 4.2]
    a, _, _ = qi.build_circuit(M, reps=2); b, _, _ = qi.build_circuit(M, reps=2, library_ansatz=True)
    pa = qi.qiskit_qaoa._Engine(a, 1).probs(x); pb = qi.qiskit_qaoa._Engine(b, 1).probs(x)
    assert np.abs(pa - pb).max() < 1e-9


def test_unreduced_qubo_is_identical_to_project_build_qubo():
    M = small(); Q0, off0, nv0 = build_qubo(M); Q1, off1, nv1 = qi.build_reduced_qubo(M, qi.presolve(M, enable=False))
    assert nv0 == nv1 and off0 == pytest.approx(off1) and Q0.keys() == Q1.keys()
    assert all(Q0[k] == pytest.approx(Q1[k]) for k in Q0)


def _hard_model():
    """Trends with a negative-gain option, an unaffordable option and a budget larger than max_picks can ever spend."""
    T = [{"name": "good1", "cat": 0, "p": .8, "E": 40, "hl": 20, "c": 3}, {"name": "good2", "cat": 1, "p": .7, "E": 35, "hl": 20, "c": 2},
         {"name": "dud", "cat": 2, "p": .05, "E": 5, "hl": 20, "c": 4}, {"name": "pricey", "cat": 3, "p": .9, "E": 50, "hl": 20, "c": 40}]
    return Model(Opt(budget=30, trends=T, max_picks=2, lam_waste=3))


def test_presolve_drops_options_and_tightens_budget_without_losing_the_optimum():
    M = _hard_model(); red = qi.presolve(M)
    assert red.removed["over_budget"] >= 1 and red.nv < M.nO + O.slack_bits(M.b.budget)
    assert red.budget <= 7 and red.removed["budget_tightened_from"] == 30     # two picks can spend at most 3 + 2 here
    ef, eA = exact(M)[0][0]
    assert all(a in red.active or a == -1 for a in eA)                          # the exact optimum only uses kept options
    qi.configure(restarts=1, maxiter=30, shots=4096); r = qi.run_qiskit(M, ef)
    assert r["verified"] and r["ratio"] == pytest.approx(1.0, abs=1e-6) and r["presolve"]["qubits_used"] < r["presolve"]["qubits_full"]


def test_presolve_never_changes_exact_optimum_on_random_models():
    rng = random.Random(5)
    for _ in range(25):
        n = rng.randint(3, 6)
        T = [{"name": f"t{i}", "cat": rng.randint(0, 2), "p": rng.random(), "E": rng.randint(5, 60), "hl": 15, "c": rng.randint(1, 6)} for i in range(n)]
        M = Model(Opt(budget=rng.randint(3, 14), trends=T, max_picks=rng.randint(1, 4), lam_waste=rng.random() * 4, seed=rng.randint(1, 99)))
        full = exact(M)[0]
        if not full: continue
        red = qi.presolve(M); ef, eA = full[0]
        assert all(a in red.active for a in eA if a >= 0), (ef, eA, red)
        assert red.budget >= M.cost(eA)


def test_trivial_problem_returns_empty_verified_plan():
    T = [{"name": "dud", "cat": 0, "p": .01, "E": 1, "hl": 5, "c": 3}]
    M = Model(Opt(budget=5, trends=T, lam_waste=5)); r = qi.run_qiskit(M, 0.0)
    assert r["verified"] and r["trends"] == [] and r["qubits"] == 0


def test_depth_growth_records_convergence_and_never_gets_worse():
    M = small(); qi.configure(reps=3, restarts=1, maxiter=30, grow_tol=-1.0)       # grow_tol < 0 forces every depth to run
    o = qi.optimise_qaoa(M)
    assert [c["reps"] for c in o["convergence"]] == [1, 2, 3]
    assert o["cvar"] <= min(c["cvar"] for c in o["convergence"]) + 1e-9


def test_interp_warm_start_preserves_a_constant_schedule_and_adds_a_layer():
    x = qi.qiskit_qaoa._interp([.5, .5, .3, .3], 2)
    assert x == pytest.approx([.5, .5, .5, .3, .3, .3])


def test_cache_hit_is_one_evaluation_and_same_answer():
    M = small(); ef = exact(M)[0][0][0]; qi.configure(restarts=1, maxiter=30, shots=2048)
    a = qi.run_qiskit(M, ef); b = qi.run_qiskit(M, ef)
    assert a["cache_hit"] is False and b["cache_hit"] is True and b["optimiser_evaluations"] == 1
    assert a["params"] == b["params"] and a["objective"] == b["objective"]
    qi.configure(cache=False); assert qi.run_qiskit(M, ef)["cache_hit"] is False


def test_cache_persists_to_disk(tmp_path):
    M = small(); qi.configure(restarts=1, maxiter=20, cache_dir=str(tmp_path)); a = qi.optimise_qaoa(M)
    qi.clear_cache(); b = qi.optimise_qaoa(M)        # memory cleared: must come back from the JSON file
    assert b["cache_hit"] and b["params"] == a["params"] and (tmp_path / "qaoa_params.json").exists()


def test_time_budget_stops_cleanly_and_keeps_best_angles():
    M = small(); qi.configure(restarts=3, maxiter=200, cache=False); o = qi.optimise_qaoa(M, time_budget_s=0.01)
    assert o["stopped_by_time"] and len(o["params"]) == 2 * o["reps"] and o["evals"] >= 1


def test_share_of_shots_reaching_optimum_is_reported():
    M = small(); ef = exact(M)[0][0][0]; qi.configure(restarts=1, maxiter=30, shots=2048)
    r = qi.run_qiskit(M, ef); assert 0.0 <= r["share_of_shots_reaching_optimum"] <= 1.0
