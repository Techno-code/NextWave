"""QAOA on Qiskit for the Trend Portfolio Optimizer.

The project already builds an explicit QUBO (app.optimizer.build_qubo) and ships a hand-written NumPy QAOA
simulator. This module takes the *same* QUBO and solves it with real Qiskit objects:

    QUBO  ->  Ising SparsePauliOp  ->  qaoa_ansatz circuit  ->  Aer (ideal or noisy) / IBM Quantum  ->  decode

Everything that comes out is decoded, repaired and re-verified by the project's own classical checker
(Model.repair / Model.ok / Model.check) and compared with the exact branch-and-bound reference, exactly like
the existing backends. Nothing here claims a quantum advantage.

Backends added by register():
    "qiskit"      ideal (or optionally noisy) local simulation with Qiskit Aer. Free, no account.
    "qiskit-ibm"  real IBM Quantum hardware through qiskit-ibm-runtime. Paid: needs an account, a stated cost per run,
                  hardware_confirm and a hardware_max_cost, the same guard rails as the existing "hardware" backend.

Bit order: option a is qubit a, budget slack bit k is qubit nO + k. Qiskit count keys are big-endian strings, so
int(key, 2) gives the same integer mask the rest of the project uses (bit a = option a).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, replace

import numpy as np

from app import optimizer as _opt
from app.optimizer import BackendUnavailable, Model, _result, build_qubo, slack_bits

__all__ = ["CONFIG", "configure", "register", "available", "versions", "qubo_to_ising", "ising_diagonal",
           "presolve", "build_reduced_qubo", "build_cost_operator", "build_circuit", "optimise_qaoa", "sample_counts",
           "run_qiskit", "run_ibm", "export_qasm3", "depolarizing_noise_model", "probabilities_numpy_reference", "clear_cache"]

INSTALL_HINT = "Install the Qiskit extras with: pip install -r qiskit_integration/requirements-qiskit.txt"


# ------------------------------------------------------------------ configuration
@dataclass
class Config:
    reps: int = 3                  # maximum QAOA depth p; depth is grown 1, 2, ... and stops early when it stops helping
    min_reps: int = 1              # never stop before this depth
    grow_tol: float = 0.2          # stop growing depth when CVaR improves by less than this fraction
    presolve: bool = True          # drop options that can never be in an optimal plan, and tighten the budget, before building qubits
    cache: bool = True             # reuse tuned angles for an identical problem (memory; add cache_dir to persist on disk)
    cache_dir: str | None = os.environ.get("QISKIT_CACHE_DIR")
    time_budget_s: float = 60.0    # optimisation stops and keeps its best angles when this is exceeded
    alpha: float = 0.1             # CVaR fraction (1.0 = plain expectation)
    shots: int = 16384             # final sampling shots
    restarts: int = 2              # random restarts at depth 1 (deeper layers start from the interpolated previous optimum)
    maxiter: int = 60              # COBYLA iterations per start at depth 2+ (depth 1 uses half, it only has 2 angles)
    top_k: int = 1000              # distinct sampled bitstrings that are decoded and repaired
    seed: int = 11
    qubit_cap: int = int(os.environ.get("QISKIT_QUBIT_CAP", "20"))   # 2^20 amplitudes = 16 MB per statevector
    noise_model: object = None     # an Aer NoiseModel for the final sampling only (see depolarizing_noise_model)
    ibm_shots: int = 4096
    ibm_optimization_level: int = 3
    extra: dict = field(default_factory=dict)


CONFIG = Config()


def configure(**kw) -> Config:
    """Change defaults for every later run, e.g. configure(reps=3, shots=8192). Returns the new config."""
    global CONFIG
    bad = [k for k in kw if not hasattr(CONFIG, k)]
    if bad:
        raise TypeError(f"Unknown setting(s): {', '.join(bad)}")
    CONFIG = replace(CONFIG, **kw)
    return CONFIG


# ------------------------------------------------------------------ availability
def _import_qiskit():
    try:
        import qiskit  # noqa: F401
        from qiskit.circuit.library import qaoa_ansatz  # noqa: F401
    except Exception as e:  # ImportError, or a Qiskit too old to have qaoa_ansatz
        raise BackendUnavailable(f"Qiskit 2.1 or newer is not installed ({e}). {INSTALL_HINT}")


def _aer():
    try:
        from qiskit_aer import AerSimulator
        return AerSimulator
    except Exception:
        return None


def versions() -> dict:
    out = {}
    for name in ("qiskit", "qiskit_aer", "qiskit_ibm_runtime"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    return out


def available() -> dict:
    v = versions()
    return {"qiskit": {"available": v["qiskit"] is not None, "aer": v["qiskit_aer"] is not None, "max_qubits": CONFIG.qubit_cap,
                       "versions": v},
            "qiskit-ibm": {"available": v["qiskit_ibm_runtime"] is not None and bool(os.environ.get("QISKIT_IBM_COST_PER_RUN")),
                           "cost_per_run": os.environ.get("QISKIT_IBM_COST_PER_RUN")}}


# ------------------------------------------------------------------ QUBO -> Ising
def qubo_to_ising(Q: dict, offset: float, nv: int):
    """x_i = (1 - Z_i) / 2.  Returns (h, J, const) with  E(s) = const + sum h_i s_i + sum J_ij s_i s_j,  s_i = +1 for bit 0, -1 for bit 1."""
    h = np.zeros(nv); J = {}; const = float(offset)
    for (i, j), v in Q.items():
        if i == j:
            const += v / 2; h[i] -= v / 2
        else:
            const += v / 4; h[i] -= v / 4; h[j] -= v / 4
            key = (min(i, j), max(i, j)); J[key] = J.get(key, 0.0) + v / 4
    return h, J, const


def ising_diagonal(h, J, const, nv: int):
    """Energy of every bitstring (index z, bit i = qubit i) from the Ising terms."""
    z = np.arange(1 << nv, dtype=np.int64); s = [1.0 - 2.0 * ((z >> i) & 1) for i in range(nv)]
    E = np.full(1 << nv, const)
    for i in range(nv):
        if h[i]: E += h[i] * s[i]
    for (i, j), v in J.items():
        if v: E += v * s[i] * s[j]
    return E


# ------------------------------------------------------------------ presolve: fewer qubits, same optimum
@dataclass
class Reduced:
    """Which options become qubits, the tightened budget, and what was dropped. active[r] = original option index of qubit r."""
    active: list
    budget: int
    n_slack: int
    removed: dict
    trivial: bool = False            # nothing worth funding, or nothing affordable: the empty plan is optimal / only feasible

    @property
    def nv(self): return len(self.active) + self.n_slack


def presolve(M: Model, enable: bool = True) -> Reduced:
    """Exact reductions (they cannot remove the optimum):
      1. an option with gain <= 0 is never needed: every overlap penalty is >= 0, so dropping it keeps feasibility and never lowers the value;
      2. an option that costs more than the budget can never be chosen;
      3. the budget is capped at the most that max_picks trends could ever spend, which can save slack qubits.
    With enable=False nothing is dropped and the QUBO equals optimizer.build_qubo exactly."""
    B = int(M.b.budget); allo = list(range(M.nO)); removed = {"nonpositive_gain": 0, "over_budget": 0, "budget_tightened_from": None}
    if not enable:
        return Reduced(allo, B, slack_bits(M.b.budget), removed)
    keep = []
    for a in allo:
        o = M.O[a]
        if o["gain"] <= 1e-12: removed["nonpositive_gain"] += 1
        elif o["c"] > M.b.budget: removed["over_budget"] += 1
        else: keep.append(a)
    if not keep or M.b.max_picks <= 0 or M.b.budget < 0:
        return Reduced([], B, 1, removed, trivial=True)
    top = sorted((max(M.O[a]["c"] for a in keep if M.O[a]["trend"] == i) for i in {M.O[a]["trend"] for a in keep}), reverse=True)[: M.b.max_picks]
    cap = int(math.floor(sum(top) + 1e-9))
    if cap < B: removed["budget_tightened_from"] = B; B = cap
    return Reduced(keep, B, slack_bits(max(B, 0)), removed)


def build_reduced_qubo(M: Model, red: Reduced):
    """Same penalty scheme as optimizer.build_qubo, restricted to red.active options and red.budget.
    Variables: the active options in order, then red.n_slack budget slack bits. Returns (Q, offset, nv)."""
    act = red.active; n = len(act); nS = red.n_slack; nv = n + nS; B = red.budget
    rvmax = max(M.O[a]["rv"] for a in act); P_oh = rvmax * 1.5 + 1; Pn = max(rvmax, 1) * .7; Q = {}
    add = lambda i, j, x: Q.__setitem__((min(i, j), max(i, j)), Q.get((min(i, j), max(i, j)), 0.0) + x)
    for r, a in enumerate(act):
        add(r, r, -M.O[a]["gain"])
        for r2 in range(r + 1, n):
            a2 = act[r2]; add(r, r2, P_oh if M.O[a2]["trend"] == M.O[a]["trend"] else M.S[a][a2])
    coef = [M.O[a]["c"] for a in act] + [2 ** k for k in range(nS)]
    for u in range(nv):
        add(u, u, Pn * (coef[u] ** 2 - 2 * B * coef[u]))
        for v in range(u + 1, nv): add(u, v, Pn * 2 * coef[u] * coef[v])
    return Q, Pn * B * B, nv


def _full_mask(red: Reduced, z: int) -> int:
    """Reduced bitstring (qubit r = option red.active[r]) -> mask over all original options."""
    m = 0
    for r, a in enumerate(red.active):
        if z >> r & 1: m |= 1 << a
    return m


def build_cost_operator(M: Model, normalise: bool = True, red: Reduced | None = None):
    """Returns (SparsePauliOp H_C, info). red=None means the full, un-reduced problem (identical to optimizer.build_qubo).
    With normalise=True H_C is scaled so its spectrum spans width 1, which makes one gamma range suit every problem."""
    from qiskit.quantum_info import SparsePauliOp
    red = red or presolve(M, False); Q, off, nv = build_reduced_qubo(M, red); h, J, const = qubo_to_ising(Q, off, nv)
    E = ising_diagonal(h, J, const, nv); emin, emax = float(E.min()), float(E.max()); span = (emax - emin) or 1.0
    terms = [("Z", [i], float(h[i])) for i in range(nv) if h[i]] + [("ZZ", [i, j], float(v)) for (i, j), v in sorted(J.items()) if v]
    H = SparsePauliOp.from_sparse_list(terms, num_qubits=nv) if terms else SparsePauliOp("I" * nv, 0.0)
    if normalise: H = (H / span).simplify()
    return H, {"num_qubits": nv, "const": const, "emin": emin, "emax": emax, "span": span, "diag": E, "h": h, "J": J, "Q": Q, "offset": off, "reduced": red}


def _native_circuit(info, reps: int):
    """Hadamards, then reps x [cost layer, X mixer], built directly from RZ / RZZ / RX gates.
    Equivalent to qiskit.circuit.library.qaoa_ansatz (tested) but ~100x cheaper to construct, and Aer runs it without transpiling."""
    from qiskit.circuit import ParameterVector, QuantumCircuit
    nv, span = info["num_qubits"], info["span"]; beta = ParameterVector("β", reps); gamma = ParameterVector("γ", reps)
    qc = QuantumCircuit(nv, name=f"QAOA_p{reps}"); qc.h(range(nv))
    for l in range(reps):
        for i in range(nv):
            if info["h"][i]: qc.rz(2 * gamma[l] * float(info["h"][i] / span), i)
        for (i, j), v in sorted(info["J"].items()):
            if v: qc.rzz(2 * gamma[l] * float(v / span), i, j)
        qc.rx(2 * beta[l], range(nv))
    return qc


def build_circuit(M: Model, reps: int | None = None, measure: bool = False, library_ansatz: bool = False, presolved: bool = True):
    """The parametrised QAOA circuit for this request: (circuit, cost operator, info). Parameters are beta[0..p-1], gamma[0..p-1].
    presolved=True builds the (smaller) presolved problem the backend actually runs; library_ansatz=True uses qiskit's qaoa_ansatz instead of the native builder."""
    _import_qiskit()
    red = presolve(M, CONFIG.presolve and presolved); H, info = build_cost_operator(M, True, red); p = CONFIG.reps if reps is None else reps
    if library_ansatz:
        from qiskit.circuit.library import qaoa_ansatz
        qc = qaoa_ansatz(H, reps=p, name=f"QAOA_p{p}")
    else:
        qc = _native_circuit(info, p)
    if measure: qc.measure_all()
    return qc, H, info


def export_qasm3(M: Model, params=None, reps: int | None = None) -> str:
    """OpenQASM 3 text of the (optionally parameter-bound) circuit, for use outside this project."""
    from qiskit import qasm3
    qc, _, _ = build_circuit(M, reps)
    if params is not None: qc = _bind(qc, params)
    return qasm3.dumps(qc)


def _bind(qc, x):
    """x = [beta_0..beta_{p-1}, gamma_0..gamma_{p-1}] (the ansatz order)."""
    named = {p.name: p for p in qc.parameters}; p_ = len(x) // 2
    m = {named[f"β[{k}]"]: float(x[k]) for k in range(p_)}; m.update({named[f"γ[{k}]"]: float(x[p_ + k]) for k in range(p_)})
    return qc.assign_parameters(m)


# ------------------------------------------------------------------ simulation engine
class _Engine:
    """Exact output probabilities of a QAOA circuit. Uses Aer when installed (the native circuit needs no transpile), else qiskit.quantum_info."""

    def __init__(self, qc, seed):
        self.qc = qc; self.names = {p.name: p for p in qc.parameters}; self.aer = None
        Aer = _aer()
        if Aer is not None:
            self.aer = Aer(method="statevector", seed_simulator=seed)
            c = qc.copy(); c.save_probabilities(); self.tc = c

    def probs(self, x) -> np.ndarray:
        p_ = len(x) // 2
        if self.aer is not None:
            binds = {self.names[f"β[{k}]"]: [float(x[k])] for k in range(p_)}; binds.update({self.names[f"γ[{k}]"]: [float(x[p_ + k])] for k in range(p_)})
            pr = np.asarray(self.aer.run(self.tc, parameter_binds=[binds]).result().data(0)["probabilities"], dtype=float)
        else:
            from qiskit.quantum_info import Statevector
            pr = Statevector(_bind(self.qc, x)).probabilities()
        return pr / pr.sum()


def _cvar(pr, E, alpha):
    """Mean of the lowest alpha-fraction of probability mass of E. alpha=1 is the plain expectation."""
    if alpha >= 1.0: return float(pr @ E)
    o = np.argsort(E, kind="stable"); return _cvar_sorted(pr[o], E[o], alpha)


def _cvar_sorted(p_sorted, E_sorted, alpha):
    cum = np.cumsum(p_sorted); w = np.minimum(p_sorted, np.maximum(alpha - (cum - p_sorted), 0.0))
    return float((w * E_sorted).sum() / alpha)


class _Stop(Exception): pass


def _interp(x, p):
    """Depth p -> p+1 starting angles by linear interpolation of the optimal schedule (Zhou et al., INTERP)."""
    b, g = list(x[:p]), list(x[p:]); out = []
    for v in (b, g):
        nv_ = [(i / p) * (v[i - 1] if i >= 1 else 0.0) + ((p - i) / p) * (v[i] if i < p else 0.0) for i in range(p + 1)]
        out.append(nv_)
    return out[0] + out[1]


# ------------------------------------------------------------------ parameter cache
_CACHE: dict = {}


def clear_cache(disk: bool = False):
    _CACHE.clear()
    if disk and CONFIG.cache_dir:
        try: os.remove(os.path.join(CONFIG.cache_dir, "qaoa_params.json"))
        except OSError: pass


def _cache_key(info, c, alpha):
    h = hashlib.sha1()
    h.update(np.round(info["h"] / info["span"], 10).tobytes()); h.update(repr(sorted((k, round(float(v / info["span"]), 10)) for k, v in info["J"].items())).encode())
    h.update(repr((c.reps, c.min_reps, c.grow_tol, alpha, c.restarts, c.maxiter, c.seed)).encode()); return h.hexdigest()


def _cache_get(key):
    if key in _CACHE: return _CACHE[key]
    if CONFIG.cache_dir:
        try:
            with open(os.path.join(CONFIG.cache_dir, "qaoa_params.json")) as f: d = json.load(f)
            if key in d: _CACHE[key] = d[key]; return d[key]
        except (OSError, ValueError): pass
    return None


def _cache_put(key, val):
    _CACHE[key] = val
    if CONFIG.cache_dir:
        try:
            os.makedirs(CONFIG.cache_dir, exist_ok=True); path = os.path.join(CONFIG.cache_dir, "qaoa_params.json")
            try:
                with open(path) as f: d = json.load(f)
            except (OSError, ValueError): d = {}
            d[key] = val
            with open(path + ".tmp", "w") as f: json.dump(d, f)
            os.replace(path + ".tmp", path)
        except OSError: pass


def optimise_qaoa(M: Model, reps=None, alpha=None, restarts=None, maxiter=None, seed=None, time_budget_s=None, use_cache=None):
    """Tunes the 2p QAOA angles with COBYLA on the exact (noise-free) CVaR of the energy distribution, growing the depth 1, 2, ... p_max.
    Depth 1 starts from a linear ramp plus random restarts; each deeper layer starts from the interpolated previous optimum, which is
    far cheaper than restarting. Growth stops early when the extra layer improves CVaR by less than grow_tol. Returns a dict."""
    _import_qiskit()
    from scipy.optimize import minimize
    c = CONFIG; pmax = c.reps if reps is None else reps; alpha = c.alpha if alpha is None else alpha
    restarts = c.restarts if restarts is None else restarts; maxiter = c.maxiter if maxiter is None else maxiter; seed = c.seed if seed is None else seed
    budget = c.time_budget_s if time_budget_s is None else time_budget_s; use_cache = c.cache if use_cache is None else use_cache
    red = presolve(M, c.presolve); H, info = build_cost_operator(M, True, red); nq = info["num_qubits"]
    if nq > c.qubit_cap: raise ValueError(f"{nq} qubits exceed the cap of {c.qubit_cap}")
    En = (info["diag"] - info["emin"]) / info["span"]; order = np.argsort(En, kind="stable"); Eo = En[order]
    t_start = time.time(); rng = random.Random(seed); ckey = _cache_key(info, replace(c, reps=pmax, restarts=restarts, maxiter=maxiter, seed=seed), alpha)
    hit = _cache_get(ckey) if use_cache else None
    if hit:   # identical problem and settings: one evaluation instead of a full optimisation
        p_ = len(hit["params"]) // 2; qc = _native_circuit(info, p_); eng = _Engine(qc, seed); pr = eng.probs(hit["params"])
        return {"params": list(hit["params"]), "cvar": _cvar_sorted(pr[order], Eo, alpha), "cvar_start": hit["cvar_start"], "evals": 1, "reps": p_, "alpha": alpha,
                "engine": "aer" if eng.aer is not None else "quantum_info", "probs": pr, "info": info, "circuit": qc, "reduced": red, "convergence": hit["convergence"],
                "cache_hit": True, "stopped_by_time": False}
    evals = [0]; convergence = []; stopped = False; best = None   # best = (cvar, x, reps, engine)
    prev_x = None; prev_cvar = None; start_val = None
    for p in range(1, pmax + 1):
        qc = _native_circuit(info, p); eng = _Engine(qc, seed); e0 = evals[0]; t_p = time.time()

        def f(x):
            if time.time() - t_start > budget and best is not None: raise _Stop
            evals[0] += 1; return _cvar_sorted(eng.probs(x)[order], Eo, alpha)

        if p == 1: starts = [[.8 * .5, 2 * math.pi * .35 * .5]] + [[rng.random() * 1.6, rng.random() * 2 * math.pi] for _ in range(restarts)]
        else: starts = [_interp(prev_x, p - 1)]
        px, pv = None, 1e18
        try:
            for x0 in starts:
                v0 = f(x0)
                if start_val is None: start_val = v0
                if v0 < pv: px, pv = list(x0), v0
                r = minimize(f, x0, method="COBYLA", options={"maxiter": max(maxiter if p > 1 else maxiter // 2, 2 * p + 2), "rhobeg": .3, "tol": 2e-3})
                if r.fun < pv: px, pv = [float(v) for v in r.x], float(r.fun)
        except _Stop:
            stopped = True
        convergence.append({"reps": p, "cvar": round(pv, 6), "evals": evals[0] - e0, "ms": round((time.time() - t_p) * 1000)})
        if px is not None and (best is None or pv < best[0] - 1e-12): best = (pv, px, p, eng)
        if stopped or px is None: break
        improved = prev_cvar is None or (prev_cvar - pv) > c.grow_tol * max(prev_cvar, 1e-9)
        prev_x, prev_cvar = px, pv
        if p >= c.min_reps and not improved: break
    pv, x, p_, eng = best; pr = eng.probs(x)
    out = {"params": x, "cvar": pv, "cvar_start": float(start_val), "evals": evals[0], "reps": p_, "alpha": alpha, "engine": "aer" if eng.aer is not None else "quantum_info",
           "probs": pr, "info": info, "circuit": eng.qc, "reduced": red, "convergence": convergence, "cache_hit": False, "stopped_by_time": stopped}
    if use_cache and not stopped: _cache_put(ckey, {"params": x, "cvar_start": float(start_val), "convergence": convergence})
    return out


# ------------------------------------------------------------------ sampling
def depolarizing_noise_model(p1=0.001, p2=0.01, readout=0.02):
    """A simple Aer noise model: depolarising error on 1- and 2-qubit gates plus symmetric readout error."""
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    nm = NoiseModel(basis_gates=["rz", "sx", "x", "cx", "id"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["rz", "sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    if readout: nm.add_all_qubit_readout_error(ReadoutError([[1 - readout, readout], [readout, 1 - readout]]))
    return nm


def sample_counts(qc, params, shots, seed, noise_model=None) -> dict:
    """Runs the bound circuit with measurements and returns {int mask over the circuit's qubits: count}."""
    from qiskit import transpile
    bound = _bind(qc, params); bound.measure_all(); Aer = _aer()
    if Aer is None:
        if noise_model is not None: raise BackendUnavailable("Noisy simulation needs qiskit-aer. " + INSTALL_HINT)
        from qiskit.primitives import StatevectorSampler
        res = StatevectorSampler(default_shots=shots, seed=seed).run([bound]).result()[0]
        return {int(k, 2): v for k, v in res.data.meas.get_counts().items()}
    if noise_model is not None:
        sim = Aer(method="automatic", noise_model=noise_model, seed_simulator=seed)
        tc = transpile(bound, basis_gates=noise_model.basis_gates, optimization_level=1, seed_transpiler=seed)
    else:
        sim = Aer(method="statevector", seed_simulator=seed); tc = bound    # RZ / RZZ / RX / H are native to Aer
    counts = sim.run(tc, shots=shots, seed_simulator=seed).result().get_counts()
    return {int(k.replace(" ", ""), 2): v for k, v in counts.items()}


def _decode(M: Model, red: Reduced, counts: dict, top_k: int, ef: float | None = None):
    """Counts over the circuit's qubits -> (best repaired assignment, raw feasibility, shots, share of shots whose repaired plan reaches the exact optimum)."""
    total = sum(counts.values()); feas = 0; merged = {}
    for z, c_ in counts.items():
        m = _full_mask(red, z); merged[m] = merged.get(m, 0) + c_
        if M.mask_ok(m): feas += c_
    bestA = None; hit = 0; seen = 0
    for m_, c_ in sorted(merged.items(), key=lambda kv: -kv[1])[:top_k]:
        A = M.repair(M.from_mask(m_)); fa = M.f(A); seen += c_
        if ef is not None and fa >= ef - 1e-9: hit += c_
        if bestA is None or fa > M.f(bestA): bestA = A
    if bestA is None: bestA = [-1] * M.n
    return bestA, (feas / total if total else 0.0), total, (hit / total if total and ef is not None else None)


def _too_big(M: Model, red: Reduced, backend: str):
    kept_nonint = any(not float(M.O[a]["c"]).is_integer() for a in red.active)
    if red.nv > CONFIG.qubit_cap or kept_nonint:
        return {"skipped": f"The Qiskit backend needs integer costs and at most {CONFIG.qubit_cap} qubits (options plus budget bits); this request needs {red.nv}"
                           f" even after presolve. Use the anneal backend for larger problems.", "backend": backend}
    return None


def _label(extra=""):
    v = versions(); return f"Qiskit {v['qiskit']}" + (f" + Aer {v['qiskit_aer']}" if v["qiskit_aer"] else "") + extra


def _presolve_info(M, red):
    return {"qubits_full": M.nO + slack_bits(M.b.budget), "qubits_used": red.nv, "options_dropped": M.nO - len(red.active), **{k: v for k, v in red.removed.items() if v}}


def _trivial(M, ef, t0, backend):
    return _result(M, [-1] * M.n, ef, t0, qubits=0, shots=0, depth=0, backend=backend, note="Presolve proved the empty plan is the only sensible one (nothing affordable has positive gain).")


# ------------------------------------------------------------------ backend: Aer
def run_qiskit(M: Model, ef: float):
    """Backend 'qiskit'. Same contract as optimizer.run_simulator: returns a verified result dict, or {'skipped': ...}."""
    _import_qiskit(); c = CONFIG; red = presolve(M, c.presolve)
    if red.trivial: return _trivial(M, ef, time.time(), "Qiskit (presolved)")
    sk = _too_big(M, red, "Qiskit Aer simulator")
    if sk: return sk
    t0 = time.time(); opt = optimise_qaoa(M); qc = opt["circuit"]
    counts = sample_counts(qc, opt["params"], c.shots, c.seed, c.noise_model)
    A, feas, shots, p_opt = _decode(M, red, counts, c.top_k, ef)
    from qiskit import transpile
    dc = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=c.seed); ops = dc.count_ops()
    return _result(M, A, ef, t0, qubits=opt["info"]["num_qubits"], shots=shots, depth=dc.depth(), two_qubit_gates=int(ops.get("cx", 0)),
                   reps=opt["reps"], raw_feasibility=round(feas, 3), share_of_shots_reaching_optimum=None if p_opt is None else round(p_opt, 3),
                   cvar_start=round(opt["cvar_start"], 4), cvar_end=round(opt["cvar"], 4), optimiser_evaluations=opt["evals"], convergence=opt["convergence"],
                   cache_hit=opt["cache_hit"], presolve=_presolve_info(M, red), params=[round(v, 4) for v in opt["params"]],
                   backend=f"{_label()}, QAOA p={opt['reps']} (grown from 1), CVaR {opt['alpha']:g}, " + ("noisy sampling" if c.noise_model else "ideal sampling"))


# ------------------------------------------------------------------ backend: IBM Quantum
def run_ibm(M: Model, ef: float, ibm_backend=None):
    """Backend 'qiskit-ibm'. Angles are tuned on the ideal simulator (free), then ONE sampling job is sent to the device.
    ibm_backend lets a test or a user pass a backend object (a real one, or a fake device for local noisy runs)."""
    _import_qiskit()
    try:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except Exception as e:
        raise BackendUnavailable(f"qiskit-ibm-runtime is not installed ({e}). {INSTALL_HINT}")
    real = ibm_backend is None
    if real:
        cost_s = os.environ.get("QISKIT_IBM_COST_PER_RUN")
        if cost_s is None: raise BackendUnavailable("The operator has not stated a cost per run (QISKIT_IBM_COST_PER_RUN), so paid compute cannot be used.")
        cost = float(cost_s)
        if not M.b.hardware_confirm: raise BackendUnavailable(f"This backend costs {cost:g} per run. Set hardware_confirm to true to confirm.")
        if cost > M.b.hardware_max_cost: raise BackendUnavailable(f"This backend costs {cost:g} per run, above your spending cap of {M.b.hardware_max_cost:g}.")
    c = CONFIG; red = presolve(M, c.presolve)
    if red.trivial: return _trivial(M, ef, time.time(), "IBM Quantum (presolved, no job sent)")
    sk = _too_big(M, red, "IBM Quantum")
    if sk: return sk
    t0 = time.time(); opt = optimise_qaoa(M); nq = opt["info"]["num_qubits"]
    try:
        if real:
            kw = {k: os.environ[e] for k, e in (("token", "QISKIT_IBM_TOKEN"), ("instance", "QISKIT_IBM_INSTANCE"), ("channel", "QISKIT_IBM_CHANNEL")) if os.environ.get(e)}
            svc = QiskitRuntimeService(**kw); name = os.environ.get("QISKIT_IBM_BACKEND")
            ibm_backend = svc.backend(name) if name else svc.least_busy(operational=True, simulator=False, min_num_qubits=nq)
        bound = _bind(opt["circuit"], opt["params"]); bound.measure_all()
        isa = generate_preset_pass_manager(backend=ibm_backend, optimization_level=c.ibm_optimization_level, seed_transpiler=c.seed).run(bound)
        res = SamplerV2(mode=ibm_backend).run([isa], shots=c.ibm_shots).result()[0]
        counts = {int(k, 2): v for k, v in res.data.meas.get_counts().items()}
    except Exception as e:
        return {"skipped": f"The IBM Quantum job did not complete: {e}", "backend": "IBM Quantum"}
    A, feas, shots, p_opt = _decode(M, red, counts, c.top_k, ef); ops = isa.count_ops()
    two_q = int(sum(v for k, v in ops.items() if k in ("cx", "cz", "ecr", "rzz", "swap")))
    ex = {"cost_charged": float(os.environ["QISKIT_IBM_COST_PER_RUN"])} if real else {}
    dev = getattr(ibm_backend, "name", str(ibm_backend))
    return _result(M, A, ef, t0, qubits=nq, shots=shots, depth=isa.depth(), two_qubit_gates=two_q, reps=opt["reps"], raw_feasibility=round(feas, 3),
                   share_of_shots_reaching_optimum=None if p_opt is None else round(p_opt, 3), cvar_start=round(opt["cvar_start"], 4), cvar_end=round(opt["cvar"], 4),
                   convergence=opt["convergence"], presolve=_presolve_info(M, red), params=[round(v, 4) for v in opt["params"]], device=dev, **ex,
                   backend=f"{_label(f' + qiskit-ibm-runtime {versions()['qiskit_ibm_runtime']}')}, QAOA p={opt['reps']}, device {dev}")


# ------------------------------------------------------------------ cross-check against the project's NumPy simulator
def probabilities_numpy_reference(M: Model, params) -> np.ndarray:
    """The project's own NumPy QAOA (same maths as optimizer.run_simulator.run) on the FULL un-reduced problem, for params = [beta..., gamma...]
    in the Qiskit convention. Used to prove the Qiskit circuit and the existing simulator agree."""
    nq = M.nO + slack_bits(M.b.budget); N = 1 << nq; E = _opt.energies(M, np.arange(N), slack_bits(M.b.budget))
    En = (E - E.min()) / ((E.max() - E.min()) or 1); p_ = len(params) // 2; psi = np.full(N, 1 / math.sqrt(N), dtype=complex)
    for l in range(p_):
        beta, gamma = params[l], params[p_ + l]
        psi *= np.exp(-1j * gamma * En); cb, sb = math.cos(beta), math.sin(beta)
        for kq in range(nq):
            vw = psi.reshape(-1, 2, 1 << kq); a = vw[:, 0, :].copy(); b = vw[:, 1, :].copy()
            vw[:, 0, :] = cb * a - 1j * sb * b; vw[:, 1, :] = cb * b - 1j * sb * a
    return np.abs(psi) ** 2


# ------------------------------------------------------------------ registration
def register():
    """Adds 'qiskit' and 'qiskit-ibm' to optimizer.BACKENDS / RUNNERS so /api/v1/optimize accepts them. Safe to call twice."""
    _opt.RUNNERS["qiskit"] = run_qiskit; _opt.RUNNERS["qiskit-ibm"] = run_ibm
    _opt.BACKENDS = tuple(dict.fromkeys(tuple(_opt.BACKENDS) + ("qiskit", "qiskit-ibm")))
    return _opt.BACKENDS
