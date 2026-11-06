"""One pass from a described problem to a verified decision.

Steps that run side by side (a thread pool):
  classify   trained problem-type model (injected by the caller)
  retrieve   nearest problems in the 1M catalogue, which also votes for a family
  parse      pull options, costs and a budget out of the text, when they are there
Then, in order: choose the family, fetch its playbook, solve the instance with the reference solver, re-derive the answer a second,
independent way, verify it against the raw instance, and (for small budgeted selections) run the QAOA simulator as a third check.

Numbers only ever come from the person's instance. When none is supplied the result is a formulation and a playbook, or, on request,
a solve of a generated illustrative instance that is labelled as such everywhere it appears.
"""
import itertools, math, random, threading, time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, minimize
from . import bigdata, ref_solvers as R
from .solvers import parse

FAMILIES = bigdata.FAMILIES
SOLVE_TIMEOUT = 30
_TL = threading.local()   # fast mode: what-if re-solves skip the second-method cross-checks (the headline answer always gets them)
DP_CELLS = 8_000_000   # table size (items x capacity) up to which the exact table method is used; above it the MIP solver is used
MIP_SECONDS = 10
TOL = 1e-6

def _chk(name, ok, detail): return {"name": name, "passed": bool(ok), "detail": detail}
def _num(x, what):
    try: v = float(x)
    except (TypeError, ValueError): raise ValueError(f"{what} must be a number, got {x!r}")
    if not math.isfinite(v): raise ValueError(f"{what} must be a finite number")
    return v
def _vec(inst, key, n=None, nonneg=False):
    if key not in inst: raise ValueError(f"Missing '{key}'. See the instance format for this family.")
    v = [_num(x, f"{key}[{i}]") for i, x in enumerate(inst[key])]
    if n is not None and len(v) != n: raise ValueError(f"'{key}' has {len(v)} entries; expected {n}.")
    if nonneg and any(x < 0 for x in v): raise ValueError(f"'{key}' must not contain negative numbers.")
    return v
def _mat(inst, key, rows=None, cols=None):
    if key not in inst: raise ValueError(f"Missing '{key}'. See the instance format for this family.")
    M = [[_num(x, f"{key} entry") for x in r] for r in inst[key]]
    if not M or any(len(r) != len(M[0]) for r in M): raise ValueError(f"'{key}' must be a non-empty rectangular table.")
    if rows is not None and len(M) != rows: raise ValueError(f"'{key}' has {len(M)} rows; expected {rows}.")
    if cols is not None and len(M[0]) != cols: raise ValueError(f"'{key}' has {len(M[0])} columns; expected {cols}.")
    return M
def _names(inst, key, n, default):
    nm = inst.get(key)
    if nm is None: return [f"{default} {i + 1}" for i in range(n)]
    if len(nm) != n: raise ValueError(f"'{key}' has {len(nm)} names; expected {n}.")
    return [str(x) for x in nm]
def _limit(n, cap, what):
    if n > cap: raise ValueError(f"{what}: {n} exceeds the limit of {cap} for this server. Reduce the instance or use a dedicated solver such as OR-Tools or Gurobi (see the playbook).")

# ------------------------------------------------------------------ knapsack
def s_knapsack(inst):
    v = _vec(inst, "values"); n = len(v); w = _vec(inst, "weights", n, True); cap = _num(inst.get("capacity"), "capacity")
    if n < 1: raise ValueError("Give at least one item.")
    _limit(n, 5000, "Items")
    if cap < 0: raise ValueError("Capacity must not be negative.")
    nm = _names(inst, "names", n, "Item")
    scale = 1 if all(float(x).is_integer() for x in w + [cap]) else 100
    iw = [int(round(x * scale)) for x in w]; icap = int(math.floor(cap * scale + 1e-9))
    checks = []; exact = icap * n <= DP_CELLS
    if exact:
        val, ch = R.knapsack(v, iw, icap); method, guar = "Dynamic programming over capacity (exact)", "Optimal" + (" (weights rounded to 0.01)" if scale == 100 else "")
    else:
        # Too large for the table method. Try the mixed-integer solver (HiGHS) with a time limit and keep the better of it and the ratio greedy.
        val, ch = R.knapsack_greedy(v, [max(x, 1e-12) for x in w], cap); method, guar = "Ratio greedy plus best single item", "pending"
        try:
            r_ = milp(c=-np.array(v), constraints=LinearConstraint(np.array([w]), -np.inf, cap), integrality=np.ones(n), bounds=Bounds(0, 1), options={"time_limit": MIP_SECONDS, "mip_rel_gap": 1e-9})
            if r_.x is not None:
                ch2 = [i for i in range(n) if r_.x[i] > .5]; v2 = sum(v[i] for i in ch2)
                if sum(w[i] for i in ch2) <= cap + TOL and v2 >= val - TOL: val, ch = v2, ch2; method = "Mixed-integer programming (HiGHS) with greedy fallback"
                if r_.status == 0 and v2 >= val - TOL: guar = "Optimal (proven by the solver)"
        except Exception: pass
    # independent re-derivation
    if n <= 200 and exact and not getattr(_TL, "fast", False):
        v2, c2 = R.multi_knapsack(v, [w], [cap]); checks.append(_chk("Independent solve by mixed-integer programming", abs(v2 - val) <= TOL * max(1, abs(val)) * (1 if scale == 1 else 100), f"DP {val:.6g} vs MIP {v2:.6g}"))
    # upper bound from the LP relaxation (fractional greedy)
    ub = 0.0; room = cap
    for i in sorted((i for i in range(n) if v[i] > 0), key=lambda i: -(v[i] / w[i] if w[i] > 0 else math.inf)):
        if w[i] <= room: ub += v[i]; room -= w[i]
        else: ub += v[i] * (room / w[i]) if w[i] > 0 else 0; break
    ub = max(ub, val)
    tw, tv = sum(w[i] for i in ch), sum(v[i] for i in ch)
    checks += [_chk("Capacity respected (recomputed)", tw <= cap + TOL, f"{tw:.6g} of {cap:.6g}"), _chk("Objective recomputed from the chosen items", abs(tv - val) <= 1e-6 * max(1, abs(val)), f"{tv:.6g}"),
               _chk("Objective within the LP upper bound", val <= ub + TOL, f"bound {ub:.6g}")]
    if guar == "pending": guar = (f"Within {round((ub - tv) / ub * 100, 3)}% of the optimum (LP upper bound)" if ub > 0 else "Optimal") + "; at least 1/2 of the optimum guaranteed"
    return {"decision": {"selected": [nm[i] for i in ch], "indices": sorted(ch), "resource_used": round(tw, 6), "resource_limit": cap}, "objective": tv, "objective_label": "total value", "method": method, "guarantee": guar,
            "checks": checks, "bound": {"upper": round(ub, 6), "gap_pct": round((ub - tv) / ub * 100, 3) if ub > 0 else 0.0, "kind": "LP relaxation"}}

def _knap_value(inst, cap):
    v = _vec(inst, "values"); w = _vec(inst, "weights", len(v), True)
    scale = 1 if all(float(x).is_integer() for x in w + [cap]) else 100
    return R.knapsack(v, [int(round(x * scale)) for x in w], int(math.floor(cap * scale + 1e-9)))[0]

def quantum_check(inst, val):
    """Runs the project's QAOA simulator on a small integer knapsack and compares with the exact value. Returns None when it does not apply."""
    from .optimizer import Opt, run_opt, slack_bits
    v = [float(x) for x in inst["values"]]; w = [float(x) for x in inst["weights"]]; cap = float(inst["capacity"]); n = len(v)
    if not (all(x.is_integer() and x >= 1 for x in w) and cap.is_integer() and cap >= 1 and all(x >= 0 for x in v)): return {"ran": False, "reason": "needs positive integer weights and a positive integer capacity"}
    if n > 16 or n + slack_bits(int(cap)) > 16: return {"ran": False, "reason": f"{n} items plus {slack_bits(int(cap))} budget bits exceed the 16-qubit simulator limit"}
    T = [{"name": f"i{i}", "cat": i, "p": 1.0, "E": v[i], "hl": 1e9, "c": int(w[i])} for i in range(n)]
    r = run_opt(Opt(budget=cap, trends=T, lam_overlap=0, lam_waste=0, ship_days=0, max_picks=n, backend="simulator"))
    q = r["quantum"]
    return {"ran": True, "backend": q.get("backend", "simulator"), "objective": q.get("objective"), "exact_objective": val, "matches_exact": bool(q.get("objective") is not None and abs(q["objective"] - val) < 1e-6 * max(1, abs(val))),
            "approximation_ratio": q.get("ratio"), "qubits": q.get("qubits"), "verified": q.get("verified"), "note": "Measured on a software simulator. No quantum advantage is claimed."}

# ------------------------------------------------------------------ multi-dimensional knapsack
def s_mdk(inst):
    v = _vec(inst, "values"); n = len(v); _limit(n, 120, "Items")
    A = _mat(inst, "A", None, n); b = _vec(inst, "b", len(A), True); nm = _names(inst, "names", n, "Item")
    if any(x < 0 for r in A for x in r): raise ValueError("'A' must not contain negative numbers.")
    val, ch = R.multi_knapsack(v, A, b); checks = []
    used = [sum(A[k][i] for i in ch) for k in range(len(A))]
    checks.append(_chk("Every resource limit respected (recomputed)", all(u <= bb + TOL for u, bb in zip(used, b)), "; ".join(f"resource {k + 1}: {u:.6g} of {bb:.6g}" for k, (u, bb) in enumerate(zip(used, b)))))
    tv = sum(v[i] for i in ch); checks.append(_chk("Objective recomputed", abs(tv - val) <= 1e-6 * max(1, abs(val)), f"{tv:.6g}"))
    if n <= 18 and not getattr(_TL, "fast", False):
        X = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(float); feas = (X @ np.array(A, float).T <= np.array(b) + 1e-9).all(1); best = float((X[feas] @ np.array(v)).max())
        checks.append(_chk("Independent solve by exhaustive search", abs(best - tv) <= 1e-6 * max(1, abs(best)), f"search {best:.6g} vs MIP {tv:.6g}")); bound = None
    else:
        r = milp(c=-np.array(v), constraints=LinearConstraint(np.array(A), -np.inf, np.array(b)), integrality=np.zeros(n), bounds=Bounds(0, 1)); ub = -r.fun
        checks.append(_chk("Objective within the LP upper bound", tv <= ub + TOL, f"bound {ub:.6g}")); bound = {"upper": round(ub, 6), "gap_pct": round((ub - tv) / ub * 100, 3) if ub > 0 else 0.0, "kind": "LP relaxation"}
    return {"decision": {"selected": [nm[i] for i in ch], "indices": [int(i) for i in ch], "resource_used": [round(u, 6) for u in used], "resource_limit": b}, "objective": tv, "objective_label": "total value",
            "method": "Mixed-integer programming (HiGHS)", "guarantee": "Optimal (solver proves optimality within its tolerance)", "checks": checks, "bound": bound}

# ------------------------------------------------------------------ assignment
def s_assignment(inst):
    C = _mat(inst, "cost"); m, k = len(C), len(C[0]); _limit(max(m, k), 400, "Rows or columns"); mx = bool(inst.get("maximize", False))
    rn = _names(inst, "row_names", m, "Row"); cn = _names(inst, "col_names", k, "Column")
    capc = inst.get("col_capacity"); cols = []
    if capc is not None:
        capc = [int(_num(x, "col_capacity")) for x in capc]
        if len(capc) != k or any(c < 0 for c in capc): raise ValueError("'col_capacity' needs one non-negative whole number per column.")
        for j, c in enumerate(capc): cols += [j] * c
        if m > len(cols): raise ValueError(f"{m} rows cannot all be placed: total column capacity is {len(cols)}.")
        E = [[C[i][j] for j in cols] for i in range(m)]
    else: cols = list(range(k)); E = C
    tot, pairs = R.assignment(E, maximize=mx); pairs = [(i, cols[j]) for i, j in pairs]; checks = []
    rows_used = [i for i, _ in pairs]
    checks.append(_chk("Each row assigned once", len(set(rows_used)) == len(rows_used) == min(m, len(cols)), f"{len(rows_used)} of {m} rows assigned"))
    load = {}
    for _, j in pairs: load[j] = load.get(j, 0) + 1
    checks.append(_chk("Column capacity respected", all(load[j] <= (capc[j] if capc else 1) for j in load), "one per column" if not capc else "within stated capacity"))
    re_tot = sum(C[i][j] for i, j in pairs); checks.append(_chk("Total recomputed", abs(re_tot - tot) <= TOL * max(1, abs(tot)), f"{re_tot:.6g}"))
    # independent: MILP on the expanded matrix
    if m * len(cols) <= 4000:
        Ee = np.array(E, float); mm, nn = Ee.shape; Aeq = []; Aub = []
        for i in range(mm): a = np.zeros(mm * nn); a[i * nn:(i + 1) * nn] = 1; Aeq.append(a)
        for j in range(nn): a = np.zeros(mm * nn); a[j::nn] = 1; Aub.append(a)
        cons = [LinearConstraint(np.array(Aeq), 1, 1), LinearConstraint(np.array(Aub), 0, 1)] if mm <= nn else [LinearConstraint(np.array(Aeq), 0, 1), LinearConstraint(np.array(Aub), 1, 1)]
        r = milp(c=(-1 if mx else 1) * Ee.ravel(), constraints=cons, integrality=np.ones(mm * nn), bounds=Bounds(0, 1)); alt = (-1 if mx else 1) * r.fun
        checks.append(_chk("Independent solve by mixed-integer programming", r.success and abs(alt - tot) <= 1e-6 * max(1, abs(tot)), f"Hungarian {tot:.6g} vs MIP {alt:.6g}"))
    unassigned = [rn[i] for i in range(m) if i not in set(rows_used)]
    return {"decision": {"pairs": [{"row": rn[i], "column": cn[j], "cost": C[i][j]} for i, j in sorted(pairs)], "unassigned_rows": unassigned}, "objective": tot, "objective_label": "total " + ("value" if mx else "cost"),
            "method": "Hungarian algorithm (Jonker-Volgenant form)", "guarantee": "Optimal (polynomial time)", "checks": checks, "bound": None}

# ------------------------------------------------------------------ set cover
def s_set_cover(inst):
    U = list(inst.get("universe") or []); S = [list(s) for s in inst.get("sets") or []]; n = len(S)
    if not U or not S: raise ValueError("Give a 'universe' list and a 'sets' list.")
    costs = _vec(inst, "costs", n, True) if "costs" in inst else [1.0] * n; nm = _names(inst, "names", n, "Set"); _limit(n, 100_000, "Sets")
    allc = set().union(*map(set, S)); miss = [u for u in U if u not in allc]
    if miss: raise ValueError(f"Elements that no set covers: {miss[:5]}. Add a set that covers them or remove them from the universe.")
    gc, gch = R.set_cover_greedy(U, S, costs); checks = []
    d = max(len(set(s) & set(U)) for s in S); H = sum(1 / i for i in range(1, d + 1))
    exact = n <= 250 and len(U) <= 3000
    if exact:
        ix = {u: i for i, u in enumerate(U)}; M = np.zeros((len(U), n))
        for j, s in enumerate(S):
            for u in s:
                if u in ix: M[ix[u], j] = 1
        r = milp(c=np.array(costs), constraints=LinearConstraint(M, 1, np.inf), integrality=np.ones(n), bounds=Bounds(0, 1), options={"time_limit": 20})
        if r.x is None: exact = False
    if exact:
        ch = [int(j) for j in np.flatnonzero(np.round(r.x) == 1)]; tot = sum(costs[j] for j in ch); proven = r.status == 0
        method, guar = "Mixed-integer programming (HiGHS)", "Optimal (proven)" if proven else "Best found within the time limit; not proven optimal"
        checks.append(_chk("Greedy within its H(d) guarantee", gc <= H * tot + TOL, f"greedy {gc:.6g}, optimum {tot:.6g}, bound factor {H:.3f}"))
    else: ch, tot = gch, gc; method, guar = "Greedy weighted set cover", f"Within a factor of {H:.3f} of the optimum (H(d), d={d})"
    cov = set().union(*[set(S[j]) for j in ch]); checks.insert(0, _chk("Every element covered (recomputed)", set(U) <= cov, f"{len(set(U) & cov)} of {len(set(U))} elements"))
    checks.append(_chk("Cost recomputed", abs(sum(costs[j] for j in ch) - tot) <= TOL * max(1, tot), f"{tot:.6g}"))
    return {"decision": {"selected": [nm[j] for j in ch], "indices": sorted(ch), "greedy_cost": gc}, "objective": tot, "objective_label": "total cost", "method": method, "guarantee": guar, "checks": checks, "bound": None}

# ------------------------------------------------------------------ facility location
def s_facility(inst):
    D = _mat(inst, "dist"); m, n = len(D), len(D[0]); _limit(m * n, 6000, "Demand points times sites"); f = _vec(inst, "fixed", n, True); d = _vec(inst, "demand", m, True)
    k = inst.get("k"); k = int(k) if k is not None else None; sn = _names(inst, "site_names", n, "Site"); dn = _names(inst, "demand_names", m, "Demand")
    if k is not None and not 1 <= k <= n: raise ValueError(f"'k' must be between 1 and {n}.")
    val, op = R.facility_location(f, D, d, k); op = [int(j) for j in op]; checks = []
    if not op: raise ValueError("The solver opened no site; check the instance.")
    serve = [min(op, key=lambda j: D[i][j]) for i in range(m)]
    tot = sum(f[j] for j in op) + sum(d[i] * D[i][serve[i]] for i in range(m))
    checks.append(_chk("Cost recomputed from open sites and nearest service", abs(tot - val) <= 1e-6 * max(1, abs(val)), f"{tot:.6g} vs solver {val:.6g}"))
    checks.append(_chk("Site-count cap respected", k is None or len(op) <= k, f"{len(op)} open" + (f", cap {k}" if k else "")))
    if n <= 14:
        Dn = np.array(D); dn_ = np.array(d); best = math.inf
        for r_ in range(1, (k or n) + 1):
            for S in itertools.combinations(range(n), r_): best = min(best, sum(f[j] for j in S) + float((Dn[:, S].min(1) * dn_).sum()))
        checks.append(_chk("Independent solve by exhaustive search over site sets", abs(best - tot) <= 1e-6 * max(1, best), f"search {best:.6g}"))
    return {"decision": {"open_sites": [sn[j] for j in op], "indices": op, "service": [{"demand": dn[i], "served_by": sn[serve[i]], "distance": D[i][serve[i]]} for i in range(m)]}, "objective": tot, "objective_label": "total cost (fixed + service)",
            "method": "Mixed-integer programming (uncapacitated facility location)", "guarantee": "Optimal (solver proves optimality within its tolerance)", "checks": checks, "bound": None}

# ------------------------------------------------------------------ budget allocation
def _curve(c):
    t, a, b = c.get("type", "exp"), _num(c.get("a", 1), "a"), _num(c.get("b", 1), "b")
    if a < 0 or b <= 0: raise ValueError("Curve parameters need a >= 0 and b > 0.")
    if t == "exp": return lambda s: a * (1 - math.exp(-b * s))
    if t == "log": return lambda s: a * math.log1p(b * s)
    if t == "sqrt": return lambda s: a * math.sqrt(b * s)
    raise ValueError(f"Unknown curve type {t!r}. Use exp, log or sqrt (all concave).")
def s_budget(inst):
    cs = inst.get("curves") or []; n = len(cs); B = _num(inst.get("budget"), "budget"); step = _num(inst.get("step", 1), "step")
    if n < 1: raise ValueError("Give at least one curve.")
    _limit(n, 500, "Options")
    if B < 0 or step <= 0 or B / step > 200_000: raise ValueError("Budget must be non-negative and budget/step at most 200,000.")
    F = [_curve(c) for c in cs]; lo = [float(c.get("min", 0)) for c in cs]; hi = [float(c.get("max", math.inf)) for c in cs]; nm = [str(c.get("name", f"Option {i + 1}")) for i, c in enumerate(cs)]
    if sum(lo) > B + TOL: raise ValueError(f"Minimum spends total {sum(lo):g}, above the budget {B:g}.")
    marg = [(lambda s, j=j: 0.0 if s >= hi[j] - 1e-12 else F[j](s + step) - F[j](s)) for j in range(n)]
    base = [lo[j] for j in range(n)]; left = B - sum(base)
    # shift each marginal by its minimum spend
    shifted = [(lambda s, j=j: marg[j](s + lo[j])) for j in range(n)]
    extra = R.budget_allocation(shifted, left, step); s = [base[j] + extra[j] for j in range(n)]
    obj = sum(F[j](s[j]) for j in range(n)); checks = []
    checks.append(_chk("Total spend within budget", sum(s) <= B + 1e-6, f"{sum(s):.6g} of {B:.6g}"))
    checks.append(_chk("Per-option limits respected", all(lo[j] - 1e-9 <= s[j] <= hi[j] + step * 1e-9 + 1e-9 or hi[j] == math.inf for j in range(n)), "minimum and maximum spends"))
    checks.append(_chk("Objective recomputed", abs(sum(F[j](s[j]) for j in range(n)) - obj) < 1e-9, f"{obj:.6g}"))
    bnds = [(lo[j], min(hi[j], B)) for j in range(n)]
    r = minimize(lambda x: -sum(F[j](x[j]) for j in range(n)), np.array([max(lo[j], B / n) if hi[j] == math.inf else min(hi[j], B / n) for j in range(n)]), method="SLSQP", bounds=bnds,
                 constraints=[{"type": "ineq", "fun": lambda x: B - sum(x)}])
    ub = -r.fun if r.success else None; bound = None
    if ub is not None:
        checks.append(_chk("Not above the continuous optimum (independent solve)", obj <= ub + 1e-4 * max(1, abs(ub)), f"continuous {ub:.6g}, step {step:g}")); bound = {"upper": round(ub, 6), "gap_pct": round(max(0, (ub - obj) / ub * 100), 3) if ub > 0 else 0.0, "kind": "continuous optimum (gap comes from the step size)"}
    return {"decision": {"allocation": [{"option": nm[j], "spend": round(s[j], 6)} for j in range(n)]}, "objective": obj, "objective_label": "total response",
            "method": "Greedy equimarginal allocation", "guarantee": "Optimal for separable concave responses on the step grid", "checks": checks, "bound": bound}

# ------------------------------------------------------------------ portfolio selection
def _qp(mu, cov, tr, mw, idx=None):
    idx = list(range(len(mu))) if idx is None else idx; m = np.array([mu[i] for i in idx]); c = np.array([[cov[i][j] for j in idx] for i in idx]); w, var = R.portfolio_min_variance(m, c, tr, mw)
    # The reference call stops at SLSQP's default tolerance; polish from its answer at a tight one and keep whichever feasible point is better.
    cons = [{"type": "eq", "fun": lambda x: x.sum() - 1}, {"type": "ineq", "fun": lambda x: x @ m - tr}]
    r = minimize(lambda x: x @ c @ x, np.clip(w, 0, mw), method="SLSQP", bounds=[(0, mw)] * len(idx), constraints=cons, options={"ftol": 1e-14, "maxiter": 1000})
    if r.success and abs(r.x.sum() - 1) < 1e-8 and r.x @ m >= tr - 1e-8 and r.fun < var: w, var = r.x, float(r.fun)
    return idx, w, var
def s_portfolio(inst):
    mu = _vec(inst, "mu"); n = len(mu); cov = _mat(inst, "cov", n, n); tr = _num(inst.get("target_return"), "target_return"); mw = _num(inst.get("max_weight", 1.0), "max_weight"); K = inst.get("max_assets"); nm = _names(inst, "names", n, "Asset")
    _limit(n, 200, "Assets")
    if not 0 < mw <= 1: raise ValueError("'max_weight' must be in (0, 1].")
    if mw * n < 1 - 1e-9: raise ValueError("'max_weight' is too small: the weights cannot add to 1.")
    C = np.array(cov); 
    if not np.allclose(C, C.T, atol=1e-9): raise ValueError("'cov' must be symmetric.")
    if np.linalg.eigvalsh(C).min() < -1e-9: raise ValueError("'cov' must be positive semi-definite.")
    top = 0.0; left = 1.0
    for x in sorted(mu, reverse=True): t = min(mw, left); top += x * t; left -= t
    if tr > top + 1e-9: raise ValueError(f"Target return {tr:g} is above the highest achievable return {top:.6g} under these limits.")
    checks = []; method, guar = "Sequential quadratic programming (convex, long-only)", "Optimal for the convex model"
    if K is not None:
        K = int(K)
        if not 1 <= K <= n: raise ValueError(f"'max_assets' must be between 1 and {n}.")
    if K is not None and K < n:
        combos = math.comb(n, K)
        if combos <= 800:
            best = None
            for S in itertools.combinations(range(n), K):
                if mw * K < 1 - 1e-9: continue
                try: idx, w, var = _qp(mu, cov, tr, mw, list(S))
                except Exception: continue
                if abs(w.sum() - 1) < 1e-6 and w @ np.array([mu[i] for i in idx]) >= tr - 1e-6 and (best is None or var < best[2]): best = (idx, w, var)
            if best is None: raise ValueError("No subset of that size reaches the target return.")
            idx, w, var = best; method, guar = f"Enumeration of all {combos} size-{K} subsets, each solved as a convex QP", "Optimal for the cardinality-constrained model"
        else:
            idx, w, var = _qp(mu, cov, tr, mw)
            while len(idx) > K:
                drop = int(np.argmin(w)); keep = [i for k_, i in enumerate(idx) if k_ != drop]; idx, w, var = _qp(mu, cov, tr, mw, keep)
            method, guar = "Backward elimination with convex QP re-solves", "Heuristic for the cardinality limit; no optimality proof"
        W = np.zeros(n); W[idx] = w
    else:
        idx, w, var = _qp(mu, cov, tr, mw); W = np.zeros(n); W[idx] = w
    W = np.where(np.abs(W) < 1e-9, 0, W); var = float(W @ C @ W); ret = float(W @ np.array(mu))
    checks += [_chk("Weights sum to 1", abs(W.sum() - 1) < 1e-6, f"{W.sum():.8f}"), _chk("Long-only and within the weight cap", W.min() >= -1e-9 and W.max() <= mw + 1e-6, f"max weight {W.max():.4f}"),
               _chk("Target return met", ret >= tr - 1e-6, f"{ret:.6g} against {tr:.6g}"), _chk("Cardinality respected", K is None or int((W > 1e-6).sum()) <= K, f"{int((W > 1e-6).sum())} assets held")]
    if K is None or K >= n:
        rng = np.random.default_rng(3); alt = math.inf
        for _ in range(3):
            x0 = rng.dirichlet(np.ones(n)); r = minimize(lambda w_: w_ @ C @ w_, x0, method="SLSQP", bounds=[(0, mw)] * n, constraints=[{"type": "eq", "fun": lambda w_: w_.sum() - 1}, {"type": "ineq", "fun": lambda w_: w_ @ np.array(mu) - tr}], options={"ftol": 1e-14, "maxiter": 1000})
            if r.success and abs(r.x.sum() - 1) < 1e-8 and r.x @ np.array(mu) >= tr - 1e-8: alt = min(alt, float(r.fun))
        checks.append(_chk("Independent solve from random starts", abs(var - alt) <= 1e-4 * max(var, alt) + 1e-10, f"{var:.6g} vs {alt:.6g}"))
    eq = np.full(n, 1 / n); checks.append(_chk("Not worse than equal weights in variance, if equal weights meet the target", float(np.array(mu) @ eq) < tr or var <= float(eq @ C @ eq) + 1e-9, f"equal-weight variance {float(eq @ C @ eq):.6g}"))
    return {"decision": {"weights": [{"asset": nm[i], "weight": round(float(W[i]), 6)} for i in range(n) if W[i] > 0], "expected_return": round(ret, 6), "volatility": round(math.sqrt(max(var, 0)), 6)}, "objective": var,
            "objective_label": "portfolio variance", "method": method, "guarantee": guar, "checks": checks, "bound": None}

# ------------------------------------------------------------------ scheduling
def s_schedule(inst):
    t = _vec(inst, "durations", None, True); n = len(t); m = int(inst.get("machines", 1)); w = inst.get("weights"); nm = _names(inst, "names", n, "Job"); _limit(n, 100_000, "Jobs")
    if m < 1 or n < 1 or any(x <= 0 for x in t): raise ValueError("Need at least one job with positive durations and at least one machine.")
    checks = []
    if m == 1 and w is not None:
        w = _vec(inst, "weights", n); 
        if any(x <= 0 for x in w): raise ValueError("Weights must be positive.")
        order = R.wspt_order(w, t); c = 0.0; tot = 0.0
        for j in order: c += t[j]; tot += w[j] * c
        if n <= 8:
            best = min(sum(w[j] * sum(t[k] for k in p[:p.index(j) + 1]) for j in p) for p in itertools.permutations(range(n)))
            checks.append(_chk("Independent solve by exhaustive search", abs(best - tot) <= TOL * max(1, tot), f"search {best:.6g}"))
        seq = [j for j in order]; re = 0.0; c = 0.0
        for j in seq: c += t[j]; re += w[j] * c
        checks.append(_chk("Objective recomputed from the sequence", abs(re - tot) < 1e-6 * max(1, tot), f"{re:.6g}")); checks.append(_chk("Every job scheduled once", sorted(seq) == list(range(n)), f"{len(seq)} jobs"))
        return {"decision": {"sequence": [nm[j] for j in order]}, "objective": tot, "objective_label": "total weighted completion time", "method": "Smith's rule (weighted shortest processing time)", "guarantee": "Optimal", "checks": checks, "bound": None}
    mk, plan = R.schedule_identical_machines(t, m); lb = max(sum(t) / m, max(t)); loads = [sum(t[j] for j in p) for p in plan]
    checks += [_chk("Every job on exactly one machine", sorted(j for p in plan for j in p) == list(range(n)), f"{n} jobs on {m} machines"), _chk("Makespan recomputed", abs(max(loads) - mk) < 1e-9 * max(1, mk), f"{max(loads):.6g}"),
               _chk("Not below the lower bound", mk >= lb - 1e-9, f"lower bound {lb:.6g}")]
    guar = f"Within {(4 / 3 - 1 / (3 * m)):.3f} of the optimum (Graham's LPT bound)"
    if n <= 12 and m <= 4:
        x = n * m + 1; c = np.zeros(x); c[-1] = 1; A = []; lo = []; hi = []
        for j in range(n): a = np.zeros(x); a[j * m:(j + 1) * m] = 1; A.append(a); lo.append(1); hi.append(1)
        for k in range(m): a = np.zeros(x); a[[j * m + k for j in range(n)]] = t; a[-1] = -1; A.append(a); lo.append(-np.inf); hi.append(0)
        r = milp(c, constraints=LinearConstraint(np.array(A), lo, hi), integrality=np.concatenate([np.ones(n * m), [0]]), bounds=Bounds(0, np.concatenate([np.ones(n * m), [sum(t)]])), options={"time_limit": 15})
        if r.x is not None and r.status == 0:
            opt = float(r.fun); checks.append(_chk("LPT within its 4/3 guarantee of the exact optimum (independent solve)", mk <= (4 / 3 - 1 / (3 * m)) * opt + TOL, f"LPT {mk:.6g}, optimum {opt:.6g}"))
            if opt < mk - 1e-9:  # report the better schedule the exact solver found
                xs = np.round(r.x[:n * m]).reshape(n, m); plan = [[j for j in range(n) if xs[j, k] == 1] for k in range(m)]; mk = opt; guar = "Optimal (proven by mixed-integer programming)"; loads = [sum(t[j] for j in p) for p in plan]
                checks.append(_chk("Exact schedule recomputed", abs(max(loads) - opt) < 1e-6 * max(1, opt), f"{max(loads):.6g}"))
            else: guar = "Optimal (confirmed by mixed-integer programming)"
    return {"decision": {"machines": [{"machine": k + 1, "jobs": [nm[j] for j in p], "load": round(sum(t[j] for j in p), 6)} for k, p in enumerate(plan)]}, "objective": mk, "objective_label": "makespan",
            "method": "Longest processing time first" + (" (improved by exact solve)" if guar.startswith("Optimal (proven") else ""), "guarantee": guar, "checks": checks, "bound": {"lower": round(lb, 6), "gap_pct": round((mk - lb) / lb * 100, 3), "kind": "max(total/m, longest job)"}}

# ------------------------------------------------------------------ routing
def _rlen(depot, pts, r): return math.dist(depot, pts[r[0]]) + sum(math.dist(pts[r[k]], pts[r[k + 1]]) for k in range(len(r) - 1)) + math.dist(pts[r[-1]], depot)
def _two_opt(depot, pts, r):
    r = r[:]; imp = True
    while imp and len(r) > 2:
        imp = False
        for i in range(len(r) - 1):
            for j in range(i + 1, len(r)):
                c = r[:i] + r[i:j + 1][::-1] + r[j + 1:]
                if _rlen(depot, pts, c) < _rlen(depot, pts, r) - 1e-12: r = c; imp = True
    return r
def s_routing(inst):
    depot = tuple(_vec({"d": inst.get("depot")}, "d", 2)); pts = [tuple(_vec({"p": p}, "p", 2)) for p in inst.get("points") or []]; n = len(pts); _limit(n, 400, "Stops")
    if n < 1: raise ValueError("Give at least one stop in 'points'.")
    dem = _vec(inst, "demand", n, True) if "demand" in inst else [1.0] * n; cap = _num(inst.get("capacity", sum(dem) or 1), "capacity"); nm = _names(inst, "names", n, "Stop")
    if any(x > cap for x in dem): raise ValueError("A stop's demand is larger than the vehicle capacity, so it cannot be served.")
    d0, routes = R.vehicle_routes_savings(depot, pts, dem, cap); checks = []
    routes = [_two_opt(depot, pts, r) for r in routes]; tot = sum(_rlen(depot, pts, r) for r in routes)
    checks += [_chk("Every stop visited exactly once", sorted(i for r in routes for i in r) == list(range(n)), f"{n} stops on {len(routes)} routes"), _chk("Vehicle capacity respected on every route", all(sum(dem[i] for i in r) <= cap + TOL for r in routes), f"capacity {cap:g}"),
               _chk("Distance recomputed", abs(tot - sum(_rlen(depot, pts, r) for r in routes)) < 1e-9, f"{tot:.6g}"), _chk("2-opt did not lengthen the savings routes", tot <= d0 + 1e-9, f"savings {d0:.6g}, after 2-opt {tot:.6g}")]
    guar = "Heuristic: no optimality guarantee"; bound = None
    if n <= 9 and sum(dem) <= cap + TOL:
        best = min(_rlen(depot, pts, list(p)) for p in itertools.permutations(range(n)))
        checks.append(_chk("Compared with exhaustive search (single route fits all stops)", tot >= best - 1e-9, f"best possible {best:.6g}")); bound = {"upper": None, "gap_pct": round((tot - best) / best * 100, 3) if best > 0 else 0.0, "kind": "gap to the exact optimum"}
        if tot > best + 1e-9:
            p = min(itertools.permutations(range(n)), key=lambda p: _rlen(depot, pts, list(p))); routes = [list(p)]; tot = best; guar = "Optimal (exhaustive search)"
        else: guar = "Optimal (confirmed by exhaustive search)"
    return {"decision": {"routes": [{"vehicle": k + 1, "stops": [nm[i] for i in r], "load": sum(dem[i] for i in r), "distance": round(_rlen(depot, pts, r), 6)} for k, r in enumerate(routes)]}, "objective": tot, "objective_label": "total distance",
            "method": "Clarke-Wright savings with 2-opt improvement", "guarantee": guar, "checks": checks, "bound": bound}

SOLVERS = {"knapsack": s_knapsack, "multi_dim_knapsack": s_mdk, "assignment": s_assignment, "set_cover": s_set_cover, "facility_location": s_facility, "budget_allocation": s_budget,
           "portfolio_selection": s_portfolio, "scheduling": s_schedule, "routing": s_routing}

def solve_instance(family, inst, quantum=True):
    """Solve a supplied instance. Raises ValueError with a plain message for bad input."""
    if family not in SOLVERS: raise ValueError(f"Unknown family {family!r}. Choose one of: {', '.join(FAMILIES)}.")
    if not isinstance(inst, dict): raise ValueError("'instance' must be a JSON object.")
    t0 = time.time(); r = SOLVERS[family](inst)
    if family == "knapsack":
        sens = []
        try:
            cap = float(inst["capacity"])
            for f_ in (.9, 1.1): sens.append({"capacity_change_pct": round((f_ - 1) * 100), "objective": round(_knap_value(inst, cap * f_), 6)})
        except Exception: sens = []
        if sens: r["sensitivity"] = sens
        if quantum:
            try: r["quantum_check"] = quantum_check(inst, r["objective"])
            except Exception as e: r["quantum_check"] = {"ran": False, "reason": f"not run: {e}"}
    r["verified"] = all(c["passed"] for c in r["checks"]); r["runtime_ms"] = round((time.time() - t0) * 1000, 1); r["family"] = family
    r["objective"] = round(float(r["objective"]), 6); return r

def _fast_solve(family, inst):
    _TL.fast = True
    try: return solve_instance(family, inst, quantum=False)
    finally: _TL.fast = False

# ------------------------------------------------------------------ instance formats and illustrative instances
FORMATS = {
    "knapsack": {"values": "list of numbers", "weights": "list of non-negative numbers", "capacity": "number", "names": "optional list"},
    "multi_dim_knapsack": {"values": "list", "A": "resources x items table", "b": "one limit per resource", "names": "optional"},
    "assignment": {"cost": "rows x columns table", "maximize": "optional true/false", "col_capacity": "optional whole number per column", "row_names": "optional", "col_names": "optional"},
    "set_cover": {"universe": "list of elements", "sets": "list of lists of elements", "costs": "optional, one per set", "names": "optional"},
    "facility_location": {"fixed": "opening cost per site", "dist": "demands x sites table", "demand": "weight per demand point", "k": "optional cap on open sites"},
    "budget_allocation": {"curves": "list of {name, type: exp|log|sqrt, a, b, min?, max?}", "budget": "number", "step": "optional, default 1"},
    "portfolio_selection": {"mu": "expected return per asset", "cov": "covariance table", "target_return": "number", "max_weight": "optional, default 1", "max_assets": "optional"},
    "scheduling": {"durations": "list", "machines": "whole number", "weights": "optional; with one machine this minimises weighted completion time"},
    "routing": {"depot": "[x, y]", "points": "list of [x, y]", "demand": "optional list", "capacity": "optional number"}}

def demo_instance(family, n=8, seed=7):
    """A generated instance for trying the solvers. It is not the person's data and every result built on it says so."""
    R_ = random.Random(seed); n = max(3, min(int(n), 12))
    if family == "knapsack": w = [R_.randint(2, 15) for _ in range(n)]; return {"values": [R_.randint(10, 60) for _ in range(n)], "weights": w, "capacity": sum(w) // 2, "names": [f"Option {i + 1}" for i in range(n)]}
    if family == "multi_dim_knapsack": return {"values": [R_.randint(10, 60) for _ in range(n)], "A": [[R_.randint(1, 9) for _ in range(n)], [R_.randint(1, 9) for _ in range(n)]], "b": [sum(range(n)) // 2 + 6, n * 2 + 4], "names": [f"Option {i + 1}" for i in range(n)]}
    if family == "assignment": k = min(n, 6); return {"cost": [[R_.randint(2, 30) for _ in range(k)] for _ in range(k)], "row_names": [f"Person {i + 1}" for i in range(k)], "col_names": [f"Task {j + 1}" for j in range(k)]}
    if family == "set_cover":
        U = list(range(1, n + 3)); S = [sorted(R_.sample(U, R_.randint(2, 4))) for _ in range(n)]; S.append(U[:2]); S.append(U[2:]); return {"universe": U, "sets": S, "costs": [R_.randint(2, 9) for _ in S]}
    if family == "facility_location": s = min(n, 6); m = n; return {"fixed": [R_.randint(8, 20) for _ in range(s)], "dist": [[R_.randint(1, 12) for _ in range(s)] for _ in range(m)], "demand": [R_.randint(1, 5) for _ in range(m)]}
    if family == "budget_allocation": return {"curves": [{"name": f"Channel {i + 1}", "type": R_.choice(["exp", "log", "sqrt"]), "a": R_.randint(20, 80), "b": round(R_.uniform(.05, .4), 2)} for i in range(min(n, 6))], "budget": 40, "step": 1}
    if family == "portfolio_selection":
        k = min(n, 6); mu = [round(R_.uniform(.03, .14), 3) for _ in range(k)]; A_ = np.array([[R_.uniform(-.1, .2) for _ in range(k)] for _ in range(k)]); C = (A_ @ A_.T) * .12 + np.diag([.01] * k)
        return {"mu": mu, "cov": [[round(float(x), 5) for x in r] for r in C], "target_return": round(sum(mu) / k, 3), "max_weight": .5, "names": [f"Asset {i + 1}" for i in range(k)]}
    if family == "scheduling": return {"durations": [R_.randint(2, 12) for _ in range(n)], "machines": 3}
    if family == "routing": return {"depot": [0, 0], "points": [[R_.randint(-10, 10), R_.randint(-10, 10)] for _ in range(n)], "demand": [R_.randint(1, 4) for _ in range(n)], "capacity": 10}
    raise ValueError(f"Unknown family {family!r}.")

def family_from_instance(inst):
    """The keys of a supplied instance identify its family, so no guessing is needed when numbers are given."""
    if not isinstance(inst, dict): return None
    k = set(inst)
    for fam, need in (("multi_dim_knapsack", {"values", "A", "b"}), ("knapsack", {"values", "weights", "capacity"}), ("assignment", {"cost"}), ("set_cover", {"universe", "sets"}),
                      ("facility_location", {"fixed", "dist", "demand"}), ("budget_allocation", {"curves", "budget"}), ("portfolio_selection", {"mu", "cov"}),
                      ("routing", {"depot", "points"}), ("scheduling", {"durations"})):
        if need <= k: return fam
    return None

def instance_from_text(text):
    """Reads '- Name: cost X value Y' lines and 'Budget of N' into a knapsack instance. Returns None when the text does not contain one."""
    p = parse(text)
    if len(p["opts"]) >= 2 and p["budget"] is not None:
        return {"values": [o["v"] for o in p["opts"]], "weights": [o["c"] for o in p["opts"]], "capacity": p["budget"], "names": [o["n"] for o in p["opts"]]}, p
    return None, p

# ------------------------------------------------------------------ formulation text (generated from the catalogue row, not copied)
FORM = {
    "knapsack": ("x_i in {0,1}: choose option i. {sense} sum_i s_i x_i", "sum_i r_i x_i <= R"),
    "multi_dim_knapsack": ("x_i in {0,1}: choose item i. {sense} sum_i s_i x_i", "sum_i a_ik x_i <= R_k for every resource k"),
    "portfolio_selection": ("w_i >= 0 share, y_i in {0,1} if held. {sense} mu'w - lambda*risk(w)", "sum_i w_i = 1; w_i <= u_i y_i; sum_i y_i <= K"),
    "budget_allocation": ("s_j >= 0 spend on option j. {sense} sum_j f_j(s_j), f_j a fitted response curve", "sum_j s_j <= B; l_j <= s_j <= u_j"),
    "scheduling": ("x_jt in {0,1}: task j starts in slot t. {sense} sum_jt c_jt x_jt", "sum_t x_jt = 1; load in slot t <= cap_t"),
    "assignment": ("x_ij in {0,1}: i is matched to j. {sense} sum_ij c_ij x_ij", "sum_j x_ij = 1; sum_i x_ij <= cap_j"),
    "facility_location": ("y_j in {0,1}: site j open; x_ij in [0,1]: share of demand i served by j. {sense} sum_j f_j y_j + sum_ij d_i c_ij x_ij", "sum_j x_ij = 1; x_ij <= y_j"),
    "set_cover": ("x_i in {0,1}: pick set i. {sense} sum_i c_i x_i", "sum over sets covering e of x_i >= 1 for every target e"),
    "routing": ("x_ijk in {0,1}: vehicle k drives from i to j. {sense} total travel + lateness", "each stop visited once; load <= Q; time windows a_i <= t_i <= b_i")}
def formulation(family, row=None):
    d, c = FORM[family]; row = row or {}; sense = "min" if str(row.get("objective", "")).lower().startswith("min") else "max"
    out = {"family": family, "decision_and_objective": d.replace("{sense}", sense), "constraints": c}
    if row.get("key_constraints"): out["problem_specific_limits"] = row["key_constraints"]
    if row.get("typical_variables"): out["data_to_collect"] = [x.strip() for x in row["typical_variables"].split(",")]
    if row.get("objective"): out["objective_in_words"] = row["objective"]
    out["note"] = "Generated from the problem's catalogue row and the family template. Confirm it before relying on it."
    return out

# ------------------------------------------------------------------ the pipeline
def run(text, classify_fn=None, instance=None, family=None, catalogue_id=None, use_demo=False, demo_n=8, seed=7, quantum=True, detail=True):
    t0 = time.time(); trail = []; row = None
    if catalogue_id:
        row = bigdata.get(catalogue_id)
        if not row: raise ValueError(f"Catalogue id {catalogue_id!r} was not found. " + ("" if bigdata.available() else "The catalogue index has not been built on this server (python -m app.bigdata build ...)."))
        text = text or row["problem"]; trail.append(f"Catalogue row {row['id']} loaded")
    text = (text or "").strip()
    if not text and not instance and not family: raise ValueError("Describe the problem, give a catalogue id, or supply a family and an instance.")
    jobs = {"parse": lambda: instance_from_text(text) if text else (None, {"opts": [], "budget": None, "unread": [], "stakes": "Medium"}), "classify": lambda: classify_fn(text) if (classify_fn and text) else [],
            "retrieve": lambda: (bigdata.search(text, 5), bigdata.family_vote(text)) if text else ([], {})}
    with ThreadPoolExecutor(3) as ex: fut = {k: ex.submit(f) for k, f in jobs.items()}; res = {}
    for k, f in fut.items():
        try: res[k] = f.result(timeout=20)
        except Exception as e: res[k] = None; trail.append(f"{k}: skipped ({type(e).__name__})")
    text_inst, parsed = res["parse"] if res.get("parse") else (None, {"opts": [], "unread": [], "budget": None})
    model = res.get("classify") or []; near, vote = res.get("retrieve") or ([], {}); trail.append(f"Ran in parallel: parse, classify ({len(model)} candidates), retrieve ({len(near)} neighbours)")
    # The trained classifier decides. Catalogue retrieval is shown as supporting context only: measured on held-out statements it was far less accurate
    # than the classifier, and averaging the two was worse than the classifier alone (tests/measure_family.py). The vote is the fallback when no classifier ran.
    ranked = [{"family": m["family"], "score": m["p"]} for m in model] or [{"family": f, "score": p} for f, p in vote.items()]
    # choose the family
    src = None
    shape = family_from_instance(instance)
    if family: src = "supplied"
    elif shape: family, src = shape, "shape of the numbers supplied"
    elif row: family, src = row["formulation_family"], "catalogue row"
    elif text_inst: family, src = "knapsack", "options and budget read from the text"
    elif model: family, src = model[0]["family"], "classifier"
    elif ranked: family, src = ranked[0]["family"], "catalogue vote (no classifier result)"
    if not family: return {"tier": 5, "family": None, "confidence": {"level": "provisional", "reason": "The problem type could not be determined."}, "recommendation": "I could not tell which kind of problem this is. Describe what you choose between, what you want to maximise or minimise, and the limits.",
                           "questions": ["What are you choosing between?", "What do you want to maximise or minimise?", "Which limits apply?"], "trail": trail, "neighbours": near}
    if family not in SOLVERS: raise ValueError(f"Unknown family {family!r}.")
    if instance and shape and shape != family: raise ValueError(f"The numbers supplied look like a {shape.replace('_', ' ')} instance, but the problem type is {family.replace('_', ' ')}. Change the type or the numbers.")
    agree = bool(model and model[0]["family"] == family) + bool(vote and next(iter(vote)) == family)
    row = row or (next((h for h in near if h["formulation_family"] == family), None) if near else None)
    sens = (row or {}).get("trend_sensitivity", "medium")
    inst, inst_src = None, None
    if instance: inst, inst_src = instance, "supplied"
    elif text_inst and family == "knapsack": inst, inst_src = text_inst, "read from your text"
    elif use_demo: inst, inst_src = demo_instance(family, demo_n, seed), "illustrative (generated by the engine; not your data)"
    n_items = len(next(iter(inst.values()))) if inst and isinstance(next(iter(inst.values()), None), list) else 20
    pb = bigdata.expand(bigdata.playbook_for(family, n_items, sens)); trail.append(f"Family {family} ({src}); playbook {pb['playbook_id'] if pb else 'none'}")
    out = {"family": family, "family_source": src, "classification": {"model": model, "catalogue_vote": vote, "methods_agree": agree}, "neighbours": near, "catalogue_row": row,
           "formulation": formulation(family, row), "playbook": pb, "instance_format": FORMATS[family], "instance_source": inst_src, "trail": trail, "parsed": {"unread": parsed.get("unread", []), "budget": parsed.get("budget")}}
    alt = [x for x in (ranked or []) if x["family"] != family][:2]
    out["alternatives"] = [{"family": x["family"], "score": x["score"], "when": f"if the problem is really about: {bigdata_hint(x['family'])}"} for x in alt]
    out["sources"] = [f"Theorem {t['theorem_id']}: {t['name']}" for t in (pb or {}).get("theorems", [])] + [f"{r['authors']} ({r['year']}), {r['title']}, {r['venue']}" for r in (pb or {}).get("references", [])]
    if near: out["sources"].append("Catalogue problems: " + ", ".join(h["id"] for h in near[:3]))
    if inst is None:
        out.update(tier=3, confidence={"level": "provisional", "reason": "No instance data was supplied, so there are no numbers to solve. This is the formulation and the method the playbook recommends."},
                   recommendation=f"This reads as a {family.replace('_', ' ')} problem. Method: {pb['recommended_algorithm'] if pb else 'see the formulation'}. Send the data listed below, or ask for an illustrative run, to get a verified decision.",
                   questions=([f"What are your figures for: {x}?" for x in out["formulation"].get("data_to_collect", [])[:3]] + ([f"Which of these limits apply: {row['key_constraints']}?"] if row and row.get("key_constraints") else []))
                   or [f"Send the data in this form: {', '.join(FORMATS[family])}. GET /api/v1/solve/formats shows an example.", "Which limits apply, and what deadline?"])
        out["runtime_ms"] = round((time.time() - t0) * 1000, 1); return out
    with ThreadPoolExecutor(1) as ex:
        f = ex.submit(solve_instance, family, inst, quantum)
        try: sol = f.result(timeout=SOLVE_TIMEOUT)
        except FutureTimeout: out.update(tier=3, confidence={"level": "provisional", "reason": f"The solver did not finish within {SOLVE_TIMEOUT} seconds."}, recommendation="The instance was too large for this server within the time budget. Use the playbook's solver stack for this size.", questions=["Can the instance be reduced?"]); return out
    out["solution"] = sol; illus = inst_src.startswith("illustrative"); agreed = src in ("supplied", "shape of the numbers supplied", "catalogue row", "options and budget read from the text") or (src == "classifier" and model[0]["p"] >= .6)
    lvl = "high" if (sol["verified"] and not illus and agreed and not parsed.get("unread")) else "provisional" if (illus or not sol["verified"]) else "medium"
    why = []
    if sol["verified"]: why.append(f"All {len(sol['checks'])} checks passed, including an independent re-derivation" if any("Independent" in c["name"] or "exhaustive" in c["name"] or "random starts" in c["name"] for c in sol["checks"]) else f"All {len(sol['checks'])} checks passed")
    else: why.append("At least one check failed, so the answer is not presented as verified")
    if illus: why.append("the numbers are illustrative, not yours")
    if not agreed: why.append("the problem type was inferred with low confidence")
    if src == "classifier" and agree < 2: why.append("the catalogue vote does not confirm the problem type")
    if parsed.get("unread"): why.append("some lines of your text could not be read and were left out")
    out["confidence"] = {"level": lvl, "reason": "; ".join(why) + "."}; out["tier"] = 1 if sol["verified"] else 3
    out["recommendation"] = headline(family, sol, inst_src)
    out["assumptions"] = [{"item": "Instance data", "value": inst_src, "source": "supplied" if inst_src in ("supplied", "read from your text") else "default"}, {"item": "Problem family", "value": family, "source": "supplied" if src == "supplied" else "inferred"}]
    out["risks"] = [(pb or {}).get("uncertainty_handling", "")] + ([f"Gap to the upper bound: {sol['bound']['gap_pct']}%."] if sol.get("bound") and sol["bound"].get("gap_pct") else []); out["monitor"] = (pb or {}).get("kpis_to_monitor", "")
    out["questions"] = ["Which estimates in this instance are least certain? The playbook suggests testing the answer against them."]
    if detail and sol["verified"]:
        try:
            from . import insight
            an = insight.analyze(family, inst, sol, _fast_solve); out["analysis"] = an; out["detail"] = insight.narrative(family, sol, an, out["recommendation"], illus)
            trail.append(f"What-if analysis: {len(an.get('what_if', []))} changes tried, {an.get('stability', {}).get('trials', 0)} stability re-solves, {an['seconds']} s")
        except Exception as e: trail.append(f"What-if analysis skipped ({type(e).__name__})")
    out["runtime_ms"] = round((time.time() - t0) * 1000, 1); return out

def bigdata_hint(f): return {"knapsack": "choosing items within one budget", "multi_dim_knapsack": "choosing under several limits", "portfolio_selection": "balancing return and risk", "budget_allocation": "splitting money with diminishing returns",
                             "scheduling": "placing work in time", "assignment": "matching one to one", "facility_location": "choosing sites", "set_cover": "covering every need cheaply", "routing": "ordering stops"}.get(f, f)

def headline(f, s, src):
    d = s["decision"]; tag = " (illustrative data)" if src.startswith("illustrative") else ""
    if f in ("knapsack", "multi_dim_knapsack"): return f"Choose {', '.join(d['selected']) or 'nothing'} for a {s['objective_label']} of {s['objective']:g}{tag}."
    if f == "assignment": return f"Make {len(d['pairs'])} assignments for a {s['objective_label']} of {s['objective']:g}{tag}."
    if f == "set_cover": return f"Pick {', '.join(d['selected'])} to cover everything at a {s['objective_label']} of {s['objective']:g}{tag}."
    if f == "facility_location": return f"Open {', '.join(d['open_sites'])} for a {s['objective_label']} of {s['objective']:g}{tag}."
    if f == "budget_allocation": return "Spend " + ", ".join(f"{a['spend']:g} on {a['option']}" for a in d["allocation"] if a["spend"] > 0) + f" for a {s['objective_label']} of {s['objective']:g}{tag}."
    if f == "portfolio_selection": return "Hold " + ", ".join(f"{w['weight']:.1%} {w['asset']}" for w in d["weights"]) + f" for an expected return of {d['expected_return']:.2%} and volatility {d['volatility']:.2%}{tag}."
    if f == "scheduling": return (f"Run the jobs in this order: {', '.join(d['sequence'])}" if "sequence" in d else f"Spread the jobs over {len(d['machines'])} machines for a makespan of {s['objective']:g}") + f"{tag}."
    if f == "routing": return f"Use {len(d['routes'])} route(s) for a total distance of {s['objective']:g}{tag}."
    return str(d)
