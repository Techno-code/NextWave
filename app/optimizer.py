"""Trend portfolio optimiser (Trend Portfolio Optimizer PRD, section 6B).

Each trend may be funded at one tier (for example test, standard, full) through one channel (for example organic, paid, creator), or not at all.
With no tiers and no channels every trend has a single option and the model is the original one-option-per-trend model.

Pieces
  Model            builds the options, costs, values and overlap penalties from a request
  exact()          branch-and-bound reference with a proof of optimality unless the node cap is hit
  build_qubo()     explicit QUBO (one-hot, budget slack bits) that any annealer or gate-model solver can take
  run_simulator()  statevector QAOA, p=2, CVaR, up to 16 qubits
  run_anneal()     software simulated annealing (classical; same interface a quantum annealer would plug into)
  run_hardware()   adapter that sends the QUBO to an external solver service the operator configures; nothing is faked when none is configured
Every backend result is decoded, repaired and re-verified by the same classical checker before it is reported.
"""
import json, math, os, random, time, urllib.error, urllib.request
import numpy as np
from pydantic import BaseModel, Field

QUBIT_CAP = 16
NODE_CAP = 2_000_000
BACKENDS = ("simulator", "anneal", "hardware")

class Tier(BaseModel):
    name: str = Field(max_length=40)
    cost_mult: float = Field(gt=0, le=20)
    value_mult: float = Field(ge=0, le=20)
class Channel(BaseModel):
    name: str = Field(max_length=40)
    cost_mult: float = Field(1, gt=0, le=20)
    reach_mult: float = Field(1, ge=0, le=20)

DEFAULT_TIERS = [Tier(name="Test", cost_mult=.5, value_mult=.55), Tier(name="Standard", cost_mult=1, value_mult=1), Tier(name="Full", cost_mult=1.8, value_mult=1.4)]
DEFAULT_CHANNELS = [Channel(name="Organic", cost_mult=.7, reach_mult=.8), Channel(name="Paid social", cost_mult=1, reach_mult=1), Channel(name="Creator partnership", cost_mult=1.4, reach_mult=1.3)]

class Opt(BaseModel):
    budget: float = 12; risk: str = "medium"; lam_overlap: float = 1; lam_waste: float = .5; ship_days: int = 3; max_picks: int = 6; seed: int = 7
    trends: list[dict] = []
    tiers: list[Tier] = Field([], max_length=5)
    channels: list[Channel] = Field([], max_length=5)
    lam_channel: float = Field(.3, ge=0, le=5)
    backend: str = "simulator"
    compare_backends: bool = False
    hardware_confirm: bool = False
    hardware_max_cost: float = Field(0, ge=0)

class BackendUnavailable(Exception): pass

NM = ["Quiet luxury packaging", "Micro-session fitness", "Retro handheld gaming", "Protein snacks", "Local-first apps", "Thrift-haul videos", "Desk plant setups", "AI study planners", "Slow-travel itineraries", "Cold-plunge routines"]
CAT = [0, 1, 2, 0, 3, 4, 4, 3, 5, 1]

class Model:
    def __init__(s, b: Opt):
        R = random.Random(b.seed)
        s.b = b
        s.T = b.trends or [{"name": n, "cat": CAT[i], "p": round(.3 + R.random() * .6, 2), "E": round(20 + R.random() * 40), "hl": round(5 + R.random() * 20), "c": round(2 + R.random() * 4)} for i, n in enumerate(NM)]
        T = s.T; s.n = n = len(T)
        if not 1 <= n <= 16: raise ValueError("The exact reference supports 1 to 16 trends.")
        if b.backend not in BACKENDS: raise ValueError("Backend must be simulator, anneal or hardware.")
        k = {"low": 1.5, "medium": .8, "high": .2}[b.risk]
        s.v = [t["p"] * t["E"] * .5 ** (b.ship_days / t["hl"]) for t in T]
        s.rv_t = [x - k * x * (1 - t["p"]) * .6 for x, t in zip(s.v, T)]
        s.w_t = [t["c"] * (1 - t["p"]) * .8 for t in T]
        s.mv = mv = sum(s.v) / n
        tiers = b.tiers or [None]; chans = b.channels or [None]; s.expanded = bool(b.tiers or b.channels)
        s.O = []; s.by_trend = [[] for _ in range(n)]
        for i, t in enumerate(T):
            for ta in tiers:
                for ch in chans:
                    cm = (ta.cost_mult if ta else 1) * (ch.cost_mult if ch else 1); vm = (ta.value_mult if ta else 1) * (ch.reach_mult if ch else 1)
                    c = max(1, round(t["c"] * cm)) if s.expanded else t["c"]
                    o = {"trend": i, "tier": ta.name if ta else None, "channel": ch.name if ch else None, "c": c, "rv": s.rv_t[i] * vm, "w": s.w_t[i] * cm}
                    o["gain"] = o["rv"] - b.lam_waste * o["w"]; s.by_trend[i].append(len(s.O)); s.O.append(o)
        nO = s.nO = len(s.O)
        base = lambda i, j: (.6 if T[i]["cat"] == T[j]["cat"] else .05) * b.lam_overlap * mv * .5
        s.S = [[0.0] * nO for _ in range(nO)]
        for a in range(nO):
            for c_ in range(nO):
                i, j = s.O[a]["trend"], s.O[c_]["trend"]
                if i == j: continue
                x = base(i, j)
                if s.O[a]["channel"] is not None and s.O[a]["channel"] == s.O[c_]["channel"]: x += b.lam_channel * b.lam_overlap * mv * .5
                s.S[a][c_] = x
    # assignment = list of length n, option index or -1
    def sel(s, A): return [a for a in A if a >= 0]
    def cost(s, A): return sum(s.O[a]["c"] for a in s.sel(A))
    def ok(s, A): return s.cost(A) <= s.b.budget and len(s.sel(A)) <= s.b.max_picks
    def f(s, A):
        x = s.sel(A)
        return sum(s.O[a]["gain"] for a in x) - sum(s.S[a][c_] for k, a in enumerate(x) for c_ in x[k + 1:])
    def waste(s, A): return sum(s.O[a]["w"] for a in s.sel(A))
    def name(s, a):
        o = s.O[a]; return s.T[o["trend"]]["name"]
    def plan(s, A):
        return [{"trend": s.name(a), "tier": s.O[a]["tier"], "channel": s.O[a]["channel"], "cost": s.O[a]["c"]} for a in s.sel(A)]
    def check(s, A):
        """Independent re-verification from the raw request, not from the cached option tables."""
        x = s.sel(A); b = s.b
        if len({s.O[a]["trend"] for a in x}) != len(x): return False
        if sum(s.O[a]["c"] for a in x) > b.budget or len(x) > b.max_picks: return False
        for a in x:
            o = s.O[a]; t = s.T[o["trend"]]
            ta = next((q for q in b.tiers if q.name == o["tier"]), None); ch = next((q for q in b.channels if q.name == o["channel"]), None)
            cm = (ta.cost_mult if ta else 1) * (ch.cost_mult if ch else 1)
            c = max(1, round(t["c"] * cm)) if s.expanded else t["c"]
            if abs(c - o["c"]) > 1e-9: return False
        return True
    def from_mask(s, m):
        """Bitmask over options -> assignment, keeping the best-gain option when a trend has several bits set."""
        A = [-1] * s.n
        for a in range(s.nO):
            if m >> a & 1:
                i = s.O[a]["trend"]
                if A[i] < 0 or s.O[a]["gain"] > s.O[A[i]]["gain"]: A[i] = a
        return A
    def mask_ok(s, m):
        t = [s.O[a]["trend"] for a in range(s.nO) if m >> a & 1]
        return len(set(t)) == len(t) and s.ok(s.from_mask(m))
    def repair(s, A):
        A = A[:]
        while not s.ok(A):
            i = min((i for i in range(s.n) if A[i] >= 0), key=lambda i: s.O[A[i]]["rv"] / (s.O[A[i]]["c"] or 1e-9)); A[i] = -1
        imp = True
        while imp:
            imp = False
            for i in range(s.n):
                best, bf = A[i], s.f(A)
                for alt in [-1] + s.by_trend[i]:
                    if alt == A[i]: continue
                    Y = A[:]; Y[i] = alt
                    if s.ok(Y) and s.f(Y) > bf + 1e-9: best, bf = alt, s.f(Y)
                if best != A[i]: A[i] = best; imp = True
        return A

def _hull_segments(points):
    """Upper concave hull of (cost, gain) points joined to the origin, as (cost step, gain step, slope) segments with falling slopes."""
    pts = sorted({(c, g) for c, g in points if g > 0 and c > 0}); hull = [(0.0, 0.0)]
    best = {}
    for c, g in pts: best[c] = max(best.get(c, 0), g)
    pts = []; top = 0.0
    for c in sorted(best):                       # drop points that cost more but gain no more
        if best[c] > top: pts.append((c, best[c])); top = best[c]
    for p in pts:
        while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) >= 0: hull.pop()
        hull.append(p)
    return [(hull[i + 1][0] - hull[i][0], hull[i + 1][1] - hull[i][1], (hull[i + 1][1] - hull[i][1]) / (hull[i + 1][0] - hull[i][0])) for i in range(len(hull) - 1)]

def exact(M: Model, k=3):
    """Top-k distinct portfolios by branch and bound. Returns (list of (value, assignment), proven_optimal, nodes).
    The bound is the linear relaxation of the multiple-choice knapsack on the trends still undecided: it respects the remaining budget and ignores the overlap penalties, which can only lower a portfolio's value, so it never cuts off a better portfolio."""
    if M.b.budget < 0 or M.b.max_picks < 0: return [], True, 0      # even the empty portfolio breaks a constraint
    order = sorted(range(M.n), key=lambda i: -max(M.O[a]["gain"] for a in M.by_trend[i]))
    suffix = [[] for _ in range(M.n + 1)]
    for p in range(M.n - 1, -1, -1):
        suffix[p] = sorted(suffix[p + 1] + _hull_segments([(M.O[a]["c"], M.O[a]["gain"]) for a in M.by_trend[order[p]]]), key=lambda x: -x[2])
    def bound(p, room):
        v = 0.0
        for dc, dg, sl in suffix[p]:
            if room <= 0: break
            if dc <= room: v += dg; room -= dc
            else: v += sl * room; break
        return v
    top = []; nodes = [0]; capped = [False]; A = [-1] * M.n; chosen = []
    def rec(p, cur, cost):
        nodes[0] += 1
        if nodes[0] > NODE_CAP: capped[0] = True; return
        if p == M.n:
            key = tuple(A)
            if len(top) < k or cur > top[-1][0]:
                top.append((cur, key)); top.sort(key=lambda x: (-x[0], x[1])); del top[k:]
            return
        if len(top) == k and cur + bound(p, M.b.budget - cost) <= top[-1][0] - 1e-12: return
        i = order[p]
        for a in sorted(M.by_trend[i], key=lambda a: -M.O[a]["gain"]):
            c = cost + M.O[a]["c"]
            if c > M.b.budget or len(chosen) + 1 > M.b.max_picks: continue
            A[i] = a; delta = M.O[a]["gain"] - sum(M.S[a][q] for q in chosen); chosen.append(a)
            rec(p + 1, cur + delta, c); chosen.pop(); A[i] = -1
            if capped[0]: return
        rec(p + 1, cur, cost)
    rec(0, 0.0, 0.0)
    return [(v, list(a)) for v, a in top], not capped[0], nodes[0]

# ---------- QUBO ----------
def slack_bits(budget): return max(1, math.ceil(math.log2(int(budget) + 1)))

def energies(M: Model, z, nS):
    """Vectorised QUBO energy for every bitstring in z (options in the low bits, budget slack bits above them)."""
    E = np.zeros(len(z)); sp = np.zeros(len(z)); P_oh = max(o["rv"] for o in M.O) * 1.5 + 1
    for a in range(M.nO):
        ba = (z >> a) & 1; E -= ba * M.O[a]["gain"]; sp += ba * M.O[a]["c"]
        for c_ in range(a + 1, M.nO):
            if M.O[c_]["trend"] == M.O[a]["trend"]: E += ba * ((z >> c_) & 1) * P_oh
            else: E += ba * ((z >> c_) & 1) * M.S[a][c_]
    E += max(max(o["rv"] for o in M.O), 1) * .7 * (int(M.b.budget) - sp - (z >> M.nO)) ** 2
    return E

def build_qubo(M: Model):
    """Returns (Q, offset, num_vars). Energy of bit vector x is offset + sum Q[(i,j)] x_i x_j (i<=j). Variables: options, then budget slack bits."""
    nS = slack_bits(M.b.budget); nv = M.nO + nS; Q = {}; P_oh = max(o["rv"] for o in M.O) * 1.5 + 1; Pn = max(max(o["rv"] for o in M.O), 1) * .7; B = int(M.b.budget)
    add = lambda i, j, x: Q.__setitem__((min(i, j), max(i, j)), Q.get((min(i, j), max(i, j)), 0.0) + x)
    for a in range(M.nO):
        add(a, a, -M.O[a]["gain"])
        for c_ in range(a + 1, M.nO): add(a, c_, P_oh if M.O[c_]["trend"] == M.O[a]["trend"] else M.S[a][c_])
    coef = [M.O[a]["c"] for a in range(M.nO)] + [2 ** k for k in range(nS)]   # budget:  B - sum coef_v x_v = 0
    for u in range(nv):
        add(u, u, Pn * (coef[u] ** 2 - 2 * B * coef[u]))
        for v in range(u + 1, nv): add(u, v, Pn * 2 * coef[u] * coef[v])
    return Q, Pn * B * B, nv

def qubo_energy(Q, offset, x): return offset + sum(v * x[i] * x[j] for (i, j), v in Q.items())

# ---------- backends ----------
def _result(M, A, ef, t0, **extra):
    qf = M.f(A)
    return {"objective": round(qf, 2), "ratio": round(qf / ef, 3) if ef > 0 else 1.0, "verified": bool(M.ok(A) and M.check(A)), "trends": [M.name(a) for a in M.sel(A)], "plan": M.plan(A), "ms": round((time.time() - t0) * 1000), **extra}

def run_simulator(M: Model, ef):
    nS = slack_bits(M.b.budget); qb = M.nO + nS
    if qb > QUBIT_CAP or not all(float(o["c"]).is_integer() for o in M.O):
        return {"skipped": f"The statevector simulator needs integer costs and at most {QUBIT_CAP} qubits (options plus budget bits); this request needs {qb}. Use the anneal backend for larger problems.", "backend": "local statevector simulator"}
    t0 = time.time(); N = 1 << qb; z = np.arange(N); E = energies(M, z, nS)
    En = (E - E.min()) / ((E.max() - E.min()) or 1); order = np.argsort(En); Eo = En[order]
    def run(pr):
        psi = np.full(N, 1 / math.sqrt(N), dtype=complex)
        for l in range(len(pr) // 2):
            psi *= np.exp(-2j * math.pi * pr[2 * l] * En); c, s2 = math.cos(pr[2 * l + 1]), math.sin(pr[2 * l + 1])
            for kq in range(qb):
                vw = psi.reshape(-1, 2, 1 << kq); a = vw[:, 0, :].copy(); bb = vw[:, 1, :].copy()
                vw[:, 0, :] = c * a - 1j * s2 * bb; vw[:, 1, :] = c * bb - 1j * s2 * a
        return np.abs(psi) ** 2
    def cvar(pr):
        p_ = run(pr)[order]; cum = np.cumsum(p_)
        return float((np.minimum(p_, np.maximum(.1 - (cum - p_), 0)) * Eo).sum() / .1)
    Rg = random.Random(11); best = None; bv = 1e9
    for _ in range(40):
        pr = [Rg.random(), Rg.random() * 1.6, Rg.random(), Rg.random() * 1.6]; val = cvar(pr)
        if val < bv: bv, best = val, pr
    first, st = bv, .2
    for _ in range(70):
        pr = [x + (Rg.random() - .5) * st for x in best]; val = cvar(pr)
        if val < bv: bv, best = val, pr
        st *= .97
    pf = run(best); pf /= pf.sum(); shots = 1024
    zs, cn = np.unique(np.random.default_rng(11).choice(N, size=shots, p=pf), return_counts=True)
    lowmask = (1 << M.nO) - 1; feas = sum(int(c_) for z_, c_ in zip(zs, cn) if M.mask_ok(int(z_) & lowmask)); bestA = None
    for z_ in zs[np.argsort(-cn)][:40]:
        A = M.repair(M.from_mask(int(z_) & lowmask))
        if bestA is None or M.f(A) > M.f(bestA): bestA = A
    return _result(M, bestA, ef, t0, qubits=qb, shots=shots, depth=2 * (qb * 2 + 2) + qb, raw_feasibility=round(feas / shots, 3), cvar_start=round(first, 3), cvar_end=round(bv, 3),
                   backend="local statevector simulator, QAOA p=2, CVaR 0.1")

def run_anneal(M: Model, ef, reads=30, steps=1500):
    """Constraint-preserving simulated annealing in software. Not a quantum annealer: it is the classical stand-in behind the same interface."""
    t0 = time.time(); R = random.Random(M.b.seed + 101); scale = (sum(abs(o["gain"]) for o in M.O) / M.nO) or 1.0; best = None; bf = -1e18; finals = []
    for _ in range(reads):
        A = [-1] * M.n; cur = 0.0
        for s_ in range(steps):
            T_ = scale * (.6 * (.005 / .6) ** (s_ / steps))
            i = R.randrange(M.n); alts = [-1] + M.by_trend[i]; alt = R.choice(alts)
            if alt == A[i]: continue
            Y = A[:]; Y[i] = alt
            if not M.ok(Y): continue
            nf = M.f(Y); d = nf - cur
            if d >= 0 or R.random() < math.exp(d / T_): A, cur = Y, nf
            if cur > bf + 1e-12: bf, best = cur, A[:]
        finals.append(cur)
    A = M.repair(best if best is not None else [-1] * M.n); top = max(finals) if finals else 0
    return _result(M, A, ef, t0, reads=reads, steps_per_read=steps, reads_reaching_best=round(sum(1 for x in finals if x >= top - 1e-9) / len(finals), 3),
                   backend="software simulated annealing (classical)")

def run_hardware(M: Model, ef):
    url = os.environ.get("QUBO_SOLVER_URL")
    if not url: raise BackendUnavailable("No hardware solver is configured on this server. The operator sets QUBO_SOLVER_URL, QUBO_SOLVER_COST_PER_RUN and optionally QUBO_SOLVER_TOKEN.")
    cost = os.environ.get("QUBO_SOLVER_COST_PER_RUN")
    if cost is None: raise BackendUnavailable("The operator has not stated a cost per run (QUBO_SOLVER_COST_PER_RUN), so paid compute cannot be used.")
    cost = float(cost)
    if not M.b.hardware_confirm: raise BackendUnavailable(f"This backend costs {cost:g} per run. Set hardware_confirm to true to confirm.")
    if cost > M.b.hardware_max_cost: raise BackendUnavailable(f"This backend costs {cost:g} per run, above your spending cap of {M.b.hardware_max_cost:g}.")
    t0 = time.time(); Q, off, nv = build_qubo(M)
    body = json.dumps({"num_vars": nv, "offset": off, "qubo": [[i, j, v] for (i, j), v in Q.items()], "num_reads": 200}).encode()
    hdr = {"content-type": "application/json"}
    if os.environ.get("QUBO_SOLVER_TOKEN"): hdr["authorization"] = "Bearer " + os.environ["QUBO_SOLVER_TOKEN"]
    try: out = json.load(urllib.request.urlopen(urllib.request.Request(url, body, hdr), timeout=int(os.environ.get("QUBO_SOLVER_TIMEOUT", "120"))))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e: return {"skipped": f"The hardware solver did not answer: {e}", "backend": "external QUBO solver"}
    samples = out.get("samples") or []; bestA = None
    for smp in samples[:200]:
        bits = smp.get("bits") if isinstance(smp, dict) else smp
        try: m = sum(1 << a for a in range(M.nO) if int(bits[a]))
        except Exception: continue
        A = M.repair(M.from_mask(m))
        if bestA is None or M.f(A) > M.f(bestA): bestA = A
    if bestA is None: return {"skipped": "The hardware solver returned no usable samples.", "backend": "external QUBO solver"}
    return _result(M, bestA, ef, t0, samples=len(samples), cost_charged=cost, backend=str(out.get("backend") or "external QUBO solver"))

RUNNERS = {"simulator": run_simulator, "anneal": run_anneal, "hardware": run_hardware}

def run_opt(b: Opt):
    M = Model(b); res, proven, nodes = exact(M)
    if not res: raise ValueError("No feasible portfolio. Raise the budget or the selection cap.")
    out = [{"trends": [M.name(a) for a in M.sel(A)], "spend": M.cost(A), "net_value": round(v, 2), "verified": bool(M.ok(A) and M.check(A)), "plan": M.plan(A)} for v, A in res]
    ef, eA = res[0]
    ran = {}
    names_ = [b.backend] + ([x for x in ("simulator", "anneal") if x != b.backend] if b.compare_backends else [])
    for nme in names_:
        try: ran[nme] = RUNNERS[nme](M, ef)
        except BackendUnavailable as e:
            if nme == b.backend: raise ValueError(str(e))
            ran[nme] = {"skipped": str(e), "backend": nme}
    qa = ran[b.backend]
    better = [(k, r) for k, r in ran.items() if r.get("verified") and r["objective"] > round(ef, 2)]
    rec = ("quantum" if better[0][0] == "simulator" else better[0][0]) if better else "classical exact"
    wa = sum(M.w_t[i] for i in range(M.n) if M.T[i]["p"] >= .5); wb = M.waste(eA)
    # waste of "fund every trending trend at the base option" uses base waste; unchanged from the single-option model
    ref = {"objective": round(ef, 2), "backend": "branch and bound", "proven_optimal": proven, "nodes": nodes}
    if not proven: ref["note"] = f"Node cap of {NODE_CAP} reached: this is the best portfolio found, not a proof of optimality."
    r = {"portfolios": out, "recommended": rec, "quantum": qa, "classical": ref, "waste_avoided_pct": round((wa - wb) / wa * 100, 1) if wa else 0,
         "note": "Measured on software. No quantum advantage is claimed.", "trends": M.T, "options_per_trend": len(M.by_trend[0])}
    if len(ran) > 1: r["backends"] = ran
    return r
