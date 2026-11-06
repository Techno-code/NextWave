"""Quant Insight: how firm is a solved answer, and what would change it?

Everything here is computed by re-solving the person's own instance with the same verified solvers; nothing is estimated from a formula
that was not checked. Three parts:
  closest_calls   knapsack and multi-resource knapsack: for each item, the objective lost if the decision on it were reversed
  levers          one-at-a-time changes (a bigger budget, one more machine, ...) with the new objective and which choices move
  stability       the instance's numbers are nudged by a stated percentage many times, the problem is re-solved each time, and we report how
                  often the same decision comes back, which choices never move, which are toss-ups, and the spread of the objective
A wall-clock budget caps all of it; whatever did not finish is listed under 'skipped' rather than guessed.
"""
import copy, math, random, statistics, time

NOISE = 0.10
TRIALS = 40


def _names(inst, key, n, default):
    v = inst.get(key)
    return [str(x) for x in v] if isinstance(v, list) and len(v) == n else [f"{default} {i + 1}" for i in range(n)]


def _labels(family, inst):
    if family in ("knapsack", "multi_dim_knapsack"): n = len(inst["values"]); return _names(inst, "names", n, "Item")
    if family == "set_cover": n = len(inst["sets"]); return _names(inst, "names", n, "Set")
    if family == "facility_location": n = len(inst["fixed"]); return _names(inst, "site_names", n, "Site")
    return None


def _sig(family, sol):
    """A hashable summary of the decision, and (when the decision is a set of choices) the set of chosen indices."""
    d = sol["decision"]
    if family in ("knapsack", "multi_dim_knapsack", "set_cover", "facility_location"):
        idx = frozenset(d["indices"]); return idx, idx
    if family == "assignment": return frozenset((p["row"], p["column"]) for p in d["pairs"]), None
    if family == "scheduling": return None, None   # many different job-to-machine splits tie for the best makespan, so "same split" says nothing; only the objective is tracked
    if family == "routing": return frozenset(tuple(r["stops"]) for r in d["routes"]), None
    if family == "budget_allocation":
        tot = sum(a["spend"] for a in d["allocation"]) or 1.0; return tuple(a["spend"] / tot for a in d["allocation"]), None
    if family == "portfolio_selection": return tuple(w["weight"] for w in d["weights"]), None
    return repr(d), None


def _same(family, a, b):
    if family in ("budget_allocation", "portfolio_selection"): return max(abs(x - y) for x, y in zip(a, b)) <= .05   # no share moves by more than 5 percentage points
    return a == b


def _jitter(rnd, x, eps): return x * (1 + rnd.uniform(-eps, eps))


def _perturb(family, inst, rnd, eps):
    """A copy of the instance with its estimated numbers moved by up to +/-eps (relative). Returns None when the family has nothing to move."""
    z = copy.deepcopy(inst)
    if family in ("knapsack", "multi_dim_knapsack"): z["values"] = [_jitter(rnd, float(x), eps) for x in z["values"]]
    elif family == "set_cover":
        if not z.get("costs"): return None
        z["costs"] = [_jitter(rnd, float(x), eps) for x in z["costs"]]
    elif family == "facility_location":
        z["fixed"] = [_jitter(rnd, float(x), eps) for x in z["fixed"]]; z["dist"] = [[_jitter(rnd, float(x), eps) for x in r] for r in z["dist"]]
    elif family == "assignment": z["cost"] = [[_jitter(rnd, float(x), eps) for x in r] for r in z["cost"]]
    elif family == "scheduling": z["durations"] = [_jitter(rnd, float(x), eps) for x in z["durations"]]
    elif family == "routing": z["points"] = [[_jitter(rnd, float(c), eps / 2) for c in p] for p in z["points"]]
    elif family == "budget_allocation": z["curves"] = [{**c, "a": _jitter(rnd, float(c.get("a", 1)), eps)} for c in z["curves"]]
    elif family == "portfolio_selection": z["mu"] = [_jitter(rnd, float(x), eps) for x in z["mu"]]
    else: return None
    return z


_WHAT = {"knapsack": "the values", "multi_dim_knapsack": "the values", "set_cover": "the costs", "facility_location": "the costs and distances", "assignment": "the costs",
         "scheduling": "the durations", "routing": "the stop locations", "budget_allocation": "the response strengths", "portfolio_selection": "the expected returns"}


def _decision_delta(family, base_sig, new_sig, labels):
    if family in ("knapsack", "multi_dim_knapsack", "set_cover", "facility_location") and labels:
        add = sorted(new_sig - base_sig); rem = sorted(base_sig - new_sig)
        return {"added": [labels[i] for i in add], "removed": [labels[i] for i in rem]}
    return {"changed": new_sig != base_sig}


# ---------------------------------------------------------------- levers
def _lever_instances(family, inst):
    """(description, modified instance) pairs: the single changes a person could plausibly make."""
    out = []
    def mod(desc, **kw): z = copy.deepcopy(inst); z.update(kw); out.append((desc, z))
    if family == "knapsack":
        c = float(inst["capacity"])
        for f in (.9, 1.1, 1.25): mod(f"capacity {'cut' if f < 1 else 'raised'} by {abs(round((f - 1) * 100))}%", capacity=c * f)
    elif family == "multi_dim_knapsack":
        for k, b in enumerate(inst["b"]):
            bb = [float(x) for x in inst["b"]]; bb[k] = bb[k] * 1.1; mod(f"limit {k + 1} raised by 10%", b=bb)
    elif family == "budget_allocation":
        b = float(inst["budget"])
        for f in (.9, 1.1, 1.25): mod(f"budget {'cut' if f < 1 else 'raised'} by {abs(round((f - 1) * 100))}%", budget=b * f)
    elif family == "scheduling":
        m = int(inst["machines"]); mod("one more machine", machines=m + 1)
        if m > 1: mod("one fewer machine", machines=m - 1)
    elif family == "facility_location" and inst.get("k"):
        k = int(inst["k"]); mod("allow one more open site", k=k + 1)
        if k > 1: mod("allow one fewer open site", k=k - 1)
    elif family == "portfolio_selection":
        t = float(inst["target_return"])
        for f in (.9, 1.1): mod(f"target return {'lowered' if f < 1 else 'raised'} by 10%", target_return=t * f)
    elif family == "routing" and inst.get("capacity"):
        mod("vehicle capacity raised by 25%", capacity=float(inst["capacity"]) * 1.25)
    return out


def _levers(family, inst, base, solve, labels, deadline):
    res, skipped = [], []; bsig, _ = _sig(family, base)
    for desc, z in _lever_instances(family, inst):
        if time.time() > deadline: skipped.append(desc); continue
        try: s = solve(family, z)
        except Exception as e: res.append({"change": desc, "note": f"could not be solved ({str(e)[:80]})"}); continue
        nsig, _ = _sig(family, s); b = float(base["objective"]); o = float(s["objective"]); row = {"change": desc, "objective": round(o, 6), "delta": round(o - b, 6)}
        if b: row["delta_pct"] = round((o - b) / abs(b) * 100, 2)
        row["decision"] = _decision_delta(family, bsig, nsig, labels); res.append(row)
    return res, skipped


# ---------------------------------------------------------------- closest calls (selection problems)
def _without(family, inst, i):
    z = copy.deepcopy(inst); n = len(z["values"])
    for key in ("values", "weights", "names"):
        if isinstance(z.get(key), list) and len(z[key]) == n: del z[key][i]
    if family == "multi_dim_knapsack": z["A"] = [[x for j, x in enumerate(r) if j != i] for r in z["A"]]
    return z


def _closest_calls(family, inst, base, solve, labels, deadline, max_items=40):
    n = len(inst["values"])
    if n > max_items or not labels: return None, [f"item-by-item analysis is limited to {max_items} items"] if n > max_items else []
    chosen = set(base["decision"]["indices"]); b = float(base["objective"]); rows, skipped = [], []
    for i in range(n):
        if time.time() > deadline: skipped.append(labels[i]); continue
        try:
            z = _without(family, inst, i)
            if i in chosen:                                          # what is lost if this chosen item is taken out
                o = float(solve(family, z)["objective"]); how = "left out"
            else:                                                    # what is lost if this unchosen item is forced in
                if family == "knapsack":
                    w = float(inst["weights"][i]); cap = float(inst["capacity"]) - w
                    if cap < 0: continue
                    z["capacity"] = cap
                else:
                    A = inst["A"]; bb = [float(x) - float(r[i]) for x, r in zip(inst["b"], A)]
                    if any(x < 0 for x in bb): continue
                    z["b"] = bb
                o = float(inst["values"][i]) + float(solve(family, z)["objective"]); how = "forced in"
        except Exception: continue
        rows.append({"item": labels[i], "currently": "chosen" if i in chosen else "not chosen", "if_reversed": how, "objective_if_reversed": round(o, 6), "cost_of_reversing": round(b - o, 6),
                     "cost_pct": round((b - o) / abs(b) * 100, 3) if b else 0.0})
    rows.sort(key=lambda r: r["cost_of_reversing"])
    return rows, skipped


# ---------------------------------------------------------------- stability
def _stability(family, inst, base, solve, labels, deadline, trials, eps, seed, workers=4):
    from concurrent.futures import ThreadPoolExecutor
    rnd = random.Random(seed); bsig, bset = _sig(family, base); zs = []
    for _ in range(trials):                                   # the nudged copies are drawn in order from one seed, so results do not depend on thread timing
        z = _perturb(family, inst, rnd, eps)
        if z is None: return None
        zs.append(z)
    def one(z):
        if time.time() > deadline: return None
        try: return solve(family, z)
        except Exception: return False
    with ThreadPoolExecutor(workers) as ex: sols = list(ex.map(one, zs))
    same = 0; objs = []; counts = {}; done = 0; failed = 0; regrets = []
    for z, s in zip(zs, sols):
        if s is None: continue
        if s is False: failed += 1; continue
        done += 1; sg, st = _sig(family, s); objs.append(float(s["objective"]))
        if bsig is not None: same += _same(family, bsig, sg)
        if st is not None:
            for i in st: counts[i] = counts.get(i, 0) + 1
        if family in ("knapsack", "multi_dim_knapsack"):                      # what acting on the original choice would have earned under these numbers
            v = z["values"]; kept = sum(v[i] for i in bset); best = float(s["objective"])
            if best > 0: regrets.append((best - kept) / best * 100)
    if done < 5: return {"trials": done, "note": "too few re-solves finished within the time budget to say anything about stability"}
    o = sorted(objs); q = lambda p: o[min(len(o) - 1, max(0, int(round(p * (len(o) - 1)))))]
    out = {"noise_pct": round(eps * 100), "perturbed": _WHAT.get(family, "the inputs"), "trials": done, "objective_range": {"p5": round(q(.05), 6), "median": round(statistics.median(o), 6), "p95": round(q(.95), 6)}}
    if bsig is not None: out["decision_unchanged_pct"] = round(same / done * 100, 1)
    if family in ("budget_allocation", "portfolio_selection"): out["decision_unchanged_means"] = "no option's share moved by more than 5 percentage points"
    if failed: out["trials_unsolvable"] = failed
    if bset is not None and labels:
        always = [labels[i] for i in sorted(bset) if counts.get(i, 0) >= .9 * done]; toss = [{"item": labels[i], "chosen_pct": round(counts.get(i, 0) / done * 100)} for i in range(len(labels)) if .1 * done < counts.get(i, 0) < .9 * done]
        out["always_chosen"] = always; out["toss_ups"] = sorted(toss, key=lambda r: abs(r["chosen_pct"] - 50)); out["never_or_rarely_chosen"] = sum(1 for i in range(len(labels)) if counts.get(i, 0) <= .1 * done)
    if regrets: out["cost_of_keeping_original_choice_pct"] = {"mean": round(statistics.mean(regrets), 3), "worst": round(max(regrets), 3)}
    if "decision_unchanged_pct" in out: pct = out["decision_unchanged_pct"]; out["robustness"] = "firm" if pct >= 80 else "sensitive" if pct >= 40 else "fragile"
    return out


# ---------------------------------------------------------------- public
def analyze(family, inst, base, solve, budget_s=2.5, trials=TRIALS, eps=NOISE, seed=11):
    """solve(family, instance) -> verified solution dict (the caller passes pipeline.solve_instance with quantum off)."""
    t0 = time.time(); deadline = t0 + budget_s; labels = _labels(family, inst); skipped = []; out = {"noise_pct": round(eps * 100)}
    lev, sk = _levers(family, inst, base, solve, labels, t0 + budget_s * .3)
    if lev: out["what_if"] = lev
    skipped += sk
    if family in ("knapsack", "multi_dim_knapsack"):
        cc, sk = _closest_calls(family, inst, base, solve, labels, t0 + budget_s * .6); skipped += sk
        if cc: out["closest_calls"] = cc[:5]; out["closest_call_count_examined"] = len(cc)
    st = _stability(family, inst, base, solve, labels, deadline, trials, eps, seed)
    if st: out["stability"] = st
    if skipped: out["skipped"] = skipped[:10]
    out["seconds"] = round(time.time() - t0, 2); return out


def narrative(family, base, an, headline, illustrative=False):
    """Plain-language sections built only from numbers in the analysis. Returns a list of {heading, text}."""
    S = []; obj = float(base["objective"]); lab = base.get("objective_label", "objective")
    why = f"Method: {base['method']}. Guarantee: {base['guarantee']}. {len(base['checks'])} of {len(base['checks'])} checks passed" + (", including a second independent method" if any(w in c["name"] for c in base["checks"] for w in ("Independent", "exhaustive", "random starts")) else "") + "." if base.get("verified") else f"Method: {base['method']}. At least one check failed, so this answer is not presented as verified."
    if base.get("bound") and base["bound"].get("gap_pct") is not None: why += f" It is within {base['bound']['gap_pct']}% of the best possible {lab} ({base['bound']['kind']})."
    S.append({"heading": "Why this answer", "text": why})
    st = an.get("stability")
    if st and "decision_unchanged_pct" not in st and "objective_range" in st:
        S.append({"heading": "How firm it is", "text": f"If {st['perturbed']} were off by up to {st['noise_pct']}% in either direction, the best {lab} would likely land between {st['objective_range']['p5']:g} and {st['objective_range']['p95']:g} ({st['trials']} re-solves). Many different splits tie for the best result, so how often the same split returns is not reported."})
    elif st and "decision_unchanged_pct" in st:
        t = f"If {st['perturbed']} were off by up to {st['noise_pct']}% in either direction, the same decision came back in {st['decision_unchanged_pct']:g}% of {st['trials']} re-solves, so it is {st['robustness']}" + (f" ({st['decision_unchanged_means']})" if st.get("decision_unchanged_means") else "") + f". The {lab} would likely land between {st['objective_range']['p5']:g} and {st['objective_range']['p95']:g}."
        if st.get("always_chosen"): t += " Always chosen: " + ", ".join(st["always_chosen"][:8]) + "."
        if st.get("toss_ups"): t += " Toss-ups: " + ", ".join(f"{x['item']} ({x['chosen_pct']}%)" for x in st["toss_ups"][:5]) + "."
        r = st.get("cost_of_keeping_original_choice_pct")
        if r: t += f" Acting on the original choice would have cost {r['mean']:g}% on average (worst {r['worst']:g}%) against re-optimising with the new numbers."
        S.append({"heading": "How firm it is", "text": t})
    elif st: S.append({"heading": "How firm it is", "text": st["note"][0].upper() + st["note"][1:] + "."})
    cc = an.get("closest_calls")
    if cc: S.append({"heading": "Closest calls", "text": "Reversing these costs least, so check your estimates for them first: " + "; ".join(f"{r['item']} ({r['currently']}; reversing costs {r['cost_of_reversing']:g}, {r['cost_pct']:g}%)" for r in cc[:3]) + "."})
    wi = [w for w in an.get("what_if", []) if "delta" in w]
    if wi:
        best = max(wi, key=lambda w: w["delta"]); worst = min(wi, key=lambda w: w["delta"])
        t = f"Largest gain among the changes tried: {best['change']} ({'+' if best['delta'] >= 0 else ''}{best['delta']:g} {lab}"
        t += f", {best['delta_pct']:+g}%)" if "delta_pct" in best else ")"
        t += "."
        if worst is not best: t += f" Largest loss: {worst['change']} ({worst['delta']:+g})."
        S.append({"heading": "What would change it", "text": t})
    care = []
    if illustrative: care.append("The numbers are generated for illustration, not yours; do not act on this result.")
    if base.get("guarantee", "").startswith(("Feasible", "At least", "Heuristic")) or "heuristic" in base.get("guarantee", "").lower(): care.append("This method does not prove optimality; treat the result as a strong candidate.")
    care.append("Re-solve whenever an estimate changes materially; the answer is only as good as the numbers supplied.")
    S.append({"heading": "Before you act", "text": " ".join(care)})
    return S
