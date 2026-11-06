"""Option-selection solvers used by /api/v1/reason and by the evaluation harness.
Kept free of web and database imports so that the harness can call exactly the code that serves requests."""
import random, re

HIGH = re.compile(r"legal|lawsuit|contract|medical|patient|hospital|surgeon|diagnos|tax|loan|invest|mental|safety|regulat", re.I)
NUM = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"       # 1,200 and 1200.50 are numbers; "4, value" is a 4 followed by a separator
OPT = re.compile(r"^[-*\s]*(.+?)\s*[:|]\s*cost\s*\$?" + NUM + r"\s*[,;|]?\s*value\s*\$?" + NUM, re.I | re.M)
BUDGET = re.compile(r"budget\s*(?:of|is|:)?\s*\$?" + NUM, re.I)
LOOKS_LIKE_OPTION = re.compile(r"\bcost\b.*\bvalue\b", re.I)
num = lambda x: float(x.replace(",", ""))

def parse(t):
    """Reads 'Budget of N' and lines like '- Name: cost X value Y'. Names may contain colons, pipes and commas; numbers may use thousands separators.
    'unread' lists lines that look like options but could not be read, so that callers can say so instead of silently leaving them out."""
    m = BUDGET.search(t)
    return {"goal": re.split(r"[.\n]", t)[0][:200], "stakes": "High" if HIGH.search(t) else "Medium",
            "opts": [{"n": a.strip(), "c": num(b), "v": num(c)} for a, b, c in OPT.findall(t)],
            "budget": num(m.group(1)) if m else None,
            "unread": [l.strip()[:120] for l in t.split("\n") if LOOKS_LIKE_OPTION.search(l) and not OPT.match(l)]}

def tot(o, m, k): return sum(x[k] for i, x in enumerate(o) if m >> i & 1)
def names(o, m): return ", ".join(x["n"] for i, x in enumerate(o) if m >> i & 1) or "nothing"
def knap(o, B):
    best = None
    for m in range(1 << len(o)):
        c, v = tot(o, m, "c"), tot(o, m, "v")
        if c <= B + 1e-9 and (best is None or v > best[1]): best = (m, v, c)
    return best
def dp(o, B):
    sc = 1 if all(float(x["c"]).is_integer() for x in o) and float(B).is_integer() else 100
    W = round(B * sc)
    if W > 200000: return None
    c = [round(x["c"] * sc) for x in o]; tab = []; d = [0.0] * (W + 1)
    for i, x in enumerate(o):
        nd = d[:]
        for w in range(c[i], W + 1): nd[w] = max(nd[w], d[w - c[i]] + x["v"])
        tab.append(nd); d = nd
    w, m = W, 0
    for i in range(len(o) - 1, -1, -1):
        prev = tab[i - 1] if i else [0.0] * (W + 1)
        if tab[i][w] != prev[w]: m |= 1 << i; w -= c[i]
    return d[W], m

def greedy(o, B):
    """Value-per-cost heuristic. Returns (mask, value)."""
    w = m = 0
    for i in sorted(range(len(o)), key=lambda i: -o[i]["v"] / o[i]["c"] if o[i]["c"] else -1e18):
        if w + o[i]["c"] <= B: w += o[i]["c"]; m |= 1 << i
    return m, tot(o, m, "v")

def reason_knapsack(text, depth="Standard"):
    """The full /reason pipeline. Raises ValueError when the text is not a budgeted option list."""
    p = parse(text); o, B = p["opts"], p["budget"]
    if len(o) < 2 or B is None or len(o) > 16: raise ValueError("Need a budget and 2 to 16 options like '- Name: cost X value Y'.")
    tr = []; ex = knap(o, B); tr.append(f"Exhaustive search: {names(o, ex[0])}; value {ex[1]:g}")
    d2 = None; crit = []; stab = None; conf = "high"; esc_ = False
    if depth != "Quick":
        d2 = dp(o, B); tr.append(f"Dynamic programming: value {d2[0]:g}" if d2 else "Dynamic programming skipped")
        gm, gv = greedy(o, B); crit.append(f"Heuristic loses {ex[1] - gv:g}." if gv < ex[1] else "Heuristic matched the optimum.")
        if d2 and abs(d2[0] - ex[1]) > 1e-9: esc_, depth = True, "Deep"; tr.append("Methods disagree; escalating to Deep.")
    if depth == "Deep" and len(o) <= 14:
        R = random.Random(3); same = 0
        for _ in range(300):
            nv = [{**x, "v": x["v"] * (1 + (R.random() - .5) * .4)} for x in o]
            same += knap(nv, B)[0] == ex[0]
        stab = same / 300; tr.append(f"300 simulations: same choice {stab:.0%}")
        for d in (-1, 1):
            k = knap(o, B + d)
            if k and k[0] != ex[0]: crit.append(f"With budget {B + d:g} the best set changes to {names(o, k[0])}.")
        if stab < .7: conf = "medium"
    warnings = [f"{len(p['unread'])} line(s) looked like options but could not be read, so they were left out: " + "; ".join(p["unread"][:3])] if p["unread"] else []
    if warnings: conf = "provisional"
    checks = {"budget": tot(o, ex[0], "c") <= B + 1e-9, "value_recomputed": abs(tot(o, ex[0], "v") - ex[1]) < 1e-9, "independent_agrees": d2 is None or abs(d2[0] - ex[1]) < 1e-9}
    if not all(checks.values()): conf = "provisional"
    return {"answer": f"Choose {names(o, ex[0])}. Cost {ex[2]:g} of {B:g}; value {ex[1]:g}.", "confidence": conf, "depth": depth, "escalated": esc_, "stability": stab,
            "checks": checks, "critic": crit, "trace": tr, "warnings": warnings, "chosen_mask": ex[0], "value": ex[1], "cost": ex[2]}
