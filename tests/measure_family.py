"""Measures how well each method names the problem family on statements held out of every training step.
Run:  python tests/measure_family.py      (needs data/bigdata.sqlite for the retrieval and fused rows)
Held-out set: every third authored statement (index 2, 5, 8, ...), as stated in data/authored_problems.py. 45 statements, nine families.
The blend row records a design that was tried first and rejected because it scored below the classifier; the engine uses the classifier."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import bigdata, main, train
from data import authored_problems as A

def fuse(model, vote):
    """The equal-weight blend that was tried and rejected. Kept here only so that its measured score stays reproducible."""
    s = {}
    for x in model: s[x["family"]] = s.get(x["family"], 0) + .5 * x["p"]
    for f, p in vote.items(): s[f] = s.get(f, 0) + .5 * p
    z = sum(s.values()) or 1; return [{"family": f, "score": v / z} for f, v in sorted(s.items(), key=lambda x: -x[1])]
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return round((c - h) / d, 3), round((c + h) / d, 3)
held = [(fam, txt) for fam, items in A.P.items() for i, (_, txt) in enumerate(items) if i % 3 == 2]
res = {"classifier": 0, "retrieval vote": 0, "blend (rejected design)": 0, "classifier top-3": 0}; n = len(held); idx = bigdata.available()
if not idx: res.pop("retrieval vote"); res.pop("blend (rejected design)")
miss = []
for fam, txt in held:
    m = main.predict_family(txt, 3); res["classifier"] += m[0]["family"] == fam; res["classifier top-3"] += any(x["family"] == fam for x in m)
    if idx:
        v = bigdata.family_vote(txt); res["retrieval vote"] += bool(v) and next(iter(v)) == fam
        f = fuse(m, v); ok = bool(f) and f[0]["family"] == fam; res["blend (rejected design)"] += ok
        if not ok: miss.append((fam, f[0]["family"] if f else None, txt))
print(f"Held-out authored statements: {n}")
for k, v in res.items(): lo, hi = wilson(v, n); print(f"  {k:24s} {v}/{n} = {v / n:.1%}  (95% interval {lo:.1%} to {hi:.1%})")
for fam, got, txt in miss[:12]: print(f"  fused miss: true={fam} got={got}: {txt}")
