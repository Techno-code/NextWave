"""Trains the problem-type classifier from the dataset plus opted-in user corrections.
Run: python -m app.train      Reports cross-validated accuracy against two baselines."""
import hashlib, json, math, random, time
import os
from pathlib import Path
import joblib, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline, make_union

from .synth import synthetic
GROUP = {"knapsack": "select", "multi_dim_knapsack": "select", "portfolio_selection": "select", "set_cover": "select", "budget_allocation": "allocate", "assignment": "assign", "scheduling": "schedule", "facility_location": "locate", "routing": "route"}
ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(os.environ["DE_MODELS"]) if os.environ.get("DE_MODELS") else ROOT / "models"
DISTRACT = ["The team meets on Tuesday.", "Our office moved last year.", "Leadership wants an answer soon.", "Figures are in thousands."]

def views(r, rnd):
    p = r["problem"]
    out = [p, f"{p} {r.get('decision_type', '')}", f"{p} {r.get('objective', '')}", f"{r.get('domain', '')}: {p} {r.get('key_constraints', '')}"]
    for _ in range(3): out.append(" ".join(w for w in p.split() if rnd.random() > .2))
    out.append(p + " " + rnd.choice(DISTRACT))
    return out  # index 0 is the plain problem text, used for evaluation

def pipe():
    feats = make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True))
    return make_pipeline(feats, LogisticRegression(C=10, max_iter=3000))

def ece(conf, ok, bins=10):
    conf, ok = np.array(conf), np.array(ok, float); e = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins)
        if m.any(): e += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return float(e)

def group_probs(classes, pr):
    gs = sorted(set(GROUP.values())); return gs, np.array([[sum(p[i] for i, c in enumerate(classes) if GROUP[c] == g) for g in gs] for p in pr])

def wilson(k, n, z=1.96):
    if not n: return None
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 3), round((c + h) / d, 3)]

def selective(conf, ok, ths):
    conf, ok = np.array(conf), np.array(ok, float); out = []
    for t in ths:
        m = conf >= t; n = int(m.sum()); k = int(ok[m].sum()) if n else 0
        out.append({"threshold": t, "coverage": round(float(m.mean()), 3), "accuracy": round(k / n, 3) if n else None, "answered": n, "ci95": wilson(k, n)})
    return out

def operating_point(rows, target=.9):
    """Lowest confidence threshold whose measured accuracy reaches the target; the engine states a problem kind only above it."""
    for r in rows:
        if r["accuracy"] is not None and r["accuracy"] >= target and r["answered"] >= 30: return {**r, "target": target}
    return None

def build(rows, feedback, n_synth, key):
    rnd = random.Random(42); X, y, g, orig = [], [], [], []
    for gi, r in enumerate(rows):
        for k, t in enumerate(views(r, rnd)): X.append(t); y.append(r[key]); g.append(gi); orig.append(k == 0 and r.get("source") != "authored")
    base = len(rows)
    for j, (t, fam) in enumerate(feedback): X.append(t); y.append(fam); g.append(base + j); orig.append(False)
    for j, (t, fam) in enumerate(synthetic(n_synth)): X.append(t); y.append(fam); g.append(10 ** 6 + j); orig.append(False)
    return np.array(X, dtype=object), np.array(y), np.array(g), np.array(orig)

def cross_validate(X, y, g, orig):
    res = _cv(X, y, g, orig); res["operating_point"] = operating_point(res["group_confident"]); return res

def _cv(X, y, g, orig):
    acc, ret, maj, conf, ok, gacc, gconf, gmaj, top2, top3 = [], [], [], [], [], [], [], [], [], []
    for tr, te in GroupKFold(5).split(X, y, g):
        m = pipe().fit(X[tr], y[tr]); te = te[orig[te]]
        if not len(te): continue
        pr = m.predict_proba(X[te]); pred = m.classes_[pr.argmax(1)]
        acc += list(pred == y[te]); conf += list(pr.max(1)); ok += list(pred == y[te])
        rk = m.classes_[pr.argsort(1)[:, ::-1]]; top2 += [y[te][i] in rk[i, :2] for i in range(len(te))]; top3 += [y[te][i] in rk[i, :3] for i in range(len(te))]
        gs, gp = group_probs(m.classes_, pr); truth = np.array([GROUP[v] for v in y[te]]); gok = np.array(gs)[gp.argmax(1)] == truth
        gacc += list(gok); gconf += list(gp.max(1)); gmaj += list(truth == max(set(GROUP[v] for v in y[tr[orig[tr]]]), key=[GROUP[v] for v in y[tr[orig[tr]]]].count))
        o = tr[orig[tr]]; v = TfidfVectorizer(sublinear_tf=True).fit(X[o]); sim = (v.transform(X[te]) @ v.transform(X[o]).T).toarray()
        ret += list(y[o][sim.argmax(1)] == y[te]); maj += list(y[te] == max(set(y[tr]), key=list(y[tr]).count))
    return {"cv_accuracy": round(float(np.mean(acc)), 3), "nearest_row_baseline": round(float(np.mean(ret)), 3), "majority_baseline": round(float(np.mean(maj)), 3),
            "calibration_error_ece": round(ece(conf, ok), 3), "top2_accuracy": round(float(np.mean(top2)), 3), "top3_accuracy": round(float(np.mean(top3)), 3), "group_accuracy": round(float(np.mean(gacc)), 3), "group_majority_baseline": round(float(np.mean(gmaj)), 3),
            "group_confident": selective(gconf, gacc, [.5, .6, .7, .8]), "operating_point": None, "fine_confident": selective(conf, ok, [.5, .6, .7]), "folds": 5, "held_out_problems": len(acc)}

NOTES = ("Folds hold out whole real problems, so scores measure unseen problems; the expert-written training sentences are never used for scoring. "
         "Primary metrics use the reviewed labels (docs/LABEL_RUBRIC.md, data/label_review.csv). original_label_metrics use the labels exactly as supplied. "
         "The reviewed-label score was produced after the first accuracy run by a single reviewer, so it is a development number; an independent relabelling of a random sample is still required.")

def train(rows, feedback=(), n_synth=300):
    rows = [r for r in rows if r.get("formulation_family") in GROUP]
    X, y, g, orig = build(rows, feedback, n_synth, "formulation_family")
    metrics = cross_validate(X, y, g, orig)
    original = None
    if any(r.get("formulation_family_original") for r in rows):
        rr = [{**r, "formulation_family_original": r.get("formulation_family_original") or r["formulation_family"]} for r in rows]
        rr = [r for r in rr if r["formulation_family_original"] in GROUP]
        Xo, yo, go, oo = build(rr, feedback, n_synth, "formulation_family_original"); original = cross_validate(Xo, yo, go, oo)
    model = pipe().fit(X, y)
    MODELS.mkdir(exist_ok=True)
    prev = json.loads((MODELS / "card.json").read_text())["version"] if (MODELS / "card.json").exists() else 0
    card = {"version": prev + 1, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "dataset_rows": len(rows), "feedback_examples": len(feedback),
            "dataset_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16], "families": sorted(set(y)),
            "metrics": metrics, "group_map": GROUP, "synthetic_examples": n_synth * 9, "notes": NOTES}
    if original: card["original_label_metrics"] = original
    joblib.dump({"model": model, "card": card}, MODELS / f"family_v{card['version']}.joblib")
    (MODELS / "card.json").write_text(json.dumps(card, indent=1)); return model, card

if __name__ == "__main__":
    import csv
    f = ROOT / "data" / "custom_dataset.csv"; f = f if f.exists() else ROOT / "data" / "business_problems_dataset.csv"
    rows = [{k.strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(open(f, encoding="utf-8-sig"))]
    print(json.dumps(train(rows)[1], indent=1))
