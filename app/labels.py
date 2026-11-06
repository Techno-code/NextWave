"""Applies the reviewed labels in data/label_review.csv. Originals are kept, never overwritten.
Run: python -m app.labels apply     (writes data/custom_dataset.csv, which the app loads first)
     python -m app.labels report    (shows how many labels changed, by broad kind)"""
import csv, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data" / "business_problems_dataset.csv"; CUSTOM = ROOT / "data" / "custom_dataset.csv"; REVIEW = ROOT / "data" / "label_review.csv"

def read(path): return [{k.strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(open(path, encoding="utf-8-sig"))]

def apply_reviews(rows, reviews):
    out = []
    for r in rows:
        r = dict(r); rv = reviews.get(r.get("id"))
        r.setdefault("formulation_family_original", r.get("formulation_family", ""))
        if not r["formulation_family_original"]: r["formulation_family_original"] = r.get("formulation_family", "")
        if rv: r["formulation_family"] = rv["reviewed"]; r["label_review_rule"] = rv["rule"]
        out.append(r)
    return out

def write(rows, path):
    cols = list(dict.fromkeys(k for r in rows for k in r)); 
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def main(cmd):
    from .train import GROUP
    reviews = {r["id"]: r for r in read(REVIEW)}; rows = apply_reviews(read(CUSTOM if CUSTOM.exists() else BASE), reviews)
    for rid, rv in reviews.items():
        assert rv["reviewed"] in GROUP, f"{rid}: unknown family {rv['reviewed']}"
    changed = [r for r in rows if r["formulation_family"] != r["formulation_family_original"]]
    cross = [r for r in changed if GROUP[r["formulation_family"]] != GROUP[r["formulation_family_original"]]]
    print(f"{len(changed)} of {len(rows)} labels changed ({len(cross)} change the broad kind, {len(changed) - len(cross)} change only the variant).")
    if cmd == "apply": write(rows, CUSTOM); print(f"Wrote {CUSTOM}")
if __name__ == "__main__": main(sys.argv[1] if len(sys.argv) > 1 else "report")
