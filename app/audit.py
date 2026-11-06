"""python -m app.audit  Lists problems that read almost the same but carry different labels.
These conflicts are the main limit on how accurate any text model can be on this dataset."""
import csv
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
ROOT = Path(__file__).resolve().parent.parent
def main(th=.35):
    f = ROOT / "data" / "custom_dataset.csv"; f = f if f.exists() else ROOT / "data" / "business_problems_dataset.csv"
    rows = [{k.strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(open(f, encoding="utf-8-sig"))]
    X = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit_transform([r["problem"] + " " + r.get("decision_type", "") for r in rows]); S = (X @ X.T).toarray()
    pairs = sorted(((S[i, j], i, j) for i in range(len(rows)) for j in range(i + 1, len(rows)) if S[i, j] >= th and rows[i]["formulation_family"] != rows[j]["formulation_family"]), reverse=True)
    inv = {k for _, i, j in pairs for k in (i, j)}
    print(f"{len(pairs)} conflicting pairs involving {len(inv)} of {len(rows)} rows (similarity >= {th}). Review these labels:")
    for s, i, j in pairs[:15]: print(f"  {s:.2f}  {rows[i]['id']} {rows[i]['problem'][:46]} [{rows[i]['formulation_family']}]  vs  {rows[j]['id']} {rows[j]['problem'][:46]} [{rows[j]['formulation_family']}]")
if __name__ == "__main__": main()
