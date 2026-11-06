"""Original, project-written problems (data/authored_problems.py). Train slice is added to the dataset with source=authored and is never scored in cross-validation.
The held-out slice is never trained on.   python -m app.authored apply | evaluate"""
import sys, json
import numpy as np, joblib
from pathlib import Path
from . import train
from .labels import read, write, BASE, CUSTOM
sys.path.insert(0, str(train.ROOT))
from data.authored_problems import split

def apply():
    rows = read(CUSTOM if CUSTOM.exists() else BASE); rows = [r for r in rows if r.get("source") != "authored"]; n = max(int(r["id"][2:]) for r in rows if r["id"].startswith("BP"))
    for t in split()[0]:
        n += 1; rows.append({"id": f"BP{n:04d}", "domain": t["domain"], "problem": t["problem"], "decision_type": "", "objective": "", "key_constraints": "", "typical_variables": "",
                             "formulation_family": t["formulation_family"], "trend_sensitivity": "", "formulation_family_original": t["formulation_family"], "label_review_rule": "authored for this project; rubric label", "source": "authored"})
    write(rows, CUSTOM); print(f"{len(rows)} rows in dataset ({sum(r.get('source') == 'authored' for r in rows)} authored, training only)")

def evaluate():
    card = json.loads((train.MODELS / "card.json").read_text()); m = joblib.load(train.MODELS / f"family_v{card['version']}.joblib")["model"]; ho = split()[1]
    pr = m.predict_proba([h["problem"] for h in ho]); cl = m.classes_; pred = cl[pr.argmax(1)]; truth = np.array([h["formulation_family"] for h in ho])
    gs, gp = train.group_probs(cl, pr); gpred = np.array(gs)[gp.argmax(1)]; gt = np.array([train.GROUP[t] for t in truth]); gc = gp.max(1)
    out = {"model_version": card["version"], "n": len(ho), "nine_way": round(float((pred == truth).mean()), 3), "broad": round(float((gpred == gt).mean()), 3), "broad_ci95": train.wilson(int((gpred == gt).sum()), len(ho))}
    for th in (.6, .8):
        k = gc >= th; out[f"broad_at_{th}"] = {"accuracy": round(float((gpred[k] == gt[k]).mean()), 3), "coverage": round(float(k.mean()), 3), "answered": int(k.sum())}
    print(json.dumps(out, indent=1)); return out
if __name__ == "__main__": {"apply": apply, "evaluate": evaluate}[sys.argv[1]]()
