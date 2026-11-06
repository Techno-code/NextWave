"""Ingest books, reports and open datasets.   python -m app.ingest --help

  python -m app.ingest build <folder>      read every file in the folder, write the private/shared retrieval corpus and a review queue
  python -m app.ingest merge <queue.csv>   add reviewed rows from the queue to the training dataset, then run: python -m app.train

The folder may hold a manifest named sources.csv with columns: file,title,license,allow_training,source_url
  - Every readable file goes into the retrieval corpus, with its title and licence shown in citations.
  - Only files with allow_training=yes (and a licence that permits it) contribute candidate problems for training.
  - Files without a manifest row are retrieval-only. Nothing is ever used for training without a person approving the row in the queue."""
import csv, io, json, re, sys, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"; CORPUS = DATA / "corpus"; QUEUE = DATA / "review_queue.csv"; STATUS = CORPUS / "status.csv"
EXTS = (".pdf", ".epub", ".docx", ".txt", ".md", ".csv", ".json", ".xlsx")
OK_LICENSES = re.compile(r"cc0|public domain|cc[- ]by(?![- ]n[cd])|mit|apache|bsd|odc-by|odbl", re.I)  # allows CC BY / CC BY-SA, not NC or ND

def read_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    if ext == ".docx":
        import docx
        return "\n\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    if ext == ".epub":
        z = zipfile.ZipFile(path)
        return "\n\n".join(re.sub(r"<[^>]+>", " ", z.read(n).decode("utf-8", "replace")) for n in z.namelist() if n.endswith(("html", "xhtml")))
    if ext == ".xlsx":
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        return "\n".join(", ".join("" if c is None else str(c) for c in r) for ws in wb for r in ws.iter_rows(values_only=True))
    return path.read_text(encoding="utf-8", errors="replace")

def passages(text, size=600):
    text = re.sub(r"[ \t]+", " ", text); out = []
    for s in re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z])", text):
        s = s.strip()
        if not s: continue
        if out and len(out[-1]) < size: out[-1] += " " + s
        else: out.append(s)
    return [p for p in out if len(p) > 40]

VERB = r"(?:Choose|Select|Decide|Allocate|Assign|Schedule|Plan|Determine|Pick|Split|Distribute|Sequence|Route|Locate|Prioriti[sz]e|Match|Staff|Set|Order)"
PATTERNS = [re.compile(rf"^{VERB}\b[^.?!]{{15,140}}[.?]?$"), re.compile(r"^(?:How (?:should|much|many)|Which|Where should|When should|What should)\b[^.?!]{15,140}\?$"),
            re.compile(r"\b(?:must|needs? to|has to|have to) (?:decide|choose|select|allocate|assign|schedule) [^.?!]{10,120}[.?]")]
def candidates(text, cap=200):
    seen, out = set(), []
    for s in re.split(r"(?<=[.?!])\s+|\n+", text):
        s = re.sub(r"\s+", " ", s).strip(" -*\u2022")
        if 25 <= len(s) <= 170 and s.lower() not in seen and any(p.search(s) for p in PATTERNS):
            seen.add(s.lower()); out.append(s)
        if len(out) >= cap: break
    return out

def manifest(folder: Path):
    f = folder / "sources.csv"
    if not f.exists(): return {}
    return {r["file"].strip(): {k: (v or "").strip() for k, v in r.items()} for r in csv.DictReader(open(f, encoding="utf-8-sig"))}

def slug(s): return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "doc"

def build(folder):
    folder = Path(folder); man = manifest(folder); CORPUS.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS and p.name != "sources.csv")
    status, queue = [], []
    model = None
    try:
        from . import train; import joblib
        card = json.loads((train.MODELS / "card.json").read_text()); model = joblib.load(train.MODELS / f"family_v{card['version']}.joblib")["model"]
    except Exception: model = None
    for p in files:
        m = man.get(p.name, {}); title = m.get("title") or p.stem; lic = m.get("license") or "unspecified"; allow = m.get("allow_training", "").lower() == "yes" and bool(OK_LICENSES.search(lic))
        try:
            text = read_text(p); ps = passages(text)
            if not ps: raise ValueError("no readable text")
            with open(CORPUS / f"{slug(p.stem)}.jsonl", "w", encoding="utf-8") as f:
                for i, t in enumerate(ps, 1): f.write(json.dumps({"doc": title, "license": lic, "n": i, "text": t, "source_url": m.get("source_url", "")}) + "\n")
            nc = 0
            if allow:
                for s in candidates(text):
                    fam, conf = "", ""
                    if model is not None:
                        pr = model.predict_proba([s])[0]; i = int(pr.argmax()); fam, conf = str(model.classes_[i]), round(float(pr[i]), 2)
                    queue.append({"id": "", "domain": "", "problem": s, "suggested_family": fam, "suggested_p": conf, "reviewed_family": "", "source": title, "license": lic}); nc += 1
            status.append({"file": p.name, "status": "ready", "passages": len(ps), "training_candidates": nc, "license": lic, "training_allowed": "yes" if allow else "no", "reason": "" if allow or not m.get("allow_training") else "licence does not permit training"})
        except Exception as e:
            status.append({"file": p.name, "status": "failed", "passages": 0, "training_candidates": 0, "license": lic, "training_allowed": "no", "reason": str(e)[:200]})
    with open(STATUS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["file", "status", "passages", "training_candidates", "license", "training_allowed", "reason"]); w.writeheader(); w.writerows(status)
    if queue:
        with open(QUEUE, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, list(queue[0])); w.writeheader(); w.writerows(queue)
    for s in status: print(f"{s['status']:7} {s['file']}  passages={s['passages']} candidates={s['training_candidates']} licence={s['license']} {s['reason']}")
    print(f"Corpus written to {CORPUS}. Review queue: {len(queue)} candidate problems" + (f" in {QUEUE}. Fill reviewed_family (and domain) for rows you approve, then run: python -m app.ingest merge {QUEUE}" if queue else "."))
    return status, queue

def merge(path):
    from .train import GROUP
    from .labels import read, write, BASE, CUSTOM
    rows = read(CUSTOM if CUSTOM.exists() else BASE); have = {r["problem"].lower() for r in rows}; n = max([int(re.sub(r"\D", "", r["id"]) or 0) for r in rows] + [0]); added = skipped = 0
    for q in read(path):
        fam = q.get("reviewed_family", "")
        if not fam: continue
        if fam not in GROUP or q["problem"].lower() in have: skipped += 1; continue
        n += 1; have.add(q["problem"].lower())
        rows.append({"id": f"BP{n:04d}", "domain": q.get("domain", "") or "Corpus", "problem": q["problem"], "decision_type": "", "objective": "", "key_constraints": "", "typical_variables": "",
                     "formulation_family": fam, "trend_sensitivity": "", "formulation_family_original": fam, "label_review_rule": f"human-reviewed from {q.get('source', '')} ({q.get('license', '')})"}); added += 1
    write(rows, CUSTOM); print(f"Added {added} reviewed rows, skipped {skipped} (unknown family or duplicate). Now run: python -m app.train")

if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 2 and a[0] == "build": build(a[1])
    elif len(a) == 2 and a[0] == "merge": merge(a[1])
    else: print(__doc__)
