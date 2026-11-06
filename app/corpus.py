"""Shared, operator-supplied corpus (open-licence books and reports) for retrieval and citations.
Built by `python -m app.ingest build`. Passages live in data/corpus/*.jsonl. Per-user private documents stay in the database and are searched separately."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "data" / "corpus"
_cache = {"sig": None}

def _load():
    files = sorted(DIR.glob("*.jsonl")) if DIR.exists() else []
    sig = tuple((f.name, f.stat().st_mtime_ns) for f in files)
    if _cache["sig"] == sig: return _cache
    P = [json.loads(l) for f in files for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    _cache.update(sig=sig, P=P, vec=None, M=None)
    if P:
        from sklearn.feature_extraction.text import TfidfVectorizer
        v = TfidfVectorizer(sublinear_tf=True, stop_words="english", ngram_range=(1, 2), min_df=1); _cache.update(vec=v, M=v.fit_transform([p["text"] for p in P]))
    return _cache

def search(query, k=2, min_score=.12):
    """Returns [(score, passage)] best first. Passages carry doc title, licence and passage number so every citation can be traced."""
    c = _load()
    if not c.get("P") or not query.strip(): return []
    s = (c["vec"].transform([query]) @ c["M"].T).toarray()[0]
    return [(float(s[i]), c["P"][i]) for i in s.argsort()[::-1][:k] if s[i] >= min_score]
