"""Knowledge layer: the 1M-problem catalogue, the 108 solution playbooks, the theorem table and the reference list.

Small tables (playbooks, theorems, references) ship in data/knowledge/ and are always available.
The 1M-row catalogue is indexed into data/bigdata.sqlite by:
    python -m app.bigdata build business_problems_1M.csv business_problem_solutions_1M.csv
The index keeps one row per problem (text, domain, decision type, objective, limits, variables, family, sensitivity, playbook id) and a
full-text index over the problem text. If the index has not been built the engine still runs; catalogue lookups then report that plainly.
The solutions file holds a playbook id, algorithm and model text per problem and no numeric answers (every row says needs_instance_data),
so numbers only ever come from instance data the user supplies or from a labelled illustrative instance."""
import csv, os, re, sqlite3, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KNOW = ROOT / "data" / "knowledge"
INDEX = Path(os.environ["DE_BIGDATA"]) if os.environ.get("DE_BIGDATA") else ROOT / "data" / "bigdata.sqlite"
FAMILIES = ["knapsack", "multi_dim_knapsack", "portfolio_selection", "budget_allocation", "scheduling", "assignment", "facility_location", "set_cover", "routing"]

def _read(name):
    with open(KNOW / name, newline="", encoding="utf-8") as f: return list(csv.DictReader(f))
PLAYBOOKS = {r["playbook_id"]: r for r in _read("solution_playbooks.csv")}
THEOREMS = {r["theorem_id"]: r for r in _read("business_theorems.csv")}
REFERENCES = {r["ref_id"]: r for r in _read("research_references.csv")}

def scale_bucket(n):
    """Size bucket used by the playbooks, chosen from the number of decision items in the instance."""
    return "small" if n <= 20 else "medium" if n <= 200 else "large" if n <= 5000 else "xlarge"

def playbook_for(family, n_items=20, sensitivity="medium"):
    key = {"low": "L", "medium": "M", "high": "H"}.get(sensitivity, "M")
    hits = [p for p in PLAYBOOKS.values() if p["formulation_family"] == family and p["scale_bucket"] == scale_bucket(n_items) and p["trend_sensitivity"] == sensitivity]
    return hits[0] if hits else next((p for p in PLAYBOOKS.values() if p["formulation_family"] == family), None)

def expand(pb):
    """A playbook with its theorems and references written out, so every claim can be traced to a row."""
    if not pb: return None
    out = dict(pb)
    out["steps"] = [s.strip() for s in pb["implementation_steps"].split("|")]
    out["theorems"] = [{k: THEOREMS[t][k] for k in ("theorem_id", "name", "statement", "business_use")} for t in pb["core_theorem_ids"].split(";") if t in THEOREMS]
    out["references"] = [{k: REFERENCES[r][k] for k in ("ref_id", "authors", "year", "title", "venue", "key_result")} for r in pb["core_ref_ids"].split(";") if r in REFERENCES]
    return out

# ---------- the 1M catalogue ----------
_local = threading.local()
def available(): return INDEX.exists()
def _db():
    if not hasattr(_local, "c"): _local.c = sqlite3.connect(f"file:{INDEX}?mode=ro", uri=True)
    return _local.c

def build(problems_csv, solutions_csv, out=None, batch=50000, log=print):
    out = Path(out or INDEX); tmp = out.with_suffix(".tmp")
    if tmp.exists(): tmp.unlink()
    db = sqlite3.connect(tmp); t0 = time.time()
    db.executescript("""pragma journal_mode=off; pragma synchronous=off;
    create table dim(kind text, id integer, val text, primary key(kind, id));
    create table p(n integer primary key, domain integer, dtype integer, obj integer, cons integer, vars integer, fam integer, sens integer, pb integer, problem text);""")
    dims = {k: {} for k in ("domain", "dtype", "obj", "cons", "vars", "fam", "sens", "pb")}
    def d(kind, v):
        m = dims[kind]
        if v not in m: m[v] = len(m)
        return m[v]
    with open(problems_csv, newline="", encoding="utf-8") as fp, open(solutions_csv, newline="", encoding="utf-8") as fs:
        rp, rs = csv.DictReader(fp), csv.DictReader(fs); rows = []; n = 0
        for a, b in zip(rp, rs):
            if a["id"] != b["id"]: raise ValueError(f"row {n + 2}: ids differ ({a['id']} vs {b['id']}); the two files must be in the same order")
            n += 1
            rows.append((int(a["id"][2:]), d("domain", a["domain"]), d("dtype", a["decision_type"]), d("obj", a["objective"]), d("cons", a["key_constraints"]),
                         d("vars", a["typical_variables"]), d("fam", a["formulation_family"]), d("sens", a["trend_sensitivity"]), d("pb", b["playbook_id"]), a["problem"]))
            if len(rows) >= batch: db.executemany("insert into p values(?,?,?,?,?,?,?,?,?,?)", rows); rows = []; log(f"  {n:,} rows, {time.time() - t0:.0f}s")
        if rows: db.executemany("insert into p values(?,?,?,?,?,?,?,?,?,?)", rows)
    for k, m in dims.items(): db.executemany("insert into dim values(?,?,?)", [(k, i, v) for v, i in m.items()])
    db.executescript("create index p_fam on p(fam, sens); create index p_dom on p(domain, fam);")
    log("building full-text index"); db.executescript("create virtual table fts using fts5(problem, content='p', content_rowid='n', tokenize='porter unicode61'); insert into fts(fts) values('rebuild');")
    db.executescript("create table meta(k text, v text); insert into meta values('rows','%d'); insert into meta values('built','%s');" % (n, time.strftime("%Y-%m-%dT%H:%M:%S")))
    db.commit(); db.execute("vacuum"); db.close(); tmp.replace(out); log(f"done: {n:,} problems in {time.time() - t0:.0f}s -> {out} ({out.stat().st_size / 1e6:.0f} MB)"); return n

_dimcache = {}
def _dim(kind):
    if kind not in _dimcache: _dimcache[kind] = dict(_db().execute("select id,val from dim where kind=?", (kind,)).fetchall())
    return _dimcache[kind]

SELECT = "select n,domain,dtype,obj,cons,vars,fam,sens,pb,problem from p"
def _row(t):
    n, dom, dt, ob, co, va, fa, se, pb, pr = t
    return {"id": f"BP{n:07d}", "domain": _dim("domain")[dom], "problem": pr, "decision_type": _dim("dtype")[dt], "objective": _dim("obj")[ob], "key_constraints": _dim("cons")[co],
            "typical_variables": _dim("vars")[va], "formulation_family": _dim("fam")[fa], "trend_sensitivity": _dim("sens")[se], "playbook_id": _dim("pb")[pb]}

def stats():
    if not available(): return {"available": False, "note": "Catalogue index not built. Run: python -m app.bigdata build <problems.csv> <solutions.csv>"}
    db = _db(); rows = db.execute("select v from meta where k='rows'").fetchone()[0]; built = db.execute("select v from meta where k='built'").fetchone()[0]
    fam = {_dim("fam")[f]: c for f, c in db.execute("select fam,count(*) from p group by fam")}
    return {"available": True, "rows": int(rows), "built": built, "families": fam, "domains": len(_dim("domain"))}

def get(pid):
    """Look up one catalogue row by id such as BP0000123."""
    if not available(): return None
    m = re.fullmatch(r"(?:BP)?0*(\d{1,9})", str(pid).strip(), re.I)
    if not m: return None
    t = _db().execute(SELECT + " where n=?", (int(m.group(1)),)).fetchone(); return _row(t) if t else None

def search(text, k=5, family=None):
    """Nearest catalogue problems by full-text rank (BM25). Returns [] when the index is absent or nothing matches."""
    if not available(): return []
    words = [w for w in re.findall(r"[A-Za-z]{3,}", text.lower())][:24]
    if not words: return []
    stop = {"the", "and", "for", "with", "that", "this", "our", "are", "has", "have", "how", "what", "which", "should", "from", "into", "over", "per", "not"}
    words = [w for w in words if w not in stop] or words
    qs = " OR ".join(f'"{w}"' for w in dict.fromkeys(words))
    sql = "select p.n,p.domain,p.dtype,p.obj,p.cons,p.vars,p.fam,p.sens,p.pb,p.problem, bm25(fts) from fts join p on p.n=fts.rowid where fts match ? "
    args = [qs]
    if family: sql += "and p.fam=? "; args.append(next((i for i, v in _dim("fam").items() if v == family), -1))
    sql += "order by bm25(fts) limit ?"; args.append(k)
    try: return [dict(_row(t[:10]), score=round(-t[10], 2)) for t in _db().execute(sql, args).fetchall()]
    except sqlite3.OperationalError: return []

def family_vote(text, k=25):
    """Share of the nearest catalogue problems that belong to each family. A retrieval signal that complements the trained classifier."""
    hits = search(text, k)
    if not hits: return {}
    tally = {}
    for h in hits: tally[h["formulation_family"]] = tally.get(h["formulation_family"], 0) + max(h["score"], .01)
    z = sum(tally.values()); return {f: round(v / z, 3) for f, v in sorted(tally.items(), key=lambda x: -x[1])}

if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "build": build(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    elif len(sys.argv) >= 2 and sys.argv[1] == "stats": print(stats())
    else: print(__doc__)
