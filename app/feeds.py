"""Trend data feeds for the forecaster (Trend Portfolio Optimizer PRD, section 6A: connectors).

Three connectors, all producing the same thing: daily attention for several trends, aligned on one date axis.
  csv        an uploaded file, wide (date + one column per trend) or long (date, trend, value)
  url        a CSV or JSON file on an HTTPS address you control, re-fetched on a schedule
  wikipedia  daily page views of named articles from the public Wikimedia REST API (no key needed)
Nothing here talks to a database. app/main.py stores the result and serves it to the forecaster.
"""
import csv, io, ipaddress, json, os, socket, urllib.error, urllib.parse, urllib.request
from datetime import date, datetime, timedelta, timezone
import numpy as np

MAX_BYTES = 20 * 1024 * 1024
MAX_TRENDS = 500
MAX_DAYS = 730
MIN_DAYS = 150          # the forecaster needs MIN_ORIGIN (90) + 60 days
MIN_TRENDS = 8
GAP_FILL_DAYS = 3
MAX_MISSING_SHARE = .10
def ua():
    c = os.environ.get("DE_CONTACT")
    return "QuantInsight/1.0 (trend forecaster" + (f"; contact: {c}" if c else "") + ")"

class FeedError(ValueError): pass

def _day(s):
    s = str(s).strip()
    for f in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%Y%m%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try: return datetime.strptime(s, f).date()
        except ValueError: pass
    raise FeedError(f"Could not read the date '{s[:30]}'. Use YYYY-MM-DD.")

def _num(x):
    try: v = float(str(x).replace(",", "").strip())
    except ValueError: return None
    return v if np.isfinite(v) and v >= 0 else None

def align(points, source):
    """points: {trend: {date: value}} -> series dict. Fills gaps of up to 3 days, rejects feeds with more than 10 percent missing days, floors values at 1 so that growth ratios stay defined."""
    points = {k: v for k, v in points.items() if v}
    if len(points) < MIN_TRENDS: raise FeedError(f"The feed has {len(points)} usable trends; at least {MIN_TRENDS} are needed to learn what a break-out looks like.")
    if len(points) > MAX_TRENDS: raise FeedError(f"At most {MAX_TRENDS} trends are supported.")
    last = min(max(v) for v in points.values()); first = max(min(v) for v in points.values())
    days = (last - first).days + 1
    if days < MIN_DAYS: raise FeedError(f"The trends share only {max(days, 0)} days of history; at least {MIN_DAYS} are needed.")
    first = max(first, last - timedelta(days=MAX_DAYS - 1)); days = (last - first).days + 1
    names = sorted(points); M = np.full((len(names), days), np.nan)
    for r, k in enumerate(names):
        for d, v in points[k].items():
            if first <= d <= last: M[r, (d - first).days] = v
    missing = float(np.isnan(M).mean())
    if missing > MAX_MISSING_SHARE: raise FeedError(f"{missing:.0%} of the daily values are missing; at most {MAX_MISSING_SHARE:.0%} is allowed.")
    filled = 0
    for r in range(len(names)):
        run = 0; last_v = np.nan
        for c in range(days):
            if np.isnan(M[r, c]):
                run += 1
                if np.isnan(last_v) or run > GAP_FILL_DAYS: raise FeedError(f"'{names[r]}' has a gap longer than {GAP_FILL_DAYS} days near {first + timedelta(days=c)}.")
                M[r, c] = last_v; filled += 1
            else: run = 0; last_v = M[r, c]
    floored = int((M < 1).sum()); M = np.maximum(M, 1.0)
    return {"dates": [(first + timedelta(days=i)).isoformat() for i in range(days)], "names": names, "matrix": M, "source": source,
            "notes": [f"{filled} missing values were carried forward from the previous day." if filled else None, f"{floored} values below 1 were raised to 1." if floored else None]}

def parse_text(text, content_type=""):
    """CSV (wide or long) or JSON ({"trend": {"YYYY-MM-DD": value}} or a list of {date, trend, value})."""
    t = text.lstrip("\ufeff").strip()
    if not t: raise FeedError("The feed is empty.")
    pts = {}
    if t[0] in "{[" or "json" in content_type:
        try: j = json.loads(t)
        except ValueError: raise FeedError("The feed is not valid JSON.")
        if isinstance(j, dict) and "series" in j and isinstance(j["series"], dict): j = j["series"]
        if isinstance(j, dict):
            for k, v in j.items():
                if not isinstance(v, dict): raise FeedError("JSON objects must map each trend to {date: value}.")
                for d, x in v.items():
                    n = _num(x)
                    if n is not None: pts.setdefault(str(k), {})[_day(d)] = n
        elif isinstance(j, list):
            for r in j:
                try: d, k, x = r.get("date") or r.get("day"), r.get("trend") or r.get("name") or r.get("term"), r.get("value") or r.get("views") or r.get("count")
                except AttributeError: raise FeedError("JSON lists must hold objects with date, trend and value.")
                n = _num(x)
                if d is not None and k is not None and n is not None: pts.setdefault(str(k), {})[_day(d)] = n
        else: raise FeedError("The JSON feed has an unexpected shape.")
        return pts
    rows = list(csv.reader(io.StringIO(t)))
    if len(rows) < 2: raise FeedError("The CSV needs a header and at least one row.")
    head = [h.strip().lower() for h in rows[0]]
    dcol = next((i for i, h in enumerate(head) if h in ("date", "day", "timestamp")), None)
    if dcol is None: raise FeedError("The CSV needs a date column named date or day.")
    tcol = next((i for i, h in enumerate(head) if h in ("trend", "name", "term", "topic", "article")), None)
    vcol = next((i for i, h in enumerate(head) if h in ("value", "views", "count", "attention", "volume")), None)
    if tcol is not None and vcol is not None:
        for r in rows[1:]:
            if len(r) <= max(dcol, tcol, vcol) or not r[dcol].strip(): continue
            n = _num(r[vcol])
            if n is not None: pts.setdefault(r[tcol].strip(), {})[_day(r[dcol])] = n
    else:
        for r in rows[1:]:
            if len(r) <= dcol or not r[dcol].strip(): continue
            d = _day(r[dcol])
            for i, h in enumerate(rows[0]):
                if i != dcol and i < len(r):
                    n = _num(r[i])
                    if n is not None: pts.setdefault(h.strip(), {})[d] = n
    return pts

def check_url(url, allow_http=False):
    p = urllib.parse.urlparse(url)
    if p.scheme != "https" and not (allow_http and p.scheme == "http"): raise FeedError("Feed addresses must start with https://.")
    if not p.hostname: raise FeedError("The feed address has no host name.")
    try: infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except OSError: raise FeedError("The feed host name could not be resolved.")
    if not allow_http:
        for i in infos:
            ip = ipaddress.ip_address(i[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
                raise FeedError("Feed addresses may not point at private or local networks.")

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k): raise FeedError("The feed address redirects; give the final address.")

def fetch_text(url, allow_http=False, timeout=20):
    check_url(url, allow_http)
    try:
        r = urllib.request.build_opener(_NoRedirect).open(urllib.request.Request(url, headers={"User-Agent": ua(), "Accept": "text/csv, application/json"}), timeout=timeout)
        data = r.read(MAX_BYTES + 1); ctype = r.headers.get("content-type", "")
    except FeedError: raise
    except urllib.error.HTTPError as e: raise FeedError(f"The feed address answered with HTTP {e.code}.")
    except (urllib.error.URLError, TimeoutError, OSError) as e: raise FeedError(f"The feed could not be fetched: {getattr(e, 'reason', e)}")
    if len(data) > MAX_BYTES: raise FeedError("The feed is larger than 20 MB.")
    return data.decode("utf-8", "replace"), ctype

def from_url(url, allow_http=False, fetch=fetch_text):
    text, ctype = fetch(url, allow_http)
    return align(parse_text(text, ctype), f"url:{urllib.parse.urlparse(url).netloc}")

WIKI = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/{access}/{agent}/{article}/daily/{start}/{end}"
def _wiki_get(url, timeout=20):
    try: return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": ua(), "Accept": "application/json"}), timeout=timeout))
    except urllib.error.HTTPError as e:
        if e.code == 404: return {"items": []}
        raise FeedError(f"Wikimedia answered {e.code}.")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e: raise FeedError(f"Wikimedia could not be reached: {getattr(e, 'reason', e)}")

def from_wikipedia(articles, days=365, project="en.wikipedia", end=None, get=_wiki_get):
    """Daily page views for each article (use the title as it appears in the address, for example Pickleball). Days with no views are returned by the API as absent and are filled with zero."""
    arts = [a.strip().replace(" ", "_") for a in articles if a and a.strip()]
    if len(arts) < MIN_TRENDS: raise FeedError(f"Give at least {MIN_TRENDS} article titles.")
    if len(arts) > 100: raise FeedError("At most 100 articles per feed.")
    if not 150 <= days <= MAX_DAYS: raise FeedError(f"days must be between 150 and {MAX_DAYS}.")
    end = end or (datetime.now(timezone.utc).date() - timedelta(days=2)); start = end - timedelta(days=days - 1); pts = {}; skipped = []
    for a in arts:
        j = get(WIKI.format(project=urllib.parse.quote(project, safe="."), access="all-access", agent="user", article=urllib.parse.quote(a, safe=""), start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d")))
        items = j.get("items") or []
        if not items: skipped.append(a); continue
        d = {}
        for it in items:
            try: d[datetime.strptime(str(it["timestamp"])[:8], "%Y%m%d").date()] = float(it["views"])
            except (KeyError, ValueError): continue
        for i in range(days):
            d.setdefault(start + timedelta(days=i), 0.0)
        pts[a.replace("_", " ")] = d
    out = align(pts, f"wikipedia:{project}")
    if skipped: out["notes"].append("No data for: " + ", ".join(skipped) + ".")
    return out

def meta(s):
    return {"trends": len(s["names"]), "days": len(s["dates"]), "first_date": s["dates"][0], "as_of": s["dates"][-1], "source": s["source"], "notes": [n for n in s["notes"] if n]}

def stale_days(as_of, today=None):
    return ((today or datetime.now(timezone.utc).date()) - date.fromisoformat(as_of)).days
