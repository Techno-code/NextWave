"""Feed connectors: parsing, alignment, network safety, the Wikimedia connector (mocked), and forecasting from a feed through the API."""
import io, json, threading
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np, pytest
from fastapi.testclient import TestClient
from app import feeds, forecast, main

D0 = date(2025, 1, 1)
SIM = forecast.simulate(n=40, days=240, seed=2)
def wide_csv(n=40, days=240, names=None):
    nm = names or [f"trend {i}" for i in range(n)]
    rows = ["date," + ",".join(nm)] + [(D0 + timedelta(days=d)).isoformat() + "," + ",".join(f"{SIM['main'][i, d]:.2f}" for i in range(n)) for d in range(days)]
    return "\n".join(rows)
def long_csv(n=10, days=160):
    return "date,trend,value\n" + "\n".join(f"{(D0 + timedelta(days=d)).isoformat()},t{i},{SIM['main'][i, d]:.2f}" for i in range(n) for d in range(days))

def test_wide_and_long_csv_give_the_same_shape():
    a = feeds.align(feeds.parse_text(wide_csv()), "x"); b = feeds.align(feeds.parse_text(long_csv()), "x")
    assert a["matrix"].shape == (40, 240) and b["matrix"].shape == (10, 160) and a["dates"][0] == "2025-01-01" and a["dates"][-1] == (D0 + timedelta(days=239)).isoformat()
    assert np.allclose(a["matrix"][a["names"].index("trend 3")], np.maximum(np.round(SIM["main"][3, :240], 2), 1.0))

def test_json_shapes():
    by = {f"t{i}": {(D0 + timedelta(days=d)).isoformat(): float(SIM["main"][i, d]) for d in range(160)} for i in range(9)}
    assert feeds.align(feeds.parse_text(json.dumps(by)), "j")["matrix"].shape == (9, 160)
    lst = [{"date": k, "trend": t, "value": v} for t, dd in by.items() for k, v in dd.items()]
    assert feeds.align(feeds.parse_text(json.dumps(lst)), "j")["matrix"].shape == (9, 160)

@pytest.mark.parametrize("text,msg", [("", "empty"), ("a,b\n1,2", "date column"), ("date,x\n2025-13-45,3\n2025-01-01,4", "Could not read the date"), (wide_csv(n=5), "at least 8"), (wide_csv(days=100), "at least 150")])
def test_bad_feeds_are_rejected_with_a_reason(text, msg):
    with pytest.raises(feeds.FeedError, match=msg): feeds.align(feeds.parse_text(text), "x")

def test_small_gaps_filled_large_gaps_rejected():
    lines = wide_csv(n=10, days=200).split("\n"); del lines[50:52]; ok = feeds.align(feeds.parse_text("\n".join(lines)), "x")
    assert ok["matrix"].shape[1] == 200 and any("carried forward" in n for n in ok["notes"] if n)
    lines = wide_csv(n=10, days=200).split("\n"); del lines[50:58]
    with pytest.raises(feeds.FeedError, match="gap longer"): feeds.align(feeds.parse_text("\n".join(lines)), "x")

def test_zero_values_are_floored_so_growth_ratios_stay_defined():
    m = feeds.align(feeds.parse_text(wide_csv(n=10, days=160).replace(",", ",0,", 0)), "x")["matrix"]; assert (m >= 1).all()

@pytest.mark.parametrize("url", ["http://example.com/a.csv", "ftp://example.com/a.csv", "https:///nohost", "https://localhost/a.csv", "https://127.0.0.1/a.csv", "https://10.0.0.5/a.csv", "https://169.254.169.254/latest/meta-data", "https://[::1]/a.csv"])
def test_url_feeds_refuse_insecure_and_private_addresses(url):
    with pytest.raises(feeds.FeedError): feeds.check_url(url, allow_http=False)

def test_wikipedia_connector_builds_requests_and_fills_zero_days():
    end = date(2025, 9, 1); calls = []
    def get(url):
        calls.append(url); i = len(calls) - 1
        if "Missing_Page" in url: return {"items": []}
        start = end - timedelta(days=199)
        return {"items": [{"article": "x", "timestamp": (start + timedelta(days=d)).strftime("%Y%m%d") + "00", "views": int(SIM["main"][i % 40, d])} for d in range(200) if d % 17]}   # every 17th day absent
    arts = [f"Article {i}" for i in range(9)] + ["Missing Page"]
    s = feeds.from_wikipedia(arts, days=200, end=end, get=get)
    assert s["matrix"].shape == (9, 200) and s["dates"][-1] == "2025-09-01" and any("Missing_Page" in n for n in s["notes"] if n)
    assert "/per-article/en.wikipedia/all-access/user/Article_0/daily/20250214/20250901" in calls[0]
    assert (s["matrix"] >= 1).all()

def test_wikipedia_connector_validates_input():
    with pytest.raises(feeds.FeedError): feeds.from_wikipedia(["only one"])
    with pytest.raises(feeds.FeedError): feeds.from_wikipedia([f"a{i}" for i in range(9)], days=10)

# ---- API, with a real local web server standing in for a customer's feed address ----
class _H(BaseHTTPRequestHandler):
    body = b""
    def do_GET(self):
        if self.path == "/redirect": self.send_response(302); self.send_header("Location", "/feed.csv"); self.end_headers(); return
        if self.path == "/missing": self.send_response(404); self.end_headers(); return
        self.send_response(200); self.send_header("Content-Type", "text/csv"); self.end_headers(); self.wfile.write(_H.body)
    def log_message(self, *a): pass
@pytest.fixture(scope="module")
def server():
    _H.body = wide_csv().encode(); s = HTTPServer(("127.0.0.1", 0), _H); threading.Thread(target=s.serve_forever, daemon=True).start(); yield f"http://127.0.0.1:{s.server_port}"; s.shutdown()
c = TestClient(main.app)
def hd(email):
    v = c.get("/api/v1/terms").json()
    return {"Authorization": "Bearer " + c.post("/api/v1/auth/register", json={"email": email, "password": "password1", "accept_terms": True, "age_confirmed": True, "terms_version": v["terms_version"], "privacy_version": v["privacy_version"]}).json()["token"]}

def test_url_feed_end_to_end(server):
    h = hd("f1@example.com")
    f = c.post("/api/v1/feeds/url", headers=h, json={"name": "mine", "url": server + "/feed.csv"}); assert f.status_code == 200, f.text
    j = f.json(); assert j["trends"] == 40 and j["days"] == 240 and j["as_of"] == (D0 + timedelta(days=239)).isoformat() and j["stale_days"] > 7
    r = c.post("/api/v1/forecast", headers=h, json={"horizon": 14, "feed_id": j["id"], "threshold": 2.0}); assert r.status_code == 200, r.text
    t = r.json(); assert t["source"] == "feed" and t["feed"]["name"] == "mine" and len(t["top10"]) == 10 and "days old" in t["note"] and all(x["name"].startswith("trend") for x in t["top10"])
    o = c.post("/api/v1/optimize", headers=h, json={"trends": t["candidates"]}); assert o.status_code == 200 and all(p["verified"] for p in o.json()["portfolios"])
    assert c.post(f"/api/v1/feeds/{j['id']}/refresh", headers=h).status_code == 200
    assert c.delete(f"/api/v1/feeds/{j['id']}", headers=h).status_code == 200 and c.get("/api/v1/feeds", headers=h).json() == []

def test_feed_errors_are_clear_and_feeds_are_private(server):
    h, h2 = hd("f2@example.com"), hd("f3@example.com")
    assert c.post("/api/v1/feeds/url", headers=h, json={"name": "r", "url": server + "/redirect"}).status_code == 422
    assert "404" in c.post("/api/v1/feeds/url", headers=h, json={"name": "m", "url": server + "/missing"}).text
    csvf = c.post("/api/v1/feeds/csv", headers=h, params={"name": "up"}, files={"file": ("t.csv", wide_csv().encode(), "text/csv")}); assert csvf.status_code == 200
    fid = csvf.json()["id"]
    assert c.post(f"/api/v1/feeds/{fid}/refresh", headers=h).status_code == 422          # uploads cannot refresh
    assert c.post("/api/v1/forecast", headers=h2, json={"feed_id": fid}).status_code == 404 and c.delete(f"/api/v1/feeds/{fid}", headers=h2).status_code == 200 and c.get("/api/v1/feeds", headers=h).json()[0]["id"] == fid
    assert c.post("/api/v1/feeds/csv", headers=h, params={"name": "bad"}, files={"file": ("t.csv", b"date,a\n2025-01-01,1", "text/csv")}).status_code == 422

def test_flat_feed_explains_why_it_cannot_forecast(server):
    h = hd("f4@example.com"); flat = np.exp(np.random.default_rng(1).normal(5, .05, (12, 200))); S = {"main": flat}
    txt = "date," + ",".join(f"f{i}" for i in range(12)) + "\n" + "\n".join((D0 + timedelta(days=d)).isoformat() + "," + ",".join(f"{flat[i, d]:.2f}" for i in range(12)) for d in range(200))
    fid = c.post("/api/v1/feeds/csv", headers=h, params={"name": "flat"}, files={"file": ("t.csv", txt.encode(), "text/csv")}).json()["id"]
    r = c.post("/api/v1/forecast", headers=h, json={"feed_id": fid}); assert r.status_code == 422 and "too few to learn from" in r.json()["detail"] and "Lower the break-out threshold" in r.json()["detail"]

def test_threshold_is_configurable_and_validated():
    h = hd("f5@example.com"); assert c.post("/api/v1/forecast", headers=h, json={"threshold": 1.0}).status_code == 422
    r = c.post("/api/v1/forecast", headers=h, json={"threshold": 2.0}); assert r.status_code == 200 and r.json()["top10"][0]["threshold"] == "2x the 30-day baseline"

def test_feed_limit_per_account(server):
    h = hd("f6@example.com")
    for i in range(10): assert c.post("/api/v1/feeds/csv", headers=h, params={"name": f"f{i}"}, files={"file": ("t.csv", wide_csv(n=9, days=160).encode(), "text/csv")}).status_code == 200
    assert c.post("/api/v1/feeds/csv", headers=h, params={"name": "x"}, files={"file": ("t.csv", wide_csv(n=9, days=160).encode(), "text/csv")}).status_code == 409

def test_due_feeds_refresh_in_the_background(server):
    h = hd("f7@example.com"); j = c.post("/api/v1/feeds/url", headers=h, json={"name": "auto", "url": server + "/feed.csv", "refresh_minutes": 60}).json()
    main.q("update feeds set next_refresh=0, updated='old' where id=?", (j["id"],)); main.refresh_due_feeds()
    assert main.q("select updated from feeds where id=?", (j["id"],))[0][0] != "old" and main.q("select next_refresh from feeds where id=?", (j["id"],))[0][0] > 0


def test_user_agent_carries_the_operators_contact(monkeypatch):
    monkeypatch.delenv("DE_CONTACT", raising=False); assert "contact" not in feeds.ua()
    monkeypatch.setenv("DE_CONTACT", "ops@example.com"); assert "contact: ops@example.com" in feeds.ua()
