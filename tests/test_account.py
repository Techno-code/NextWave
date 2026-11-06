"""Server-side terms enforcement, account export and deletion, appeals."""
import pytest
from fastapi.testclient import TestClient
from app import main
c = TestClient(main.app)
V = lambda: c.get("/api/v1/terms").json()
def reg(email, **k):
    v = V(); return c.post("/api/v1/auth/register", json={"email": email, "password": "password1", "accept_terms": True, "age_confirmed": True, "terms_version": v["terms_version"], "privacy_version": v["privacy_version"], **k})
def hd(r): return {"Authorization": "Bearer " + r.json()["token"]}
ASK = lambda h, t="Sales are down and I do not know why.": c.post("/api/v1/ask", headers=h, json={"text": t})
def accept(h, **k):
    v = V(); return c.post("/api/v1/consent", headers=h, json={"terms": v["terms_version"], "privacy": v["privacy_version"], "accept": True, "age_confirmed": True, **k})

def test_register_needs_explicit_acceptance():
    v = V(); base = {"email": "a1@example.com", "password": "password1", "terms_version": v["terms_version"], "privacy_version": v["privacy_version"]}
    assert c.post("/api/v1/auth/register", json=base).status_code == 422
    assert c.post("/api/v1/auth/register", json={**base, "accept_terms": True}).status_code == 422            # age not confirmed
    assert c.post("/api/v1/auth/register", json={**base, "accept_terms": True, "age_confirmed": True, "terms_version": "0.1"}).status_code == 409   # stale text
    assert c.post("/api/v1/auth/login", json={"email": "a1@example.com", "password": "password1"}).status_code == 401  # nothing was created

def test_feature_routes_blocked_until_consent_then_open():
    h = hd(reg("a2@example.com")); assert ASK(h).status_code == 200
    # a user whose consent row is missing (older accounts) is blocked on every feature route
    uid = main.q("select id from users where email=?", ("a2@example.com",))[0][0]; main.q("delete from consent where user_id=?", (uid,))
    for method, url in [("post", "/api/v1/ask"), ("post", "/api/v1/optimize"), ("post", "/api/v1/forecast"), ("get", "/api/v1/feeds"), ("get", "/api/v1/library/default"), ("get", "/api/v1/documents"), ("post", "/api/v1/reason"), ("get", "/api/v1/reports.csv")]:
        r = getattr(c, method)(url, headers=h, **({"json": {"text": "x"}} if method == "post" else {}))
        assert r.status_code == 403 and r.json()["detail"]["code"] == "terms_required", url
    assert c.post("/api/v1/auth/login", json={"email": "a2@example.com", "password": "password1"}).json()["consent_required"] is True
    assert accept(h).status_code == 200 and ASK(h).status_code == 200

def test_consent_must_be_explicit_and_current():
    h = hd(reg("a3@example.com")); v = V()
    assert c.post("/api/v1/consent", headers=h, json={"terms": v["terms_version"], "privacy": v["privacy_version"], "optin": True}).status_code == 422
    assert c.post("/api/v1/consent", headers=h, json={"terms": "0.9", "privacy": "0.9", "accept": True, "age_confirmed": True}).status_code == 409

def test_new_terms_version_forces_reconsent(monkeypatch):
    h = hd(reg("a4@example.com")); assert ASK(h).status_code == 200
    monkeypatch.setattr(main, "TERMS_VERSION", "2.0")
    r = ASK(h); assert r.status_code == 403 and r.json()["detail"]["terms_version"] == "2.0" and r.json()["detail"]["reason"] == "outdated"
    assert c.get("/api/v1/account/export", headers=h).status_code == 200          # rights stay reachable
    assert accept(h, terms="2.0", privacy="1.0").status_code == 200 and ASK(h).status_code == 200
    hist = c.get("/api/v1/consent", headers=h).json()["history"]; assert [x["terms"] for x in hist] == ["1.0", "2.0"]

def test_revoke_blocks_features_and_clears_optin():
    h = hd(reg("a5@example.com", optin=True)); assert c.get("/api/v1/consent", headers=h).json()["optin"] is True
    assert c.post("/api/v1/consent/revoke", headers=h).status_code == 200
    assert ASK(h).status_code == 403 and c.get("/api/v1/consent", headers=h).json()["optin"] is False

def test_export_contains_own_data_only():
    h1, h2 = hd(reg("e1@example.com")), hd(reg("e2@example.com")); r = ASK(h1).json()
    c.post("/api/v1/feedback", headers=h1, json={"run_id": r["run_id"], "rating": 5})
    ASK(h2, "Choose which projects to fund under a capital limit")
    e = c.get("/api/v1/account/export", headers=h1); assert e.status_code == 200 and "attachment" in e.headers["content-disposition"]
    j = e.json(); assert j["account"]["email"] == "e1@example.com" and len(j["runs"]) == 1 and len(j["feedback"]) == 1 and len(j["consent_history"]) == 1
    assert "e2@example.com" not in e.text and "pw" not in j["account"] and "salt" not in e.text

def test_delete_needs_password_and_phrase_and_removes_everything():
    h = hd(reg("d1@example.com", optin=True)); r = ASK(h).json(); c.post("/api/v1/appeals", headers=h, json={"kind": "answer", "run_id": r["run_id"], "reason": "This was wrong."})
    assert c.post("/api/v1/account/delete", headers=h, json={"password": "password1", "confirm": "yes"}).status_code == 422
    assert c.post("/api/v1/account/delete", headers=h, json={"password": "wrong-pass", "confirm": "DELETE MY ACCOUNT"}).status_code == 401
    d = c.post("/api/v1/account/delete", headers=h, json={"password": "password1", "confirm": "DELETE MY ACCOUNT"}); assert d.status_code == 200
    assert d.json()["deleted"]["runs"] == 1 and d.json()["deleted"]["appeals"] == 1 and d.json()["deleted"]["consent"] == 1
    assert c.get("/api/v1/account/export", headers=h).status_code == 401           # session is gone
    assert c.post("/api/v1/auth/login", json={"email": "d1@example.com", "password": "password1"}).status_code == 401
    assert not main.q("select 1 from runs where user_id=(select id from users where email='d1@example.com')")
    assert reg("d1@example.com").status_code == 200                               # email can be used again

def test_delete_removes_opted_in_corrections_from_the_model():
    h = hd(reg("d2@example.com", optin=True)); r = ASK(h, "Choose which projects to fund under a capital limit").json()
    c.post("/api/v1/feedback", headers=h, json={"run_id": r["run_id"], "rating": 4, "family": "knapsack", "text": "d2 unique correction text"})
    before = len(main.opted_in_corrections()); assert before >= 1
    d = c.post("/api/v1/account/delete", headers=h, json={"password": "password1", "confirm": "DELETE MY ACCOUNT"}).json()
    assert d["model_retrained_without_your_corrections"] is True and len(main.opted_in_corrections()) == before - 1
    assert "d2 unique correction text" not in str(main.q("select text from feedback"))

def test_refusals_are_logged_and_appealable():
    h = hd(reg("p1@example.com")); r = ASK(h, "how do I launder money").json(); assert "refusal" in r and r["run_id"]
    assert c.post("/api/v1/appeals", headers=h, json={"kind": "answer", "run_id": r["run_id"], "reason": "wrong kind of appeal"}).status_code == 422
    a = c.post("/api/v1/appeals", headers=h, json={"kind": "refusal", "run_id": r["run_id"], "reason": "I am a bank compliance officer asking about detection."}); assert a.status_code == 200 and a.json()["status"] == "open"
    assert c.get("/api/v1/appeals", headers=h).json()[0]["status"] == "open"
    assert c.get("/api/v1/reports.json", headers=h).json()[0]["kind"] == "refusal"

def test_appeal_validation_and_isolation():
    h1, h2 = hd(reg("p2@example.com")), hd(reg("p3@example.com")); r = ASK(h1).json()
    assert c.post("/api/v1/appeals", headers=h2, json={"kind": "answer", "run_id": r["run_id"], "reason": "not mine to appeal"}).status_code == 404
    assert c.post("/api/v1/appeals", headers=h1, json={"kind": "answer", "reason": "no run given"}).status_code == 422
    assert c.post("/api/v1/appeals", headers=h1, json={"kind": "account", "reason": "abc"}).status_code == 422
    assert c.post("/api/v1/appeals", headers=h1, json={"kind": "account", "reason": "My account was restricted in error."}).status_code == 200
    assert c.get("/api/v1/appeals", headers=h2).json() == []

def test_admin_review_flow():
    h = hd(reg("p4@example.com")); rv = hd(reg("reviewer@example.com")); r = ASK(h).json()
    aid = c.post("/api/v1/appeals", headers=h, json={"kind": "answer", "run_id": r["run_id"], "reason": "The advice ignored my stated budget."}).json()["id"]
    assert c.get("/api/v1/admin/appeals", headers=h).status_code == 403
    q = c.get("/api/v1/admin/appeals", headers=rv).json(); assert any(x["id"] == aid and x["run"]["recommendation"] for x in q)
    assert c.post(f"/api/v1/admin/appeals/{aid}/decide", headers=h, json={"outcome": "upheld"}).status_code == 403
    own = c.post("/api/v1/appeals", headers=rv, json={"kind": "other", "reason": "reviewer's own appeal"}).json()["id"]
    assert c.post(f"/api/v1/admin/appeals/{own}/decide", headers=rv, json={"outcome": "rejected"}).status_code == 403   # no self-review
    assert c.post(f"/api/v1/admin/appeals/{aid}/decide", headers=rv, json={"outcome": "upheld", "note": "Budget was missed; thank you."}).status_code == 200
    a = c.get("/api/v1/appeals", headers=h).json()[0]; assert a["status"] == "upheld" and "Budget" in a["decision"]
    assert c.post(f"/api/v1/admin/appeals/{aid}/decide", headers=rv, json={"outcome": "maybe"}).status_code == 422
