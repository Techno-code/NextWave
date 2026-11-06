"""Trend forecasting engine (Trend Portfolio Optimizer PRD, section 6A).
Predicts the probability that a trend's attention rises to at least 5x its own 30-day baseline within the next H days (H = 7, 14 or 30).
Data: user-supplied daily series, or the built-in simulator (demo and test mode). Results on simulated data show that the pipeline works; they say nothing about real-world accuracy.
Python:  from app import forecast;  forecast.top10(H=14)        CLI:  python -m app.forecast backtest"""
import json, math, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

THRESH = 5.0; MIN_ORIGIN = 90
NAMES = ["Quiet luxury packaging", "Micro-session fitness", "Retro handheld gaming", "Protein snacks", "Local-first apps", "Thrift-haul videos", "Desk plant setups", "AI study planners", "Slow-travel itineraries", "Cold-plunge routines",
         "Mushroom coffee", "Analog photography", "Pickleball gear", "Silent book clubs", "Sleep tracking rings", "Reusable tech cases", "Home espresso", "Gravel biking", "Minimalist wallets", "Plant-based jerky"]

def simulate(n=400, days=260, seed=0, breakout_rate=.18):
    """Daily attention for a main platform and an earlier-moving leader platform. Break-outs rise over a few days and then decay; some trends have false starts."""
    R = np.random.default_rng(seed); t = np.arange(days); main = np.zeros((n, days)); lead = np.zeros((n, days)); truth = []
    for i in range(n):
        base = math.exp(R.normal(3, 1)); amp = R.uniform(0, .25); ph = R.uniform(0, 6.28); m = np.ones(days); kind = "flat"
        u = R.random()
        if u < breakout_rate: kind, M, rise, hl, t0 = "breakout", R.uniform(6, 25), R.uniform(3, 12), R.uniform(5, 20), int(R.integers(50, days - 10))
        elif u < breakout_rate + .2: kind, M, rise, hl, t0 = "false_start", R.uniform(1.5, 3.5), R.uniform(3, 12), R.uniform(3, 10), int(R.integers(40, days - 10))
        if kind != "flat":
            def curve(shift):
                x = t - (t0 - shift); c = np.ones(days); up = (x >= 0) & (x < rise); c[up] = 1 + (M - 1) * (1 - np.exp(-3 * x[up] / rise))
                top = 1 + (M - 1) * (1 - math.exp(-3)); dn = x >= rise; c[dn] = 1 + (top - 1) * .5 ** ((x[dn] - rise) / hl); return c
            m = curve(0); ml = curve(int(R.integers(2, 7)))
        else: ml = np.ones(days)
        season = 1 + amp * np.sin(t / 7 * 2 * math.pi + ph)
        main[i] = base * season * m * np.exp(R.normal(0, .15, days)); lead[i] = base * .6 * season * ml * np.exp(R.normal(0, .2, days)); truth.append({"kind": kind, "cat": int(R.integers(0, 6))})
    return {"main": main, "lead": lead, "meta": truth, "names": [NAMES[i % len(NAMES)] + ("" if i < len(NAMES) else f" {i // len(NAMES) + 1}") for i in range(n)]}

def features(main, lead, t):
    """One row per trend at forecast date t using only data before t."""
    m, l = main[:, :t], lead[:, :t]; b30 = m[:, -30:].mean(1); lb30 = l[:, -30:].mean(1); c3 = m[:, -3:].mean(1); c7 = m[:, -7:].mean(1); p7 = m[:, -14:-7].mean(1); pp7 = m[:, -21:-14].mean(1)
    l3 = l[:, -3:].mean(1); l7 = l[:, -7:].mean(1); lp7 = l[:, -14:-7].mean(1); vol = np.log(m[:, -14:]).std(1); old = m[:, -90:-30].max(1)
    return np.column_stack([c3 / b30, c7 / b30, c7 / p7, p7 / pp7, (c7 / p7) - (p7 / pp7), l3 / lb30, l7 / lp7, l7 / lb30, vol, np.log(b30), c7 / old, m[:, -1] / b30])
FEATURE_NAMES = ["attention vs 30-day baseline (3 days)", "attention vs 30-day baseline (7 days)", "weekly growth", "previous weekly growth", "acceleration", "leader platform vs baseline (3 days)", "leader weekly growth", "leader vs baseline (7 days)", "volatility", "volume (log)", "novelty vs prior peak", "latest day vs baseline"]

def label(main, t, H, thresh=THRESH): return (main[:, t:t + H].max(1) >= thresh * main[:, t - 30:t].mean(1)).astype(int)

def dataset(d, origins, H, thresh=THRESH):
    X, y, o = [], [], []
    for t in origins: X.append(features(d["main"], d["lead"], t)); y.append(label(d["main"], t, H, thresh)); o.append(np.full(len(y[-1]), t))
    return np.vstack(X), np.concatenate(y), np.concatenate(o)

class Forecaster:
    """Bootstrap ensemble of gradient-boosted trees, with isotonic calibration on a time-ordered validation block. Intervals are ensemble 10th-90th percentiles (not conformal)."""
    def __init__(s, members=5, seed=0): s.members, s.seed, s.models, s.iso = members, seed, [], None
    def fit(s, X, y, o):
        cut = np.quantile(o, .8); tr, va = o <= cut, o > cut; R = np.random.default_rng(s.seed); s.models = []
        for k in range(s.members):
            ix = R.choice(np.where(tr)[0], size=int(tr.sum()), replace=True); s.models.append(HistGradientBoostingClassifier(max_iter=120, learning_rate=.08, max_depth=4, random_state=k).fit(X[ix], y[ix]))
        raw = np.mean([m.predict_proba(X[va])[:, 1] for m in s.models], 0); s.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw, y[va]); return s
    def predict(s, X):
        P = np.array([m.predict_proba(X)[:, 1] for m in s.models]); p = s.iso.predict(P.mean(0)); lo, hi = (s.iso.predict(np.quantile(P, q, axis=0)) for q in (.1, .9)); return p, lo, hi

def ndcg(rel_ranked, k=10, total=None):
    g = lambda r: sum(x / math.log2(i + 2) for i, x in enumerate(r[:k])); ideal = g(sorted(rel_ranked, reverse=True)) if total is None else g([1] * min(k, total)); return g(rel_ranked) / ideal if ideal else None

def brier_ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y, float); e = 0.0
    for b in range(bins):
        m = (p > b / bins) & (p <= (b + 1) / bins)
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(((p - y) ** 2).mean()), float(e)

def backtest(d=None, H=14, cutoffs=(150, 180, 210), step=7, seed=0):
    """Rolling-origin test. The model sees only origins whose labels are fully known by the cutoff; test origins start after the cutoff."""
    d = d or simulate(seed=seed); days = d["main"].shape[1]; rnd = np.random.default_rng(seed + 1)
    res = {k: {"p10": [], "ndcg": [], "lead": []} for k in ("model", "growth", "volume", "random")}; P, Y = [], []
    for c in cutoffs:
        X, y, o = dataset(d, range(MIN_ORIGIN, c - H + 1, step), H); f = Forecaster(seed=seed).fit(X, y, o)
        for t in range(c, min(c + 30, days - H) + 1, step):
            Xt = features(d["main"], d["lead"], t); yt = label(d["main"], t, H); p, lo, hi = f.predict(Xt); P += list(p); Y += list(yt)
            if yt.sum() == 0: continue
            scores = {"model": p, "growth": Xt[:, 2], "volume": Xt[:, 9], "random": rnd.random(len(yt))}
            for k, s in scores.items():
                top = np.argsort(-s)[:10]; res[k]["p10"].append(yt[top].mean()); res[k]["ndcg"].append(ndcg(list(yt[top]), 10, int(yt.sum())))
                for i in top:
                    if yt[i]: res[k]["lead"].append(int(np.argmax(d["main"][i, t:t + H] >= THRESH * d["main"][i, t - 30:t].mean())))
    br, ec = brier_ece(P, Y); out = {"horizon_days": H, "threshold": f"{THRESH:g}x the 30-day baseline", "forecast_dates": len(res["model"]["p10"]), "positive_rate": round(float(np.mean(Y)), 4), "brier": round(br, 4), "brier_base_rate": round(float(np.mean(Y) * (1 - np.mean(Y))), 4), "calibration_error": round(ec, 4),
                                    "data": "simulated; shows the pipeline works, not real-world accuracy"}
    for k, v in res.items(): out[k] = {"precision_at_10": round(float(np.mean(v["p10"])), 3), "ndcg_at_10": round(float(np.mean([x for x in v["ndcg"] if x is not None])), 3), "mean_days_before_breakout": round(float(np.mean(v["lead"])), 1) if v["lead"] else None}
    return out

def top10(series=None, names=None, H=14, seed=0, d=None, thresh=THRESH):
    """Ranked top 10 for the latest date. series: optional array [trends, days] of daily attention (at least 120 days); a leader series is not required."""
    if series is not None:
        main = np.asarray(series, float); d = {"main": main, "lead": main.copy(), "names": names or [f"Trend {i + 1}" for i in range(len(main))]}
    d = d or simulate(seed=seed); days = d["main"].shape[1]
    if days < MIN_ORIGIN + 60: raise ValueError("Need at least 150 days of history.")
    X, y, o = dataset(d, range(MIN_ORIGIN, days - H + 1, 5), H, thresh); pos = int(y.sum())
    if pos < 20 or len(y) - pos < 20: raise ValueError(f"Only {pos} of {len(y)} past examples reached {thresh:g}x their 30-day baseline within {H} days, too few to learn from. Lower the break-out threshold (for example 1.5 or 2) or add more trends and history.")
    f = Forecaster(seed=seed).fit(X, y, o); t = days; Xt = features(d["main"], d["lead"], t); p, lo, hi = f.predict(Xt)
    imp = np.abs(Xt - np.median(Xt, 0)) / (Xt.std(0) + 1e-9); out = []
    for i in np.argsort(-p)[:10]:
        drv = [FEATURE_NAMES[j] for j in np.argsort(-imp[i])[:3]]
        out.append({"name": d["names"][i], "breakout_probability": round(float(p[i]), 3), "interval_10_90": [round(float(lo[i]), 3), round(float(hi[i]), 3)], "attention_vs_baseline": round(float(Xt[i, 1]), 2), "weekly_growth": round(float(Xt[i, 2]), 2), "drivers": drv, "horizon_days": H, "threshold": f"{thresh:g}x the 30-day baseline"})
    return out

def to_candidates(top, cost=3.0):
    """Hand-off to the optimizer (/api/v1/optimize 'trends'): probability p, expected value E, half-life hl, cost c, category cat."""
    return [{"name": t["name"], "cat": i % 6, "p": t["breakout_probability"], "E": round(20 + 40 * min(1, t["attention_vs_baseline"] / 5), 1), "hl": 10, "c": cost} for i, t in enumerate(top)]

if __name__ == "__main__":
    if sys.argv[1:] == ["backtest"]: print(json.dumps({H: backtest(H=H) for H in (7, 14, 30)}, indent=1))
    else: print(json.dumps(top10(), indent=1))
