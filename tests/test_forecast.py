"""Offline tests for the forecasting engine: no web framework needed."""
import numpy as np
from app import forecast as F

D = F.simulate(n=200, days=260, seed=3)

def test_features_use_only_the_past():
    t = 150; a = F.features(D["main"], D["lead"], t); m2 = D["main"].copy(); m2[:, t:] *= 50; l2 = D["lead"].copy(); l2[:, t:] *= 50
    assert np.allclose(a, F.features(m2, l2, t))

def test_label_is_five_times_baseline():
    main = np.ones((2, 100)); main[0, 60] = 5.0; main[1, 60] = 4.9; assert list(F.label(main, 50, 14)) == [1, 0] and list(F.label(main, 61, 14)) == [0, 0]

def test_simulator_has_breakouts_and_is_repeatable():
    assert sum(x["kind"] == "breakout" for x in D["meta"]) > 10 and np.array_equal(D["main"], F.simulate(n=200, days=260, seed=3)["main"])

def test_top10_shape_and_ranking():
    top = F.top10(d=D, H=14); assert len(top) == 10 and [t["breakout_probability"] for t in top] == sorted((t["breakout_probability"] for t in top), reverse=True)
    for t in top: assert 0 <= t["interval_10_90"][0] <= t["interval_10_90"][1] <= 1 and len(t["drivers"]) == 3

def test_top10_accepts_user_series():
    top = F.top10(series=D["main"][:60], names=[f"T{i}" for i in range(60)], H=7); assert len(top) == 10 and top[0]["name"].startswith("T")

def test_short_history_is_rejected():
    try: F.top10(series=D["main"][:20, :100]); assert False
    except ValueError: pass

def test_model_beats_every_baseline_in_backtest():
    r = F.backtest(d=F.simulate(n=300, days=260, seed=5), H=14, seed=5)
    assert r["model"]["precision_at_10"] > r["growth"]["precision_at_10"] > r["random"]["precision_at_10"] and r["model"]["precision_at_10"] > r["volume"]["precision_at_10"]
    assert r["brier"] < r["brier_base_rate"]

def test_candidates_match_optimizer_fields():
    c = F.to_candidates(F.top10(d=D, H=14)); assert len(c) == 10 and all({"name", "cat", "p", "E", "hl", "c"} <= set(x) for x in c)
