"""Timing checks for Model 2 (future data cannot change a forecast) and Model 3's dataset."""
import numpy as np
import pandas as pd
import pytest

import src.model1 as M1
import src.model2 as M2

T = pd.Timestamp("2023-12-01")


def test_model2_ignores_future_storage_and_weather(monkeypatch):
    s, w = M1.load_raw()
    base = M2.Simulator(n_resid_paths=20, seed=3).simulate(T)

    s2, w2 = s.astype(float).copy(), w.copy()
    s2.loc[s2.index > T, "storage_bcf"] += 777.0
    w2.loc[w2.index > T, ["hdd", "cdd"]] *= 2.5
    monkeypatch.setattr(M2, "load_raw", lambda: (s2, w2))
    pert = M2.Simulator(n_resid_paths=20, seed=3).simulate(T)
    # the realised end-of-season "actual" legitimately changes; the forecast must not
    for k in ("p10", "median", "p90", "benchmark"):
        assert base[k] == pytest.approx(pert[k]), k


def test_model3_rows_are_timed_correctly():
    from src.model3_data import PROC
    p = PROC / "model3_weekly.csv"
    if not p.exists():
        pytest.skip("Model 3 dataset not built here (licensed price data is not in the repo)")
    d = pd.read_csv(p, parse_dates=["obs_date", "origin_week", "date_1w", "date_4w"])
    assert (d.obs_date.dt.dayofweek < 5).all()
    assert (d.obs_date >= d.origin_week + pd.Timedelta(days=7)).all()  # report out before settlement
    for h in (1, 4):
        ok = d[f"date_{h}w"].notna()
        assert (d.loc[ok, f"date_{h}w"] >= d.loc[ok, "obs_date"] + pd.Timedelta(days=7 * h)).all()
    assert (d.weeks_to_expiry * 7 >= 25 - 1e-9).all()
    assert np.allclose(d.spread, d.march - d.april)
