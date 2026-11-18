"""Checks that a forecast for week t cannot see anything released after week t's forecast time."""
import numpy as np
import pandas as pd

from src.model1 import FEATURES, build_features, load_raw, ridge, walk_forward

CUTOFF = pd.Timestamp("2021-06-04")


def test_features_ignore_future_storage():
    s, w = load_raw()
    base = build_features(s, w)
    s2 = s.astype(float).copy()
    s2.loc[s2.index > CUTOFF, "storage_bcf"] += np.random.default_rng(0).normal(0, 500, (s2.index > CUTOFF).sum())
    pert = build_features(s2, w)
    # target 'chg' at CUTOFF+1 legitimately changes; features up to and including the
    # first week after the cutoff must not (they use storage only up to the prior week)
    first_after = base.index[base.index > CUTOFF][0]
    cols = [c for c in FEATURES]
    pd.testing.assert_frame_equal(base.loc[:first_after, cols], pert.loc[:first_after, cols])


def test_features_ignore_future_weather():
    s, w = load_raw()
    base = build_features(s, w)
    w2 = w.copy()
    w2.loc[w2.index > CUTOFF, ["hdd", "cdd"]] *= 3
    pert = build_features(s, w2)
    pd.testing.assert_frame_equal(base.loc[:CUTOFF, FEATURES], pert.loc[:CUTOFF, FEATURES])


def test_walk_forward_prediction_unchanged_by_future_rows():
    df = build_features()
    t = df.index[df.index >= CUTOFF][0]
    p_full = walk_forward(df, ridge(1.0), t).loc[t]
    p_trunc = walk_forward(df[df.index <= t], ridge(1.0), t).loc[t]
    assert abs(p_full - p_trunc) < 1e-9
