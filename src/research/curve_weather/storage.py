"""Part 4: a storage nowcast from the curve's implied weather, and the curve as a Model 1 feature.

4a. At Friday F (settle), forecast the storage change for the week ending T = F + 7 (EIA publishes
    it on T + 6, i.e. 13 days later). Model 1's ridge (alpha = 0.1, Model 1's own validated choice)
    is refitted walk-forward on weeks ending <= F - 7 (published by F) with realised weather and
    with the storage-lag features moved back one more week (they are not yet public at F).
    It is then fed different HDD inputs for week T:
      clim:   10-year same-week normal HDD (zero anomaly)
      pers:   normal + calibrated persistence forecast of a1 (from a0)
      market: normal + curve-implied a1 (walk-forward calibrated, front/second/Mar-Apr)
      market+pers: normal + implied a1 from curve and a0
      oracle: realised HDD of week T (upper bound: perfect forecast)
    and compared with the 5-year average change (Model 1's benchmark).
4b. Model 1 exactly (FEATURES_V2, ridge 0.1, walk-forward, refit weekly) with and without curve
    features for the same week (known at the Thursday report): the walk-forward implied wx2 and
    the raw front-month move, set to 0 outside the Nov-Feb panel weeks.
All storage and weather inputs are cut at 2021-12-31 on load.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import ROOT, RAW, CUTOFF

sys.path.insert(0, str(ROOT))
from src import model1  # noqa: E402

ALPHA = 0.1


def model1_frame():
    s = pd.read_csv(RAW / "storage.csv", parse_dates=["week_ending"])
    s = s[s.week_ending <= CUTOFF].set_index("week_ending")
    w = pd.read_csv(RAW / "weekly_degree_days_own.csv", parse_dates=["week_ending"])
    w = w[w.week_ending <= CUTOFF].set_index("week_ending")
    w["forecast_days"] = 0
    df = model1.build_features(s, w)
    assert df.index.max() <= CUTOFF
    return df


def _ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))


FEAT_2STEP = ["chg_5y", "hdd", "cdd", "hdd_anom", "cdd_anom", "hdd_sq", "cdd_sq", "sin", "cos",
              "hdd_x_cos", "cdd_x_cos", "prev_dev_l", "prev_dev4_l", "hdd_prev", "holiday"]


def set_hdd(row, hdd):
    r = row.copy()
    r["hdd"] = hdd
    r["hdd_anom"] = hdd - r["hdd_5y"]
    r["hdd_sq"] = hdd ** 2 / 100
    r["hdd_x_cos"] = hdd * r["cos"]
    return r


def nowcast(panel, P, wx):
    """panel/P: curve_weather panel and walk-forward predictions; wx: weather frame (norm10)."""
    df = model1_frame()
    df["prev_dev_l"] = df["prev_dev"].shift(1)
    df["prev_dev4_l"] = df["prev_dev4"].shift(1)
    df = df.dropna(subset=FEAT_2STEP)
    rows = []
    for i, r in panel.iterrows():
        F = r.F
        T = F + pd.Timedelta(days=7)
        if T not in df.index or F < pd.Timestamp("2015-11-01"):
            continue
        tr = df[df.index <= F - pd.Timedelta(days=7)]
        if len(tr) < 52:
            continue
        m = _ridge().fit(tr[FEAT_2STEP], tr["chg"])
        row = df.loc[T]
        norm = wx.loc[T, "hdd_norm10"]
        out = {"F": F, "T": T, "winter": r.winter, "actual": row["chg"], "bench5": row["chg_5y"]}
        scen = {"oracle": row["hdd"], "clim": norm}
        for k, col in (("pers", "a1_pers"), ("market", "a1_combo3"), ("market_front", "a1_front"),
                       ("market+pers", "a1_combo3+pers")):
            v = P.loc[i, col]
            scen[k] = norm + v if not pd.isna(v) else np.nan
        for k, h in scen.items():
            if pd.isna(h):
                out[k] = np.nan
                continue
            x = set_hdd(row, max(h, 0.0))
            out[k] = float(m.predict(pd.DataFrame([x[FEAT_2STEP]]))[0])
        rows.append(out)
    return pd.DataFrame(rows)


def model1_with_curve(panel, P):
    df = model1_frame()
    feats0 = list(model1.FEATURES_V2)
    cur = panel.set_index("F")
    imp = P.set_index(panel.F)["combo3"]
    df["curve_wx"] = imp.reindex(df.index).fillna(0.0)
    df["curve_front"] = cur["r_front"].reindex(df.index).fillna(0.0)
    df["in_panel"] = df.index.isin(panel.F[P["combo3"].notna().values])
    start = pd.Timestamp("2016-01-01")
    preds = {}
    for name, feats in (("model1", feats0), ("model1+curve_wx", feats0 + ["curve_wx"]),
                        ("model1+curve_wx+front", feats0 + ["curve_wx", "curve_front"])):
        preds[name] = model1.walk_forward(df, _ridge, start, features=feats)
    out = pd.DataFrame(preds)
    out["actual"] = df["chg"]
    out["bench5"] = df["chg_5y"]
    out["in_panel"] = df["in_panel"]
    return out


def dm(e1, e2, lags=4):
    """Diebold-Mariano on absolute errors (e1 - e2 < 0 means model 1 better); HAC t."""
    import statsmodels.api as sm
    d = (np.abs(e1) - np.abs(e2)).dropna()
    m = sm.OLS(d.values, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(m.params[0]), float(m.tvalues[0]), float(m.pvalues[0])
