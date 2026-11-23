"""Shared data + statistics for v4 agent `hidden_info` (is premium_part hidden weather news?).

Dev data only: the panel is cut at 2021-12-31 on load; realised-weather targets whose window ends
after the cut (d + 15 calendar days > 2021-12-31) are set to NaN so no post-cut outcome is evaluated.
The news/premium split is agent 1's frozen walk-forward decomposition, read from its cache
(data/processed/research_v4_news_channel/decomposed_dev.csv / decomposed_weekly_dev.csv), unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parents[3]
CUTOFF = pd.Timestamp("2021-12-31")
NC = ROOT / "data/processed/research_v4_news_channel"
CACHE = ROOT / "data/processed/research_v4_hidden_info"
FIG = ROOT / "reports/agents/figures"
CACHE.mkdir(parents=True, exist_ok=True)
CHOOSE = (2013, 2017)   # premium_part exists from winter 2012-13 (first walk-forward vintage)
CHECK = (2018, 2022)
INS = {"front": "r_front", "second": "r_second", "hj": "d_hj"}
RNG = np.random.default_rng(20261123)
MAXGAP = pd.Timedelta(days=10)   # a shifted row further away than this is a season gap -> NaN


def winter_of(d):
    d = pd.to_datetime(d)
    return (d.dt.year + (d.dt.month >= 7)).astype(int)


def _shift(p: pd.DataFrame, col: str, k: int) -> pd.Series:
    s = p[col].shift(-k)
    dd = (p.date.shift(-k) - p.date).abs()
    return s.where(dd <= MAXGAP * max(1, abs(k) / 3))


def daily() -> pd.DataFrame:
    p = pd.read_csv(ROOT / "data/processed/v4/panel_dev.csv", parse_dates=["date"])
    p = p[p.date <= CUTOFF].reset_index(drop=True)                     # dev cut, immediately
    p["W"] = winter_of(p.date)
    p["dow"] = p.date.dt.dayofweek
    p["mon"] = (p.dow == 0).astype(float)
    # realised outcomes only if the whole 15-day window is inside the dev cut
    late = p.date + pd.Timedelta(days=15) > CUTOFF
    for k in (1, 2, 3):
        p.loc[late, f"real_w{k}"] = np.nan
        p[f"sur_w{k}"] = p[f"real_w{k}"] - p[f"lvl_w{k}"]                 # realised surprise vs d forecast
    p["lvl_tot"] = p.lvl_w1 + p.lvl_w2 + p.lvl_w3
    p["real_tot"] = p.real_w1 + p.real_w2 + p.real_w3
    p["sur_tot"] = p.sur_w1 + p.sur_w2 + p.sur_w3
    # future GEFS 00Z revisions (d+1..d+k) and lagged news
    for c in ["news_w0", "news_w1", "news_w2", "news_w3", "news_tot"]:
        for j in range(1, 6):
            p[f"{c}_f{j}"] = _shift(p, c, j)
        p[f"{c}_l1"] = _shift(p, c, -1)
    for k in (1, 3, 5):
        p[f"FN{k}"] = sum(p[f"news_tot_f{j}"] for j in range(1, k + 1))
        for w in (0, 1, 2, 3):
            p[f"FN{k}_w{w}"] = sum(p[f"news_w{w}_f{j}"] for j in range(1, k + 1))
    # next run's level and the realised outcome over the next run's windows
    for k in (1, 2, 3):
        p[f"lvl_w{k}_n1"] = _shift(p, f"lvl_w{k}", 1)
        p[f"real_w{k}_n1"] = _shift(p, f"real_w{k}", 1)
    p["real_tot_n1"] = p.real_w1_n1 + p.real_w2_n1 + p.real_w3_n1
    # ex-ante state
    same_front = p.front == p.front.shift(1)
    p["d_iv"] = (p.iv_front - p.iv_front.shift(1)).where(same_front)
    p["d_nv"] = p.nv_hj - p.nv_hj.shift(1)
    p["hj_prev"] = p.hj - p.d_hj
    # decomposition (agent 1, walk-forward)
    dec = pd.read_csv(NC / "decomposed_dev.csv", parse_dates=["date"])
    dec = dec[dec.date <= CUTOFF]
    for name, ins in INS.items():
        d = dec[dec.instrument == ins].set_index("date")
        for c in ["move", "news_part", "premium_part", "in_core"]:
            p[f"{c}_{name}"] = p.date.map(d[c])
        p[f"P_{name}"] = p[f"premium_part_{name}"]
        # 5-day sum d-4..d (all five rows must exist and be within one season)
        s = sum(_shift(p, f"P_{name}", -j) for j in range(0, 5))
        p[f"P5_{name}"] = s
        for h in (5, 10):   # subsequent move over d+1..d+h (persistence / reversal)
            p[f"fmove{h}_{name}"] = sum(_shift(p, f"move_{name}", j) for j in range(1, h + 1))
    return p


def weekly() -> pd.DataFrame:
    """One row per (week_end Friday, instrument) with weekly premium and Friday-dated weather columns."""
    p = daily()
    w = pd.read_csv(NC / "decomposed_weekly_dev.csv", parse_dates=["week_end"])
    w = w[w.week_end <= CUTOFF]
    keep = ["date", "W", "lvl_w1", "lvl_w2", "lvl_w3", "lvl_tot", "real_w1", "real_w2", "real_w3",
            "real_tot", "sur_tot", "news_tot", "FN5", "FN5_w1", "FN5_w3", "mon", "hj_prev", "hj"]
    q = p[keep].rename(columns={"date": "week_end"})
    w = w.merge(q, on="week_end", how="left")
    # week's own GEFS news (sum over the week's trading days) as a control
    p["wk"] = p.date + pd.to_timedelta((4 - p.dow) % 7, unit="D")
    wn = p.groupby("wk").news_tot.sum(min_count=1).rename("news_week")
    w["news_week"] = w.week_end.map(wn)
    return w


# ---------------------------------------------------------------- statistics
def ols_fit(y, X):
    A = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ b
    return b, 1 - e @ e / ((y - y.mean()) @ (y - y.mean()))


def incr_test(df: pd.DataFrame, y: str, base: list[str], add: list[str], lags: int,
              B: int = 1000, wcol: str = "W") -> dict:
    """y on base (+ add). Returns coef of add[0] (per 1 sd of add[0]), HAC t, winter-bootstrap CI,
    in-sample incremental R2 and walk-forward (expanding by winter, >= 2 training winters) OOS dR2."""
    d = df.dropna(subset=[y] + base + add).copy()
    sd = d[add[0]].std()
    d["_z"] = d[add[0]] / sd
    addz = ["_z"] + add[1:]
    Xb, Xf, Y = d[base].values, d[base + addz].values, d[y].values
    _, r2b = ols_fit(Y, Xb)
    bf, r2f = ols_fit(Y, Xf)
    m = sm.OLS(Y, sm.add_constant(Xf)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    k = 1 + len(base)
    ws = d[wcol].values
    uw = np.unique(ws)
    idx = {w: np.where(ws == w)[0] for w in uw}
    bs, bsr = [], []
    for _ in range(B):
        ii = np.concatenate([idx[w] for w in RNG.choice(uw, len(uw))])
        bb, rr = ols_fit(Y[ii], Xf[ii])
        _, rb = ols_fit(Y[ii], Xb[ii]) if len(base) else (None, 0.0)
        bs.append(bb[k]); bsr.append(rr - rb)
    out = {"y": y, "x": add[0], "n": len(d), "nwin": len(uw), "coef_sd": bf[k], "t_hac": m.tvalues[k],
           "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5),
           "p_boot_le0": float(np.mean(np.array(bs) <= 0)),
           "dR2_in": r2f - r2b, "dR2_lo": np.percentile(bsr, 2.5), "dR2_hi": np.percentile(bsr, 97.5)}
    # walk-forward OOS
    pred_b, pred_f, ytr = np.full(len(d), np.nan), np.full(len(d), np.nan), np.full(len(d), np.nan)
    for w in uw:
        tr = ws < w
        if len(np.unique(ws[tr])) < 2:
            continue
        te = ws == w
        b0, _ = ols_fit(Y[tr], Xb[tr]) if len(base) else (np.array([Y[tr].mean()]), 0)
        b1, _ = ols_fit(Y[tr], Xf[tr])
        pred_b[te] = np.column_stack([np.ones(te.sum()), Xb[te]]) @ b0
        pred_f[te] = np.column_stack([np.ones(te.sum()), Xf[te]]) @ b1
        ytr[te] = Y[tr].mean()
    for nm, (a, b_) in {"choose": CHOOSE, "check": CHECK}.items():
        mm = (ws >= a) & (ws <= b_) & ~np.isnan(pred_f)
        if mm.sum() < 30:
            out[f"oos_dR2_{nm}"] = np.nan; continue
        s0 = ((Y[mm] - ytr[mm]) ** 2).sum()
        out[f"oos_dR2_{nm}"] = (((Y[mm] - pred_b[mm]) ** 2).sum() - ((Y[mm] - pred_f[mm]) ** 2).sum()) / s0
        out[f"n_oos_{nm}"] = int(mm.sum())
    return out
