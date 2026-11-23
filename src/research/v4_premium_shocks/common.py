"""Shared data and statistics for v4 agent `premium_shocks` (dev data only, dates <= 2021-12-31).

Builds one daily frame with, per instrument (front, second, hj):
  move      today's same-contract move (r_front / r_second in %-log, d_hj in $/MMBtu)
  news, prem  walk-forward daily split (v4_news_channel.decompose, frozen spec)
  wnews, wprem  "rolling weekly" split: the walk-forward WEEKLY coefficients (fitted on Fri-Fri weeks of
             winters < W, exactly as decompose_weekly) applied to trailing k-day sums of the news terms
  fut_h     move of the SAME contract from the d settle to the d+h settle (h trading days ahead);
            NaN if that contract is not in the panel at d+h (front expiry) or the target March changes
  nfut_h    sum of daily news parts d+1..d+h (descriptive control, not tradable)
No row after 2021-12-31 is ever loaded (load_dev_panel cuts immediately).
"""
from __future__ import annotations

import warnings
warnings.simplefilter("ignore", category=Warning)

import numpy as np
import pandas as pd

from src.research.v4_news_channel.decompose import (
    decompose, load_dev_panel, winter_of, weekly_table, WEEKLY_SPEC, ols, FIT_MONTHS, CUTOFF, ROOT)

CACHE = ROOT / "data/processed/research_v4_premium_shocks"
FIG = ROOT / "reports/agents/figures"
CHOOSE = (2013, 2017)        # winters 2012-13..2016-17 (news_part exists from 2012-13)
CHECK = (2018, 2022)         # 2017-18..Dec 2021
HS = (1, 2, 5, 10, 20)
INS = {"front": "r_front", "second": "r_second", "hj": "d_hj"}
TICK = 10.0                  # $ per tick per contract ($0.001 x 10,000 MMBtu)
MULT = 10000.0               # $ per $1/MMBtu per contract


def build(max_h: int = 20) -> pd.DataFrame:
    p = load_dev_panel()
    p["W"] = winter_of(p.date)
    p["month"] = p.date.dt.month
    p["mon"] = (p.date.dt.dayofweek == 0).astype(float)
    p["hj_prev"] = p.hj - p.d_hj
    n = len(p)
    # contract price matrix from the panel's front/second columns
    long = pd.concat([p[["front", "px_front"]].rename(columns={"front": "cid", "px_front": "px"}).assign(i=p.index),
                      p[["second", "px_second"]].rename(columns={"second": "cid", "px_second": "px"}).assign(i=p.index)])
    PX = long.pivot_table(index="i", columns="cid", values="px").reindex(range(n))
    cols = {c: j for j, c in enumerate(PX.columns)}
    A = PX.to_numpy()
    for name, cidcol in (("front", "front"), ("second", "second")):
        cj = p[cidcol].map(cols).to_numpy()
        p0 = A[np.arange(n), cj]
        p[f"px0_{name}"] = p0
        for h in range(1, max_h + 1):
            idx = np.arange(n) + h
            ok = idx < n
            ph = np.full(n, np.nan)
            ph[ok] = A[idx[ok], cj[ok]]
            p[f"fut{h}_{name}"] = 100 * np.log(ph / p0)
    hj = p.hj.to_numpy(); hy = p.hj_year.to_numpy()
    for h in range(1, max_h + 1):
        f = np.full(n, np.nan)
        f[:-h] = np.where(hy[h:] == hy[:-h], hj[h:] - hj[:-h], np.nan)
        p[f"fut{h}_hj"] = f
    # daily walk-forward split
    for name, col in INS.items():
        d = decompose(p, col)[["date", "news_part", "premium_part", "in_core"]]
        d = d.rename(columns={"news_part": f"news_{name}", "premium_part": f"prem_{name}", "in_core": f"core_{name}"})
        p = p.merge(d, on="date", how="left")
        p[f"move_{name}"] = p[col]
        nz = p[f"news_{name}"]
        p[f"nlag_{name}"] = nz.shift(1)
        cs = nz.fillna(0).cumsum()
        for h in range(1, max_h + 1):
            p[f"nfut{h}_{name}"] = cs.shift(-h) - cs
    p = add_rolling_weekly(p.copy())
    p = add_weekly_split(p.copy())
    return p.copy()


def add_weekly_split(p: pd.DataFrame) -> pd.DataFrame:
    """Friday rows: the walk-forward weekly split (decompose_weekly) of the week ending at d,
    previous week's weekly news, and the sum of the NEXT weeks' weekly news over (d, d+h] (control)."""
    from src.research.v4_news_channel.decompose import decompose_weekly
    for name, col in INS.items():
        w = decompose_weekly(p, col)[["week_end", "move", "news_part", "premium_part"]]
        w = w.rename(columns={"week_end": "date", "move": f"wkmove_{name}", "news_part": f"wknews_{name}",
                              "premium_part": f"wkprem_{name}"})
        w[f"wknlag_{name}"] = w[f"wknews_{name}"].shift(1)
        p = p.merge(w, on="date", how="left")
        cs = p[f"wknews_{name}"].fillna(0).cumsum()
        for h in (5, 10, 20):
            p[f"wknfut{h}_{name}"] = cs.shift(-h) - cs
    return p


def weekly_coefs(p: pd.DataFrame, col: str) -> dict:
    """Walk-forward weekly coefficients per winter W (fitted on Nov-Feb weeks of winters < W, >= 2 winters)."""
    spec = WEEKLY_SPEC[col]
    g = weekly_table(p, col, spec)
    W = winter_of(g.date).to_numpy()
    fit = g.date.dt.month.isin(FIT_MONTHS).to_numpy()
    X = g[spec["terms"]].to_numpy(float); y = g[col].to_numpy(float)
    out = {}
    for w in range(2011, 2023):
        tr = fit & (W < w)
        if len(np.unique(W[tr])) >= 2:
            out[w] = ols(X[tr], y[tr])[1:]
    return out


def add_rolling_weekly(p: pd.DataFrame, ks=(1, 3, 5)) -> pd.DataFrame:
    """wnews_k: weekly walk-forward betas applied to the trailing k-day sums of daily news terms."""
    if "news_tot" not in p:
        p["news_tot"] = p[["news_w0", "news_w1", "news_w2", "news_w3"]].sum(axis=1, min_count=4)
    for name, col in INS.items():
        spec = WEEKLY_SPEC[col]
        B = weekly_coefs(p, col)
        for k in ks:
            X = p[spec["terms"]].rolling(k).sum().to_numpy(float)
            if spec.get("scale"):
                g = p.hj_prev.shift(k - 1).clip(lower=0) + spec["scale"]["c"]
                X = X * g.to_numpy()[:, None]
            nw = np.full(len(p), np.nan)
            for w, b in B.items():
                m = (p.W == w).to_numpy()
                nw[m] = X[m] @ b
            mv = p[col].rolling(k).sum()
            if name == "hj":     # do not sum across a change of target March
                mv = mv.where(p.hj_year == p.hj_year.shift(k - 1))
            p[f"cmove{k}_{name}"] = mv
            p[f"wnews{k}_{name}"] = nw
            p[f"wprem{k}_{name}"] = mv - nw
            p[f"cnews{k}_{name}"] = p[f"news_{name}"].rolling(k).sum()
            p[f"cprem{k}_{name}"] = p[f"prem_{name}"].rolling(k).sum().where(mv.notna())
    return p


def win(df: pd.DataFrame, w: tuple[int, int]) -> pd.DataFrame:
    return df[(df.W >= w[0]) & (df.W <= w[1])]


# ------------------------------------------------------------------ statistics
def nw_ols(y, X, lags):
    """OLS with intercept; Newey-West (Bartlett) HAC standard errors. Returns (b, se) incl. intercept."""
    y = np.asarray(y, float); X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    Z = X * u[:, None]
    S = Z.T @ Z
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Z[L:].T @ Z[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return b, np.sqrt(np.diag(V))


def boot_slopes(df: pd.DataFrame, y: str, xs: list[str], B: int = 1000, seed: int = 7) -> np.ndarray:
    """Winter-block bootstrap of OLS slopes; returns B x len(xs)."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("W")]
    mats = [(np.column_stack([np.ones(len(g)), g[xs].to_numpy(float)]), g[y].to_numpy(float)) for g in groups]
    XtX = [X.T @ X for X, _ in mats]; Xty = [X.T @ yy for X, yy in mats]
    out = np.full((B, len(xs)), np.nan)
    k = len(groups)
    for b in range(B):
        idx = rng.integers(0, k, k)
        A = sum(XtX[i] for i in idx); c = sum(Xty[i] for i in idx)
        try:
            out[b] = np.linalg.solve(A, c)[1:]
        except np.linalg.LinAlgError:
            pass
    return out


def reg(df: pd.DataFrame, y: str, xs: list[str], lags: int, B: int = 1000) -> dict:
    d = df.dropna(subset=[y] + xs)
    if len(d) < 30:
        return {"n": len(d)}
    b, se = nw_ols(d[y], d[xs], lags)
    bs = boot_slopes(d, y, xs, B)
    r = {"n": len(d), "n_w": d.W.nunique()}
    for j, x in enumerate(xs):
        r[f"b_{x}"] = b[j + 1]; r[f"t_{x}"] = b[j + 1] / se[j + 1]
        r[f"lo_{x}"], r[f"hi_{x}"] = np.nanpercentile(bs[:, j], [2.5, 97.5])
    return r
