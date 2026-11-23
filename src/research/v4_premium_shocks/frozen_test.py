"""Proposed frozen tests H-PS1 / H-PS2 for the locked 2000-2010 panel (v4_premium_shocks).

The functions take a panel DataFrame and never read any file under data/processed/v4_locked/ themselves.
`main()` runs a DRY RUN on the development panel (dates <= 2021-12-31) with the frozen (all-dev) coefficients,
which is in-sample by construction and only checks that the code works and what dev values look like.

H-PS1 (primary, reversal slope, weekly split):
    rows   : Friday week-ends of decompose_weekly(panel, "r_second", mode="frozen"), months Nov-Feb,
             weeks with >= 4 eligible days, fut20 available
    y      : fut20 = 100*ln(P_{d+20}/P_d) of the contract that is `second` at the Friday settle d
             (same contract; 20 trading days)
    x      : weekly news_part, weekly premium_part (frozen weekly coefficients, frozen_coefs.json)
    stat   : OLS slope on premium_part, Newey-West lags 5 (weeks), one-sided p (H1: slope < 0)
    pass   : slope < 0 with one-sided p < 0.05, AND news slope not significantly negative (t > -1.645),
             AND premium slope < naive slope (raw weekly move, same rows) [descriptive, reported]
H-PS1b (design criterion, daily split, h = 5 and 10, descriptive controls):
    fut_h ~ news + prem + news(d-1) + sum news(d+1..d+h) + Monday,   NW lags h+2, one-sided p < 0.05
H-PS2 (trading): second-month fade of the rolling weekly-split premium, parameters frozen from the
    choose window: k = 5, z = 1.5, h = 5. Signal at the d settle (Nov-Feb):
        wprem5 = sum_{5 days} r_second - sum_j b_j^weekly * sum_{5 days} news_wj   (frozen weekly b)
        z = wprem5 / (sd60 * sqrt 5),  sd60 = sd of r_second over the 60 trading days to d-1
        if |z| >= 1.5: position -sign(wprem5), 1 contract, exit at the d+5 settle, same contract;
        one position at a time. Cost 2 ticks/side ($40 round trip); also report 4.
    pass   : >= 20 trades, mean net $/trade > 0 at 2 ticks, AND mean net > the naive fade
             (same rule on the raw 5-day move) over the same sample.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.research.v4_news_channel.decompose import decompose, decompose_weekly, FROZEN, load_dev_panel
from .common import nw_ols, CACHE

K, THR, H = 5, 1.5, 5
COST = {2: 40.0, 4: 80.0}


def same_contract_fut(panel: pd.DataFrame, which: str, h: int) -> pd.Series:
    """100*ln price change from d to d+h of the contract that is `which` (front/second) at d."""
    p = panel.reset_index(drop=True)
    n = len(p)
    long = pd.concat([p[["front", "px_front"]].set_axis(["cid", "px"], axis=1).assign(i=p.index),
                      p[["second", "px_second"]].set_axis(["cid", "px"], axis=1).assign(i=p.index)])
    PX = long.pivot_table(index="i", columns="cid", values="px").reindex(range(n))
    cols = {c: j for j, c in enumerate(PX.columns)}
    A = PX.to_numpy(); cj = p[which].map(cols).to_numpy()
    p0 = A[np.arange(n), cj]
    ph = np.full(n, np.nan); ok = np.arange(n) + h < n
    ph[ok] = A[(np.arange(n) + h)[ok], cj[ok]]
    return pd.Series(100 * np.log(ph / p0), index=panel.index)


def _onesided(b, se):
    t = b / se
    return t, norm.cdf(t)


def hps1(panel: pd.DataFrame) -> dict:
    panel = panel.reset_index(drop=True).copy()
    panel["fut20"] = same_contract_fut(panel, "second", 20)
    w = decompose_weekly(panel, "r_second", mode="frozen")
    w = w[w.week_end.dt.month.isin([11, 12, 1, 2])].merge(panel[["date", "fut20"]], left_on="week_end",
                                                           right_on="date").dropna(subset=["fut20"])
    b, se = nw_ols(w.fut20, w[["news_part", "premium_part"]], 5)
    tp, pp = _onesided(b[2], se[2]); tn, _ = _onesided(b[1], se[1])
    bn, sen = nw_ols(w.fut20, w[["move"]], 5)
    return {"n_weeks": len(w), "b_prem": b[2], "t_prem": tp, "p_one_sided": pp, "b_news": b[1], "t_news": tn,
            "b_naive": bn[1], "t_naive": bn[1] / sen[1],
            "pass": bool(b[2] < 0 and pp < 0.05 and tn > -1.645)}


def hps1b(panel: pd.DataFrame) -> list[dict]:
    panel = panel.reset_index(drop=True).copy()
    d = decompose(panel, "r_second", mode="frozen")
    d = d.merge(panel[["date"]].assign(i=np.arange(len(panel))), on="date")
    news = pd.Series(np.nan, index=panel.index); news.iloc[d.i] = d.news_part.to_numpy()
    cs = news.fillna(0).cumsum()
    out = []
    for h in (5, 10):
        f = same_contract_fut(panel, "second", h)
        x = pd.DataFrame({"y": f.iloc[d.i].to_numpy(), "news": d.news_part.to_numpy(), "prem": d.premium_part.to_numpy(),
                          "nlag": news.shift(1).iloc[d.i].to_numpy(), "nfut": (cs.shift(-h) - cs).iloc[d.i].to_numpy(),
                          "mon": (d.date.dt.dayofweek == 0).astype(float).to_numpy(), "core": d.in_core.to_numpy()})
        x = x[x.core].dropna()
        b, se = nw_ols(x.y, x[["news", "prem", "nlag", "nfut", "mon"]], h + 2)
        tp, pp = _onesided(b[2], se[2])
        out.append({"h": h, "n": len(x), "b_prem": b[2], "t_prem": tp, "p_one_sided": pp, "b_news": b[1],
                    "t_news": b[1] / se[1], "pass": bool(b[2] < 0 and pp < 0.05)})
    return out


def hps2(panel: pd.DataFrame) -> dict:
    p = panel.reset_index(drop=True).copy()
    cf = json.loads(FROZEN.read_text())["weekly"]["r_second"]
    terms, bw = cf["terms"], np.asarray(cf["slopes"])
    d = decompose(p, "r_second", mode="frozen")[["date", "in_core"]]
    p = p.merge(d, on="date", how="left")
    p["elig"] = p.in_core.fillna(False).astype(bool) & p[terms].notna().all(axis=1)
    p["sd60"] = p.r_second.rolling(60, min_periods=40).std().shift(1)
    mv5 = p.r_second.rolling(K).sum()
    wn5 = p[terms].rolling(K).sum().to_numpy() @ bw
    fut = same_contract_fut(p, "second", H)
    pnl = p.px_second * (np.exp(fut / 100) - 1) * 10000
    res = {}
    for name, x in (("split", mv5 - wn5), ("naive", mv5)):
        z = x / (p.sd60 * np.sqrt(K))
        ok = (p.elig & (z.abs() >= THR) & pnl.notna()).to_numpy()
        trades, nxt = [], -1
        for i in np.flatnonzero(ok):
            if i >= nxt:
                trades.append(-np.sign(x.iloc[i]) * pnl.iloc[i]); nxt = i + H
        t = np.asarray(trades)
        res[name] = {"n": len(t), "mean2": float(np.mean(t - COST[2])) if len(t) else np.nan,
                     "mean4": float(np.mean(t - COST[4])) if len(t) else np.nan,
                     "hit2": float(np.mean(t - COST[2] > 0)) if len(t) else np.nan,
                     "worst2": float(np.min(t - COST[2])) if len(t) else np.nan}
    s, nv = res["split"], res["naive"]
    res["pass"] = bool(s["n"] >= 20 and s["mean2"] > 0 and s["mean2"] > nv["mean2"])
    return res


def main():
    dev = load_dev_panel()                      # dev cut applied inside; dry run only (in-sample coefficients)
    out = {"NOTE": "DRY RUN on development data with frozen all-dev coefficients: in-sample, not evidence",
           "H-PS1": hps1(dev), "H-PS1b": hps1b(dev), "H-PS2": hps2(dev)}
    (CACHE / "frozen_test_dryrun_dev.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
