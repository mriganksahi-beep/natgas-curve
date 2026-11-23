"""Study v4 frozen tests on the locked 2000-2010 panel (spec: reports/synthesis_v4.md, committed 264f4dc
before this runner touched locked data).

  python -m src.holdout.v4_locked dev      # reproduce development reference values (no locked data read)
  python -m src.holdout.v4_locked locked   # the one-time run on data/processed/v4_locked/panel_pre2010.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.v4_news_channel.decompose import decompose, decompose_weekly, load_dev_panel
from src.research.v4_premium_shocks.common import nw_ols
from src.research.v4_premium_shocks.frozen_test import _onesided, hps1, hps1b, hps2, same_contract_fut

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/holdout"
RNG = np.random.default_rng(20261123)


def winter(d: pd.Series) -> pd.Series:
    return d.dt.year + (d.dt.month >= 7).astype(int)


def prep(panel: pd.DataFrame, locked: bool) -> pd.DataFrame:
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"]); p["prev_date"] = pd.to_datetime(p["prev_date"])
    if locked:
        p = p[p.date >= "2000-10-01"].reset_index(drop=True)
        gap = (p.date - p.prev_date).dt.days > 4
        p.loc[gap, ["r_front", "r_second", "d_hj"]] = np.nan
        p.attrs["n_gap_rows"] = int(gap.sum())
    return p


# ------------------------------------------------------------------ P1 + extras
def p1(panel):
    base = hps1(panel)
    p = panel.reset_index(drop=True).copy()
    w = decompose_weekly(p, "r_second", mode="frozen")
    w = w[w.week_end.dt.month.isin([11, 12, 1, 2])]
    extra = {}
    for h in (5, 10, 15, 20):
        f = same_contract_fut(p, "second", h).rename("fut")
        x = w.merge(p[["date"]].assign(fut=f.values), left_on="week_end", right_on="date").dropna(subset=["fut"])
        b, se = nw_ols(x.fut, x[["news_part", "premium_part"]], 5)
        extra[f"b_prem_h{h}"] = (b[2], b[2] / se[2])
        if h == 20:
            bd, sed = nw_ols(x.fut, x[["move", "premium_part"]], 5)  # coef on premium = b_P - b_N
            extra["bP_minus_bN"] = (bd[2], bd[2] / sed[2])
            x["win"] = winter(x.week_end)
            lowo = {}
            for wv in sorted(x.win.unique()):
                y = x[x.win != wv]
                bb, ss = nw_ols(y.fut, y[["news_part", "premium_part"]], 5)
                lowo[int(wv)] = round(float(bb[2]), 3)
            extra["lowo_b_prem"] = lowo
            extra["winters"] = sorted(int(v) for v in x.win.unique())
    base.update(extra)
    return base


# ------------------------------------------------------------------ S1
def r2(move, prem):
    move, prem = np.asarray(move), np.asarray(prem)
    return 1 - np.sum((prem - prem.mean()) ** 2) / np.sum((move - move.mean()) ** 2)


def s1(panel):
    d = decompose(panel, "r_front", mode="frozen")
    d = d[d.in_core].dropna(subset=["move", "premium_part"])
    d["win"] = winter(d.date)
    daily = r2(d.move, d.premium_part)
    wins = d.win.unique()
    boots = []
    for _ in range(2000):
        pick = RNG.choice(wins, len(wins))
        s = pd.concat([d[d.win == v] for v in pick])
        boots.append(r2(s.move, s.premium_part))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    w = decompose_weekly(panel, "r_front", mode="frozen")
    w = w[w.week_end.dt.month.isin([11, 12, 1, 2])].dropna(subset=["move", "premium_part"])
    weekly = r2(w.move, w.premium_part)
    wh = decompose_weekly(panel, "d_hj", mode="frozen")
    wh = wh[wh.week_end.dt.month.isin([11, 12, 1, 2])].dropna(subset=["move", "premium_part"])
    return {"daily_r2": daily, "daily_ci": [lo, hi], "n_daily": len(d), "weekly_r2": weekly, "n_weekly": len(w),
            "hj_weekly_r2_report": r2(wh.move, wh.premium_part) if len(wh) else np.nan,
            "pass": bool(daily > 0.02 and lo > 0 and weekly > 0.08)}


# ------------------------------------------------------------------ S3 + E-rule log
def carry(panel, rule="plain", theta=1.0, scale=36.85, size=1.0):
    p = panel[panel.hj.notna()].copy()
    p["x"] = ((p.lvl_w2 + p.lvl_w3) / scale).rolling(5, min_periods=5).mean()
    out = []
    for Y in sorted(p.hj_year.dropna().unique().astype(int)):
        g = p[(p.hj_year == Y)]
        g = g[g.date >= pd.Timestamp(Y - 1, 12, 1)]
        if g.empty:
            continue
        if rule == "E":
            ok = g[g.x <= theta]
            if ok.empty:
                out.append({"winter": Y, "traded": False, "pnl80": 0.0, "pnl160": 0.0, "dd": 0.0}); continue
            g = g[g.date >= ok.date.iloc[0]]
        s0 = g.hj.iloc[0]
        mtm = -(g.hj - s0) * 10_000 * size
        dd = float((mtm - mtm.cummax()).min())
        out.append({"winter": Y, "traded": True, "entry": str(g.date.iloc[0].date()), "exit": str(g.date.iloc[-1].date()),
                    "pnl80": float(mtm.iloc[-1] - 80 * size), "pnl160": float(mtm.iloc[-1] - 160 * size), "dd": dd})
    return pd.DataFrame(out)


def s3(panel, first_winter):
    t = carry(panel)
    t = t[t.winter >= first_winter]
    m = t.pnl80.mean()
    boots = [t.pnl80.sample(len(t), replace=True, random_state=int(RNG.integers(1e9))).mean() for _ in range(2000)]
    half = carry(panel, size=0.5); half = half[half.winter >= first_winter]
    E = carry(panel, rule="E"); E = E[E.winter >= first_winter]
    return {"table": t.round(0).to_dict("records"), "mean80": m, "mean160": t.pnl160.mean(),
            "ci80": list(np.percentile(boots, [2.5, 97.5])), "n_pos": int((t.pnl80 > 0).sum()), "n": len(t),
            "worst_dd": t.dd.min(), "half_mean80": half.pnl80.mean(), "half_worst_dd": half.dd.min(),
            "E_log": {"mean80": E.pnl80.mean(), "worst_dd": E.dd.min(), "table": E.round(0).to_dict("records")},
            "pass": bool(m > 0)}


# ------------------------------------------------------------------ H-HI-1 log
def hhi1(panel):
    d = decompose(panel, "r_front", mode="frozen").sort_values("date").reset_index(drop=True)
    d = d[d.in_core]
    d["news_next"] = d.news_part.shift(-1)
    d = d.dropna(subset=["news_next", "premium_part"])
    b, se = nw_ols(d.news_next, d[["premium_part"]], 2)
    return {"b": b[1], "t": b[1] / se[1], "n": len(d)}


def run(which: str):
    if which == "dev":
        panel = prep(load_dev_panel(), locked=False)
        first = 2012
    else:
        panel = prep(pd.read_csv(ROOT / "data/processed/v4_locked/panel_pre2010.csv"), locked=True)
        first = 2001
    res = {"sample": which, "n_rows": len(panel), "n_gap_rows_nulled": panel.attrs.get("n_gap_rows", 0),
           "P1": p1(panel), "S1": s1(panel), "S2": hps1b(panel), "S3": s3(panel, first),
           "LOG_HPS2": hps2(panel), "LOG_HHI1": hhi1(panel)}
    txt = json.dumps(res, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    out = OUT / f"v4_{which}_results.json"
    out.write_text(txt)
    print(txt)


if __name__ == "__main__":
    run(sys.argv[1])
