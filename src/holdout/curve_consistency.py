"""Frozen holdout test H1: extreme-kink fade (spec frozen in reports/synthesis_v2.md).

    python -m src.holdout.curve_consistency check     # validation reproduction, end 2021-12-31
    python -m src.holdout.curve_consistency holdout   # ONE run, entries 2022-01-03 .. last date

Wraps src/research/curve_consistency/* without editing it; only the end date / year ranges are
extended. Cost convention (frozen after review): k ticks per contract per side, so a 1:-2:1 fly
(4 contracts) round trip costs 4 * 2 * k * $10.
"""
import sys

import numpy as np
import pandas as pd

from src.research.curve_consistency.data import RAW, NPOS
from src.research.curve_consistency import kinks as K
from src.research.curve_consistency.reversion import summarize

THR = 0.0424          # frozen 2011-17 99th percentile of |fA| (423.96 bp)
H = 20
TICKS_PER_SIDE = (2, 4, 6, 10)


def load_curve(end):
    """Copy of data.load_curve with CUTOFF = end + 1 day."""
    cutoff = pd.Timestamp(end) + pd.Timedelta(days=1)
    d = pd.read_csv(RAW / "ng_curve_daily.csv", parse_dates=["date"])
    d = d[(d.date < cutoff) & (d.date >= "2010-06-01")].copy()
    d["k"] = d.contract_year * 12 + d.contract_month - 1
    dates = np.sort(d.date.unique())
    didx = pd.Series(np.arange(len(dates)), index=dates)
    d["di"] = didx.loc[d.date].values
    last = d.groupby("k").di.max()
    exp_in_sample = d.groupby("k").date.max() < (cutoff - pd.Timedelta(days=5))
    d["dte"] = d.k.map(last) - d.di
    d.loc[~d.k.map(exp_in_sample).astype(bool), "dte"] = 999
    d = d[d.settle.notna() & (d.settle > 0)]
    d = d.sort_values(["di", "k"])
    d["pos"] = d.groupby("di").cumcount() + 1
    d = d[d.pos <= NPOS + 2]
    return d.reset_index(drop=True)


def fit_all(d, last_year):
    """Copy of kinks.fit_all (variant A only) with the year range extended to last_year."""
    f = K.prepare(d)
    f["year"] = f.date.dt.year
    shapes = {}
    for Y in range(2011, last_year + 1):
        sub = f[(f.date >= f"{Y - K.N_YEARS}-01-01") & (f.date < f"{Y}-01-01")]
        sub = sub[sub.di.isin(sub.di.unique()[::5])]
        shapes[Y] = K.estimate_shape(sub)
    f = f[f.year >= 2011].sort_values(["di", "k"]).reset_index(drop=True)
    rA = np.full(len(f), np.nan)
    for di, ii in f.groupby("di").indices.items():
        g = f.iloc[ii]
        if len(g) < 8:
            continue
        s = shapes[g.year.iat[0]][g.cm.values - 1]
        rA[ii], _ = K.fit_day(g.x.values, g.ly.values, s, False)
    f["kA"] = rA
    f["kB"] = rA          # add_flies expects the column; unused
    return f, pd.DataFrame(shapes, index=range(1, 13))


def flies(f):
    f = f.sort_values(["di", "k"])
    g = f.groupby("di")
    prv, nxt = g.shift(1), g.shift(-1)
    ok = (prv.k == f.k - 1) & (nxt.k == f.k + 1)
    return pd.DataFrame({"date": f.date, "di": f.di, "k": f.k, "cm": f.cm, "tau": f.tau,
                         "B": f.settle - (prv.settle + nxt.settle) / 2,
                         "fA": f.kA - (prv.kA + nxt.kA) / 2})[ok].copy()


def trades(fl, start, end):
    e = fl[(fl.date >= start) & (fl.date <= end) & (fl.fA.abs() > THR)]
    ex = fl[["di", "k", "B", "date"]].rename(columns={"B": "B_h", "date": "exit_date"})
    ex["di"] -= H
    j = e.merge(ex, on=["di", "k"])          # no exit row (expiry window or beyond data) -> dropped
    j = j[j.exit_date <= end]
    j["gross"] = -np.sign(j.fA) * 2 * (j.B_h - j.B) * 10_000
    return j


def episodes(j, k=4):
    net = j.gross - 4 * 2 * k * 10
    dd = np.sort(j.di.unique())
    ep = dict(zip(dd, np.cumsum(np.r_[1, np.diff(dd) > H])))
    jj = j.assign(ep=j.di.map(ep), net=net)
    MC = "FGHJKMNQUVXZ"
    return jj.groupby("ep").agg(
        entry_month=("date", lambda s: s.min().strftime("%Y-%m")),
        last_entry=("date", lambda s: s.max().strftime("%Y-%m-%d")),
        trades=("net", "size"),
        centre=("k", lambda s: ", ".join(f"NG{MC[x % 12]}{(x // 12) % 100:02d}" for x in
                                        s.value_counts().index[:3])),
        net_mean=("net", "mean"))


def run(start, end):
    d = load_curve(end)
    f, shapes = fit_all(d, pd.Timestamp(end).year)
    fl = flies(f)
    j = trades(fl, start, end)
    rows = [dict(cost="gross", **summarize(j, ticks=0))]
    for k in TICKS_PER_SIDE:
        rows.append(dict(cost=f"{k} ticks/side", **summarize(j, ticks=2 * k)))   # 4*(2k)*$10
    res = pd.DataFrame(rows)
    ep = episodes(j)
    n_ep, pos = len(ep), int((ep.net_mean > 0).sum())
    m4 = res.loc[res.cost == "4 ticks/side", "mean"].iat[0]
    if n_ep < 3:
        verdict = "INCONCLUSIVE (< 3 episodes)"
    else:
        verdict = "PASS" if (m4 > 0 and pos / n_ep >= 0.6) else "FAIL"
    return res, ep, verdict, shapes, j


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    mode = sys.argv[1]
    if mode == "check":
        start, end = "2018-01-01", "2021-12-31"
    elif mode == "holdout":
        start, end = "2022-01-03", "2026-11-17"
    else:
        raise SystemExit("mode: check | holdout")
    res, ep, verdict, shapes, j = run(start, end)
    print(f"{mode}: {start} .. {end}   trades={len(j)}  episodes={len(ep)}  "
          f"positive={(ep.net_mean > 0).sum()}  verdict={verdict}")
    print(res.round(1).to_string())
    print(ep.round(0).to_string())
    print("centre-month mix:", j.cm.value_counts().to_dict())
    out = f"/home/claude/natgas-curve/data/processed/research_curve_consistency/holdout_{mode}"
    res.to_csv(out + "_results.csv"); ep.to_csv(out + "_episodes.csv"); j.to_csv(out + "_trades.csv")
