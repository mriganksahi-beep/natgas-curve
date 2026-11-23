"""v4_widowmaker: can weather-forecast information improve the short March-April premium trade?

    PYTHONPATH=. python -m src.research.v4_widowmaker.run

Scores every rule in rules.grid() for entry in {nov, dec}, costs {2, 4} ticks per leg per side and
execution lag {0, 1}; selects one cell per family on TRAIN (2010-11..2016-17) by Calmar
(mean / |worst DD|), subject to kept >= 0.4; checks it on 2017-18..2020-21 (+ truncated 2021-22).
Outputs in data/processed/research_v4_widowmaker/.
"""
import json
import time

import numpy as np
import pandas as pd

from src.research.v4_widowmaker import data as D
from src.research.v4_widowmaker import rules as R

NB = 10_000
RNG = np.random.default_rng(20261123)
WINDOWS = {"train": D.TRAIN, "check": D.CHECK, "check+trunc": D.CHECK + [D.TRUNC],
           "all": D.TRAIN + D.CHECK, "all+trunc": D.YEARS}


def maxdd(cum):
    c = np.r_[0.0, cum]
    return float(np.min(c - np.maximum.accumulate(c)))


def score_all():
    rows, paths = [], {}
    for entry in ("nov", "dec"):
        W = D.winters(entry)
        for fam, name, fn, kw in R.grid():
            lags = [1] if fam == "D_addon" else [0, 1]
            if fam == "bench":
                lags = [0]
            for lag in lags:
                for tk in (2, 4):
                    for Y, w in W.items():
                        kw2 = dict(kw)
                        kw2["lag"] = lag
                        pos = fn(w, **kw2)
                        d = R.pnl(w, pos, D.SIDE[tk])
                        cum = np.cumsum(d)
                        rows.append(dict(entry=entry, family=fam, name=name, lag=lag, ticks=tk, Y=Y,
                                         pnl=cum[-1], dd=maxdd(cum), avg_pos=float(np.mean(pos[:-1])),
                                         turnover=float(np.abs(np.diff(np.r_[0, pos[:-1], 0])).sum())))
                        if tk == 2:
                            paths[(entry, name, lag, Y)] = pd.DataFrame(
                                {"S": w.S.values, "pos": np.r_[pos[:-1], 0], "cum": cum,
                                 "zL23_5": w.zL23_5.values, "zN_5": w.zN_5.values}, index=w.index)
    return pd.DataFrame(rows), paths


def window_stats(t, base, ys):
    a, b = t.set_index("Y").loc[ys], base.set_index("Y").loc[ys]
    m, mb = a.pnl.mean(), b.pnl.mean()
    dd, ddb = a.dd.min(), b.dd.min()
    kept, ddk = m / mb, dd / ddb
    cal, calb = m / abs(dd), mb / abs(ddb)
    # winter bootstrap: P(rule Calmar > base Calmar) == P(beats half size); CI of mean
    idx = RNG.integers(0, len(ys), size=(NB, len(ys)))
    pa, pb, da, db = a.pnl.values[idx], b.pnl.values[idx], a.dd.values[idx], b.dd.values[idx]
    ca = pa.mean(1) / np.abs(da.min(1))
    cb = pb.mean(1) / np.abs(db.min(1))
    p_beat = float(np.mean(ca > cb))
    lo, hi = np.percentile(pa.mean(1), [2.5, 97.5])
    return dict(n=len(ys), mean=m, mean_lo=lo, mean_hi=hi, worst_dd=dd, worst_winter=a.pnl.min(),
                kept=kept, dd_kept=ddk, ratio=kept / ddk if ddk > 0 else np.nan,
                kept_plus_cut=kept + 1 - ddk, calmar=cal, calmar_base=calb, p_beat_half=p_beat,
                avg_pos=a.avg_pos.mean())


def main():
    t0 = time.time()
    res, paths = score_all()
    res.to_csv(D.CACHE / "winter_results.csv", index=False)
    pd.to_pickle(paths, D.CACHE / "paths.pkl")
    S = []
    for (entry, name, lag, tk), g in res.groupby(["entry", "name", "lag", "ticks"]):
        base = res[(res.entry == entry) & (res.name == "base") & (res.ticks == tk)]
        for wn, ys in WINDOWS.items():
            s = window_stats(g, base, ys)
            s.update(entry=entry, family=g.family.iloc[0], name=name, lag=lag, ticks=tk, window=wn)
            S.append(s)
    S = pd.DataFrame(S)
    S.to_csv(D.CACHE / "summary.csv", index=False)

    # selection: per entry and family, on train, 2 ticks, lag 0 (lag 1 for D), kept >= 0.4
    sel = []
    prim = S[(S.ticks == 2) & ((S.lag == 0) | (S.family == "D_addon"))]
    for entry in ("nov", "dec"):
        tr = prim[(prim.entry == entry) & (prim.window == "train") & (prim.family != "bench") & (prim.family != "Z_control")]
        for fam, g in tr.groupby("family"):
            g = g[(g.kept >= 0.4) & (g["mean"] > 0)]
            if len(g):
                best = g.sort_values("calmar", ascending=False).iloc[0]
                sel.append(dict(entry=entry, family=fam, name=best["name"], train_calmar=best.calmar))
    sel = pd.DataFrame(sel)
    sel.to_csv(D.CACHE / "selected.csv", index=False)

    # share of cells above the scaling line by family / window (2 ticks, lag 0)
    above = (prim[prim.family != "bench"].assign(above=lambda x: x.ratio > 1)
             .groupby(["entry", "family", "window"]).above.mean().unstack())
    above.to_csv(D.CACHE / "share_above_half.csv")

    n_var = res[["entry", "name", "lag", "ticks"]].drop_duplicates().shape[0]
    n_rule = res[["entry", "name"]].drop_duplicates().shape[0]
    with open(D.CACHE / "run_log.txt", "w") as f:
        f.write(f"variants scored (entry x rule x lag x ticks): {n_var}; rule cells (entry x rule): {n_rule}\n")
        f.write(f"runtime {time.time() - t0:.0f}s\n\nSELECTED\n{sel.to_string()}\n\nSHARE ABOVE HALF\n{above.round(2).to_string()}\n")
    json.dump({"n_variants": int(n_var), "n_rule_cells": int(n_rule)}, open(D.CACHE / "counts.json", "w"))
    print(open(D.CACHE / "run_log.txt").read())


if __name__ == "__main__":
    main()
