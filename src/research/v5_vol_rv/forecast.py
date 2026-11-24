"""Step 2: ex-ante realised-vol forecasts, walk-forward.

Target y = log realised vol of contract cid over (a) the next 20 trading days (rv20f) and
(b) the option's remaining life (rv_life).  Pooled OLS across contracts, expanding window, refitted
at each month start m on rows whose target had fully resolved before m (rv20f: row date + 20
trading days < m;  rv_life: option expiry < m).  First forecast month 2011-07 (one-year burn-in).

Models (features known at the 14:30 settle of date d):
  EWMA   log ewma (lambda 0.94), unfitted  |  EWMA+b  same + walk-forward mean bias
  GARCH  log GARCH(1,1) h-step average (persistence fitted walk-forward on the front, variance
         targeted to the contract's own prior-year level), unfitted  |  GARCH+b
  IV     log atm_iv, unfitted  |  IV+b  bias-corrected  |  MZ-IV  y ~ log iv
  HAR0   y ~ log rv5 + log rv21 + log rv63
  HAR1   HAR0 + log tdays + wdel + wtrade + wdel*wtrade     (seasonality, tenor)
  HAR2   HAR1 + storage deviation (1000 bcf; 0 before 2012) + forecast level (lvl_w1+lvl_w2)/100
  HAR3   HAR1 + log atm_iv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from .vrp import prep_base

FEATS = {
    "HAR0": ["lrv5", "lrv21", "lrv63"],
    "HAR1": ["lrv5", "lrv21", "lrv63", "ltd", "wdel", "wtr", "wxw"],
    "HAR2": ["lrv5", "lrv21", "lrv63", "ltd", "wdel", "wtr", "wxw", "sd", "lvl"],
    "HAR3": ["lrv5", "lrv21", "lrv63", "ltd", "wdel", "wtr", "wxw", "liv"],
    "MZ-IV": ["liv"],
}
RAW = {"EWMA": "lew", "GARCH": None, "IV": "liv"}


def design(b):
    x = pd.DataFrame(index=b.index)
    fl = lambda v: np.log(np.maximum(v, 0.03))
    x["lrv5"], x["lrv21"], x["lrv63"] = fl(b.rv5), fl(b.rv21), fl(b.rv63)
    x["lew"], x["liv"] = fl(b.ewma), fl(b.atm_iv)
    x["lg20"], x["lglife"] = fl(b.garch20), fl(b.garch_life)
    x["ltd"] = np.log(b.tdays)
    x["wdel"] = b.dm.isin(list(D.WINTER_DEL)).astype(float)
    x["wtr"] = b.date.dt.month.isin([11, 12, 1, 2, 3]).astype(float)
    x["wxw"] = x.wdel * x.wtr
    x["sd"] = (b.sdev / 1000).fillna(0)
    x["lvl"] = ((b.lvl_w1.fillna(0) + b.lvl_w2.fillna(0)) / 100).where(x.wtr > 0, 0)
    x["y20"], x["ylife"] = np.log(b.rv20f), np.log(b.rv_life)
    return x


def walk(b, x, target, months):
    """Return OOS predictions for every model, for rows dated in each month."""
    yc = "y20" if target == "20" else "ylife"
    gcol = "lg20" if target == "20" else "lglife"
    # resolution date of each row's target
    if target == "20":
        days = pd.DatetimeIndex(sorted(b.date.unique()))
        pos = days.searchsorted(b.date)
        res = pd.Series(days[np.minimum(pos + 20, len(days) - 1)], index=b.index)
        res[pos + 20 >= len(days)] = pd.Timestamp("2100-01-01")
    else:
        res = b.oexp
    pred = pd.DataFrame(index=b.index, columns=list(FEATS) + ["EWMA", "EWMA+b", "GARCH", "GARCH+b", "IV", "IV+b", "MEAN"],
                        dtype=float)
    mstart = b.date.values.astype("datetime64[M]")
    for m in months:
        te = np.where(mstart == np.datetime64(m, "M"))[0]
        if not len(te):
            continue
        tr = (res < m).values & (b.date < m).values & x[yc].notna().values
        trd = x[tr]
        if len(trd) < 500:
            continue
        idx = b.index[te]
        pred.loc[idx, "MEAN"] = trd[yc].mean()
        for name, f in FEATS.items():
            ok = trd[f].notna().all(axis=1)
            A = np.column_stack([np.ones(ok.sum()), trd.loc[ok, f].values])
            beta, *_ = np.linalg.lstsq(A, trd.loc[ok, yc].values, rcond=None)
            pred.loc[idx, name] = np.column_stack([np.ones(len(idx)), x.loc[idx, f].values]) @ beta
        for name, col in (("EWMA", "lew"), ("GARCH", gcol), ("IV", "liv")):
            pred.loc[idx, name] = x.loc[idx, col]
            pred.loc[idx, name + "+b"] = x.loc[idx, col] + (trd[yc] - trd[col]).mean()
    return pred


def r2_table(b, x, pred, yc, ten_max=None):
    rows = []
    w = np.where(b.date.dt.year <= 2016, "choose", "check")
    for win in ["choose", "check"]:
        sel = (w == win) & x[yc].notna().values & pred["MEAN"].notna().values & pred["HAR1"].notna().values
        if ten_max:
            sel &= (b.tdays <= ten_max).values
        y = x.loc[sel, yc]
        sse0 = ((y - pred.loc[sel, "MEAN"]) ** 2).sum()
        for c in pred.columns:
            if c == "MEAN":
                continue
            p = pred.loc[sel, c]
            ok = p.notna()
            rows.append({"win": win, "model": c, "R2_oos": 1 - ((y[ok] - p[ok]) ** 2).sum() / ((y[ok] - pred.loc[sel, 'MEAN'][ok]) ** 2).sum(),
                         "rmse": np.sqrt(((y[ok] - p[ok]) ** 2).mean()), "n": int(ok.sum())})
    return pd.DataFrame(rows)


def run(force=False):
    f = D.CACHE / "s2_pred.csv"
    b = prep_base()
    x = design(b)
    if f.exists() and not force:
        pr = pd.read_csv(f, index_col=0)
        p20 = pr.filter(like="p20_").rename(columns=lambda c: c[4:])
        pl = pr.filter(like="pl_").rename(columns=lambda c: c[3:])
    else:
        months = pd.date_range("2011-07-01", "2021-12-01", freq="MS")
        p20 = walk(b, x, "20", months)
        pl = walk(b, x, "life", months)
        pd.concat([p20.add_prefix("p20_"), pl.add_prefix("pl_")], axis=1).to_csv(f)
    return b, x, p20, pl


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    b, x, p20, pl = run(force=True)
    for name, pred, yc in (("next-20d RV", p20, "y20"), ("life RV", pl, "ylife")):
        t = r2_table(b, x, pred, yc)
        print("==", name, "(all tenors)")
        print(t.pivot(index="model", columns="win", values="R2_oos").round(3))
        t.to_csv(D.CACHE / f"s2_r2_{yc}.csv", index=False)
        t = r2_table(b, x, pred, yc, ten_max=130)
        print("==", name, "(tenor <= 130 tdays: trade universe)")
        print(t.pivot(index="model", columns="win", values="R2_oos").round(3))
        t.to_csv(D.CACHE / f"s2_r2_{yc}_short.csv", index=False)
