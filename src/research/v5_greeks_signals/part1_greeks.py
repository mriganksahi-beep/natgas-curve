"""Part 1 (+ robustness for the surface-level shocks): what does the vega shock predict?

Descriptive regressions (HAC; choose 2010-16 / check 2017-21 separately), each with a futures-only
control set so the coefficient measures information the futures path does not already carry:
  a) next-day front / second return (robustness: + |r|, r^2 controls, 1% winsorised, year-by-year sign)
  b) next-day forecast revision news_tot(t+1) (Oct-Apr rows), controls r_front(t), r_second(t)
  c) next storage surprise (storage_surprise_5y, 2015+): signal summed over the days between reports,
     control = front return over the same days
  d) realised vol: rv5 (sum of next 5 squared front returns) on z + atm1^2 + past rv5
  e) next-day / 5-day straddle P&L (vol momentum vs mean reversion)
Writes data/processed/research_v5_greeks_signals/part1_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.v5_greeks_signals import common as C
from src.research.v5_greeks_signals.features import build

SIGS = ["vshock1", "vshock123", "vshock123_5", "vega_pnl123", "resid_rel123", "d_level", "d_term",
        "pc1", "d_skew1", "d_skew1_res", "d_skew2"]


def win(x, w):
    return x[(x.index >= w[0]) & (x.index <= w[1])]


def reg(y, X, lags, w):
    b, t, n = C.hac_ols(win(y, w), win(X, w), lags)
    return b[1], t[1], n


def main():
    f = build()
    f["r_front_5"] = f.r_front.rolling(5).sum()
    f["rv_past5"] = (f.r_front ** 2).rolling(5).sum()
    z = {s: C.wf_z(f[s]) for s in SIGS + ["r_front", "r_front_5"]}
    rows = []
    W = (("choose", C.CHOOSE), ("check", C.CHECK), ("all", (C.CHOOSE[0], C.CHECK[1])))

    def add(test, sig, y, X, lags):
        rec = {"test": test, "signal": sig}
        for wn, w in W:
            b, t, n = reg(y, X, lags, w)
            rec.update({f"{wn}_b": b, f"{wn}_t": t, f"{wn}_n": n})
        rows.append(rec)

    for s in SIGS:
        base = pd.DataFrame({"z": z[s], "r1": z["r_front"], "r5": z["r_front_5"]})
        for tgt in ("r_front", "r_second"):
            y = f[tgt].shift(-1)
            add(f"{tgt}+1", s, y, base, 5)
            X = base.assign(absr=f.r_front.abs(), r2=f.r_front ** 2)
            add(f"{tgt}+1|abs,sq", s, y, X, 5)
            zw = base.assign(z=base.z.clip(-2.33, 2.33))
            yw = y.clip(y.quantile(0.01), y.quantile(0.99))
            add(f"{tgt}+1 winsor", s, yw, zw, 5)
        # news next day (Oct-Apr only rows have news)
        Xn = pd.DataFrame({"z": z[s], "r1": f.r_front, "r2": f.r_second})
        add("news_tot+1", s, f.news_tot.shift(-1), Xn, 5)
        add("news_w3+1", s, f.news_w3.shift(-1), Xn, 5)
        # contemporaneous: does the signal react to today's news beyond the futures move?
        add("signal~news_tot(t) [rev]", s, z[s], pd.DataFrame({"n": f.news_tot, "r1": f.r_front}), 5)
        # realised vol and straddle P&L
        Xv = pd.DataFrame({"z": z[s], "iv2": f.atm1 ** 2, "rvp": f.rv_past5})
        add("rv5", s, f.rv5, Xv, 5)
        add("straddle+1", s, f.st_pnl, Xv, 5)
        add("straddle 5d", s, f.st_pnl[::-1].rolling(5).sum()[::-1], Xv, 5)
        add("hj+1", s, f.d_hj.shift(-1), base, 5)
    res = pd.DataFrame(rows)
    res.to_csv(C.CACHE / "part1_regs.csv", index=False)

    # storage surprise: signal summed between reports (report day excluded) -> surprise on report day
    rep = f.index[f.storage_surprise_5y.notna()]
    srows = []
    for s in ["vshock123", "vega_pnl123", "d_level", "d_skew1", "pc1", "resid_rel123"]:
        recs = []
        for i in range(1, len(rep)):
            a, b = rep[i - 1], rep[i]
            win_ = f[(f.index > a) & (f.index < b)]
            recs.append({"date": b, "sig": z[s].reindex(win_.index).sum(), "ret": win_.r_front.sum(),
                         "surp": f.storage_surprise_5y[b]})
        d = pd.DataFrame(recs).set_index("date")
        rec = {"signal": s}
        for wn, w in (("choose(2015-16)", C.CHOOSE), ("check", C.CHECK)):
            bb, tt, n = C.hac_ols(win(d.surp, w), win(d[["sig", "ret"]], w), 2)
            rec.update({f"{wn}_b": bb[1], f"{wn}_t": tt[1], f"{wn}_n": n,
                        f"{wn}_ret_t": tt[2]})
        srows.append(rec)
    sres = pd.DataFrame(srows)
    sres.to_csv(C.CACHE / "part1_storage.csv", index=False)

    # year-by-year sign of the next-day front slope (vshock123, pc1, d_level, resid_rel123)
    yrows = []
    for s in ["vshock123", "vega_pnl123", "pc1", "d_level", "resid_rel123"]:
        X = pd.DataFrame({"z": z[s], "r1": z["r_front"], "r5": z["r_front_5"]})
        for y_ in range(2010, 2022):
            w = (pd.Timestamp(y_, 1, 1), pd.Timestamp(y_, 12, 31))
            b, t, n = reg(f.r_front.shift(-1), X, 5, w)
            yrows.append({"signal": s, "year": y_, "b": b, "t": t})
    yres = pd.DataFrame(yrows).pivot(index="year", columns="signal", values="t")
    yres.to_csv(C.CACHE / "part1_years.csv")
    return res, sres, yres


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    r, s, y = main()
    print(r.round(3).to_string())
    print(s.round(3).to_string())
    print(y.round(2).to_string())
