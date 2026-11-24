"""Q3/Q4 scoring: costed rule P&L, selection tests, IV-RV gap by disagreement, storage x disagreement cells."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from .trade import BENCH, RULES, Market, signals, simulate, weekly_outcomes

RNG = np.random.default_rng(20261124)
B = 2000
COSTS = {"gross": (0.0, 0.0), "1%": (0.01, 0.002), "1.5%": (0.015, 0.002), "2%": (0.02, 0.002)}


def boot_gy(x, gy, B=B):
    """Mean per gas year (total over the gas year, averaged across gas years) and bootstrap CI + p(>0)."""
    s = pd.Series(np.asarray(x, float)).groupby(np.asarray(gy)).sum()
    v = s.values
    k = len(v)
    bs = v[RNG.integers(0, k, size=(B, k))].mean(1)
    return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (bs <= 0).mean(), k, (v > 0).sum()


def max_dd(x):
    c = np.cumsum(x)
    return float((c - np.maximum.accumulate(np.maximum(c, 0))).min())


def run_rules(mk, S):
    out, daily = [], {}
    for name, fn in RULES.items():
        pos = pd.Series(fn(S), index=S.index)
        for cl, (c, fut) in COSTS.items():
            if c == 0:
                df, tr = simulate(mk, S, pos, c=0.0, fut=0.0)
                df["net"] = df.gross
            else:
                df, tr = simulate(mk, S, pos, c=c, fut=fut)
            daily[(name, cl)] = df
            df["gy"] = D.gas_year(df.date)
            df["win"] = D.window(df.date)
            for win in ("choose", "check", "all"):
                g = df if win == "all" else df[df.win == win]
                m, lo, hi, p, k, kpos = boot_gy(g.net, g.gy)
                wk = g.set_index("date").net.resample("W").sum()
                wk = wk[wk != 0]
                out.append({"rule": name, "cost": cl, "win": win, "per_winter": m, "lo": lo, "hi": hi, "p_le0": p,
                            "winters": k, "pos_winters": kpos, "maxdd": max_dd(g.net.values),
                            "active_frac": (g.pos != 0).mean(), "n_open": len(tr[D.window(tr.date) == win]) if win != "all" else len(tr),
                            "sharpe_wk": wk.mean() / wk.std() * np.sqrt(30) if len(wk) > 5 else np.nan})
    T = pd.DataFrame(out)
    # paired comparison vs always_long (same cost level): per-winter difference
    rows = []
    for (name, cl), df in daily.items():
        if name in BENCH:
            continue
        al = daily[("always_long", cl)].set_index("date").net
        d = df.set_index("date").net.reindex(al.index.union(df.date)).fillna(0) - al.reindex(al.index.union(df.date)).fillna(0)
        gy = D.gas_year(d.index)
        win = D.window(d.index)
        for w in ("choose", "check"):
            m, lo, hi, p, k, kpos = boot_gy(d[win == w].values, gy[win == w])
            rows.append({"rule": name, "cost": cl, "win": w, "diff_vs_long": m, "lo": lo, "hi": hi, "p_le0": p})
    Dv = pd.DataFrame(rows)
    # Holm on the choose window, 1.5% cost, non-benchmark rules, H1: per-winter net > 0
    for tab, col in ((T, "p_le0"), (Dv, "p_le0")):
        msk = (tab.win == "choose") & (tab.cost == "1.5%") & ~tab.rule.isin(BENCH)
        pv = tab.loc[msk, col].sort_values()
        m = len(pv)
        tab.loc[pv.index, "p_holm"] = np.minimum(1, np.maximum.accumulate(pv.values * (m - np.arange(m))))
    return T, Dv, daily


def selection(W, S):
    """Weekly fresh-straddle gross P&L, on vs off for each disagreement-type signal (cost-free selection test)."""
    W = W.set_index("date")
    rows = []
    for name in RULES:
        if name in ("always_long", "always_short", "storage_sign"):
            continue
        on = pd.Series(RULES[name](S), index=S.index).reindex(W.index).fillna(0) > 0
        for win in ("choose", "check", "all"):
            msk = np.ones(len(W), bool) if win == "all" else (D.window(W.index) == win)
            g = W[msk]
            o = on[msk]
            gy = D.gas_year(g.index)
            # difference in means with gas-year bootstrap
            dfx = pd.DataFrame({"x": g.gross.values, "on": o.values, "gy": gy})
            ug = np.unique(gy)
            est = dfx[dfx.on].x.mean() - dfx[~dfx.on].x.mean()
            bs = []
            grp = {k: v for k, v in dfx.groupby("gy")}
            for _ in range(B):
                pick = pd.concat([grp[k] for k in RNG.choice(ug, len(ug))])
                bs.append(pick[pick.on].x.mean() - pick[~pick.on].x.mean())
            rows.append({"rule": name, "win": win, "weeks_on": int(o.sum()), "weeks_off": int((~o).sum()),
                         "gy_on": dfx[dfx.on].gy.nunique(), "on_mean": dfx[dfx.on].x.mean(),
                         "off_mean": dfx[~dfx.on].x.mean(), "diff": est,
                         "lo": np.nanpercentile(bs, 2.5), "hi": np.nanpercentile(bs, 97.5)})
    return pd.DataFrame(rows)


def gap_by_unc():
    """IV - forward RV (20d, h=20 leg 1) by walk-forward disagreement tercile; and log gap vs HAR forecast."""
    b = D.base()
    b = b[(b.h == 20) & (b.leg == 1) & (b.date >= "2011-10-01")].set_index("date")
    S = signals()
    P = pd.read_csv(D.CACHE / "q1_preds.csv", parse_dates=["date"])
    P = P[(P.h == 20) & (P.leg == 1)].set_index("date")
    x = S.unc_n_w23.reindex(b.index)
    q33 = S.unc_n_w23.expanding(100).quantile(1 / 3).shift(1).reindex(b.index)
    q67 = S.unc_n_w23.expanding(100).quantile(2 / 3).shift(1).reindex(b.index)
    terc = np.where(x > q67, "high", np.where(x < q33, "low", "mid"))
    b["terc"] = terc
    b["gap"] = (b.atm_iv - b.rvf) * 100
    b["fgap_HAR"] = (np.log(b.atm_iv) - P.p_HAR.reindex(b.index)) * 100     # ex-ante cheapness (log pts)
    b["tight"] = np.where(b.sdev < -150, "tight", np.where(b.sdev > 150, "loose", "normal"))
    b.loc[b.sdev.isna(), "tight"] = "n/a"
    rows = []
    for (win, t), g in b.groupby([b.win, "terc"]):
        m, lo, hi, *_ = _cellmean(g.gap, g.gy)
        rows.append({"win": win, "terc": t, "n": len(g), "gy": g.gy.nunique(), "iv": g.atm_iv.mean() * 100,
                     "rv20": g.rvf.mean() * 100, "gap": m, "lo": lo, "hi": hi, "fgap_HAR": g.fgap_HAR.mean()})
    G = pd.DataFrame(rows)
    rows = []
    b["U"] = np.where(x > S.unc_n_w23_p50.reindex(b.index), "U_high", "U_low")
    for (st, u), g in b[b.tight != "n/a"].groupby(["tight", "U"]):
        m, lo, hi, *_ = _cellmean(g.gap, g.gy)
        yrs = g.groupby("gy").size()
        rows.append({"storage": st, "unc": u, "days": len(g), "gas_years": len(yrs),
                     "years": ",".join(str(int(y)) for y in yrs.index), "gap": m, "lo": lo, "hi": hi})
    C = pd.DataFrame(rows)
    return G, C, b


def _cellmean(x, gy):
    df = pd.DataFrame({"x": x.values, "g": np.asarray(gy)}).dropna()
    agg = df.groupby("g").x.agg(["sum", "count"])
    s, c = agg["sum"].values, agg["count"].values
    k = len(s)
    idx = RNG.integers(0, k, size=(B, k))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), k


def q4_cells(W, S):
    """Weekly fresh-straddle gross P&L in storage x disagreement cells, with per-gas-year contributions."""
    W = W.set_index("date").copy()
    sd = S.sdev.reindex(W.index)
    u = (S.unc_n_w23 > S.unc_n_w23_p50).reindex(W.index)
    W["storage"] = np.where(sd < -150, "tight", np.where(sd > 150, "loose", "normal"))
    W.loc[sd.isna(), "storage"] = "n/a"
    W["unc"] = np.where(u, "U_high", "U_low")
    W["gy"] = D.gas_year(W.index)
    rows, contrib = [], []
    for (st, uu), g in W[W.storage != "n/a"].groupby(["storage", "unc"]):
        m, lo, hi, k = _cellmean(g.gross, g.gy)
        by = g.groupby("gy").gross.agg(["sum", "count"])
        top = g.gross.sort_values(ascending=False)
        share_top3 = top.iloc[:3].sum() / g.gross.sum() if g.gross.sum() != 0 else np.nan
        rows.append({"storage": st, "unc": uu, "weeks": len(g), "gas_years": k, "mean": m, "lo": lo, "hi": hi,
                     "median": g.gross.median(), "top3_weeks": ",".join(d.strftime("%Y-%m-%d") for d in top.index[:3]),
                     "top3_share_of_total": share_top3})
        for gyy, r in by.iterrows():
            contrib.append({"storage": st, "unc": uu, "gy": gyy, "sum": r["sum"], "weeks": r["count"]})
    return pd.DataFrame(rows), pd.DataFrame(contrib)


def main():
    mk = Market()
    S = signals()
    S.to_csv(D.CACHE / "q3_signals.csv")
    T, Dv, daily = run_rules(mk, S)
    T.to_csv(D.CACHE / "q3_rules.csv", index=False)
    Dv.to_csv(D.CACHE / "q3_vs_long.csv", index=False)
    pd.to_pickle(daily, D.CACHE / "q3_daily.pkl")
    W = weekly_outcomes(mk, S)
    W.to_csv(D.CACHE / "q3_weekly.csv", index=False)
    Sel = selection(W, S)
    Sel.to_csv(D.CACHE / "q3_selection.csv", index=False)
    G, C, _ = gap_by_unc()
    G.to_csv(D.CACHE / "q3_gap_terc.csv", index=False)
    C.to_csv(D.CACHE / "q4_gap_cells.csv", index=False)
    Q4, contrib = q4_cells(W, S)
    Q4.to_csv(D.CACHE / "q4_cells.csv", index=False)
    contrib.to_csv(D.CACHE / "q4_contrib.csv", index=False)
    return T, Dv, Sel, G, C, Q4, contrib


def random_timing(mk, S, p_on=(1 / 3, 1 / 2), n=100, c=0.015):
    """Exposure-matched benchmark: iid weekly long/flat with P(on)=p, same engine and costs."""
    from .trade import decision_days
    dd = decision_days(mk, S)
    rng = np.random.default_rng(99)
    rows = []
    for p in p_on:
        for k in range(n):
            pos = pd.Series((rng.random(len(dd)) < p).astype(int), index=dd)
            df, tr = simulate(mk, S, pos, c=c)
            df["gy"] = D.gas_year(df.date)
            df["win"] = D.window(df.date)
            for win in ("choose", "check"):
                g = df[df.win == win]
                rows.append({"p_on": p, "draw": k, "win": win,
                             "gross_pw": g.groupby("gy").gross.sum().mean(),
                             "net_pw": g.groupby("gy").net.sum().mean(), "n_open": int((D.window(tr.date) == win).sum())})
    R = pd.DataFrame(rows)
    R.to_csv(D.CACHE / "q3_random.csv", index=False)
    return R


def gap_diff(b):
    """High - low disagreement tercile difference in IV - RV20 (vol pts), gas-year bootstrap."""
    rows = []
    for win in ("choose", "check", "all"):
        g = b if win == "all" else b[b.win == win]
        g = g[g.terc.isin(["high", "low"])]
        grp = {k: v for k, v in g.groupby("gy")}
        ug = list(grp)
        est = g[g.terc == "high"].gap.mean() - g[g.terc == "low"].gap.mean()
        bs = []
        for _ in range(B):
            q = pd.concat([grp[k] for k in RNG.choice(ug, len(ug))])
            bs.append(q[q.terc == "high"].gap.mean() - q[q.terc == "low"].gap.mean())
        rows.append({"win": win, "high_minus_low": est, "lo": np.nanpercentile(bs, 2.5), "hi": np.nanpercentile(bs, 97.5)})
    return pd.DataFrame(rows)


def extras():
    mk = Market()
    S = pd.read_csv(D.CACHE / "q3_signals.csv", index_col=0, parse_dates=True)
    R = random_timing(mk, S)
    G, C, b = gap_by_unc()
    GD = gap_diff(b)
    GD.to_csv(D.CACHE / "q3_gap_diff.csv", index=False)
    return R, GD


if __name__ == "__main__":
    import sys
    pd.set_option("display.width", 250)
    if "--extras" not in sys.argv:
        main()
    R, GD = extras()
    print(R.groupby(["p_on", "win"])[["gross_pw", "net_pw", "n_open"]].describe(percentiles=[.05, .5, .95]).round(2).T)
    print(GD.round(2))
