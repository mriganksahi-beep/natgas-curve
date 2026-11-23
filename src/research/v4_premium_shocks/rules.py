"""Steps 3-4: costed, walk-forward fade rules on the second-month outright and on H-J.

Signal at the d settle (Nov-Feb days with a walk-forward news split):
    x_k  = trailing k-day cumulative part of the move (k = 1, 3, 5):
           PREM  : daily-split premium   sum prem_{d-k+1..d}
           WPREM : rolling-weekly-split premium  (k-day move - weekly betas x k-day news terms)
           RAW   : raw cumulative move (the naive benchmark)
           NEWS  : daily-split news part (FOLLOW rule: position = +sign; z uses the trailing sd of the news part)
    z    = x_k / (sd60 * sqrt(k)),  sd60 = trailing 60-trading-day sd of the daily move through d-1
    if |z| >= thr: position = -sign(x_k) (fade; +sign for NEWS), 1 contract (1 spread), entered at the d settle,
    exited at the d+h settle on the SAME contract(s). One position at a time (no pyramiding).
    *_UP variants: fade only up-shocks (x > 0) -- post-hoc variant motivated by step 1, labelled as such.
Costs: k ticks/side/leg ($10/tick): outright round trip 2 x 2 x $10 = $40 at 2 ticks, $80 at 4;
H-J spread round trip $80 at 2 ticks, $160 at 4.
Grid: k in {1,3,5} x thr in {1.0,1.5,2.0} x h in {2,5,10,20} = 36 per signal family (h=20 added after step 1).
Selection on the choose window only (winters 2012-13..2016-17): max t-stat of mean net $ at 2 ticks, >= 30 trades
(hj: >= 20). The chosen (k, thr, h) is then applied unchanged to the check window AND to RAW (key benchmark).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .common import CACHE, CHOOSE, CHECK, MULT, TICK, win

KS, THRS, HOLDS = (1, 3, 5), (1.0, 1.5, 2.0), (2, 5, 10, 20)   # h=20 added after step 1 (see report)
FAMILIES = ("PREM", "WPREM", "RAW", "NEWS", "PREM_UP", "WPREM_UP", "RAW_UP")


def prep(p: pd.DataFrame, name: str) -> pd.DataFrame:
    p = p.copy()
    p["sd60"] = p[f"move_{name}"].rolling(60, min_periods=40).std().shift(1)
    nw = p[f"news_{name}"]
    p["sd60n"] = nw.dropna().rolling(60, min_periods=40).std().shift(1).reindex(p.index).ffill()
    p["elig"] = (p[f"core_{name}"] == True) & p[f"news_{name}"].notna()  # noqa: E712
    for h in HOLDS:
        if name == "hj":
            p[f"pnl{h}"] = p[f"fut{h}_hj"] * MULT                       # long 1 spread, $
        else:
            p[f"pnl{h}"] = p[f"px0_{name}"] * (np.exp(p[f"fut{h}_{name}"] / 100) - 1) * MULT
    return p


def signal(p, name, fam, k):
    base = fam.replace("_UP", "")
    col = {"PREM": f"cprem{k}_{name}", "WPREM": f"wprem{k}_{name}", "RAW": f"cmove{k}_{name}",
           "NEWS": f"cnews{k}_{name}"}[base]
    x = p[col]
    z = x / ((p.sd60n if base == "NEWS" else p.sd60) * np.sqrt(k))
    pos = np.sign(x) if base == "NEWS" else -np.sign(x)
    if fam.endswith("_UP"):
        pos = pos.where(x > 0, 0.0)
    return z, pos


def simulate(p, name, fam, k, thr, h):
    z, pos = signal(p, name, fam, k)
    ok = (p.elig & (z.abs() >= thr) & (pos != 0) & p[f"pnl{h}"].notna()).to_numpy()
    pnl = p[f"pnl{h}"].to_numpy(); ps = pos.to_numpy()
    trades, nxt = [], -1
    for i in np.flatnonzero(ok):
        if i < nxt:
            continue
        trades.append((i, ps[i], ps[i] * pnl[i]))
        nxt = i + h
    if not trades:
        return pd.DataFrame(columns=["date", "W", "pos", "gross"])
    idx, pp, g = zip(*trades)
    return pd.DataFrame({"date": p.date.to_numpy()[list(idx)], "W": p.W.to_numpy()[list(idx)],
                         "pos": pp, "gross": g})


def signal_days(p, name, fam, k, thr, h):
    """Every qualifying day (overlapping positions): position x h-day P&L, net of 2-tick costs."""
    z, pos = signal(p, name, fam, k)
    legs = 2 if name == "hj" else 1
    m = p.elig & (z.abs() >= thr) & (pos != 0) & p[f"pnl{h}"].notna()
    d = pd.DataFrame({"i": np.flatnonzero(m.to_numpy()), "W": p.W[m].to_numpy(),
                      "net": (pos * p[f"pnl{h}"])[m].to_numpy() - 4 * TICK * legs})
    return d


def hac_mean(d, h):
    """Mean of per-day nets with Newey-West SE on the daily calendar (lags = h)."""
    from .common import nw_ols
    if len(d) < 10:
        return np.nan, np.nan
    b, se = nw_ols(d.net.to_numpy(), np.zeros((len(d), 0)), h)
    return b[0], b[0] / se[0]


def paired_diff(t1, t2, name, B, rng):
    """Winter-bootstrap CI of mean net/trade (2 ticks) of rule 1 minus rule 2 (same resampled winters)."""
    legs = 2 if name == "hj" else 1
    a = t1.assign(net=t1.gross - 4 * TICK * legs).groupby("W").net.agg(["sum", "count"])
    b = t2.assign(net=t2.gross - 4 * TICK * legs).groupby("W").net.agg(["sum", "count"])
    ws = sorted(set(a.index) | set(b.index))
    a, b = a.reindex(ws).fillna(0), b.reindex(ws).fillna(0)
    idx = rng.integers(0, len(ws), (B, len(ws)))
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = a["sum"].to_numpy()[idx].sum(1) / a["count"].to_numpy()[idx].sum(1) - \
             b["sum"].to_numpy()[idx].sum(1) / b["count"].to_numpy()[idx].sum(1)
    obs = a["sum"].sum() / a["count"].sum() - b["sum"].sum() / b["count"].sum()
    return obs, *np.nanpercentile(bs, [2.5, 97.5])


def stats(tr: pd.DataFrame, name: str, B: int = 0, rng=None) -> dict:
    legs = 2 if name == "hj" else 1
    out = {"n": len(tr)}
    if len(tr) < 3:
        return out
    yrs = tr.W.nunique()
    for tk in (2, 4):
        net = tr.gross - 2 * tk * TICK * legs
        out[f"mean{tk}"] = net.mean()
        out[f"t{tk}"] = net.mean() / (net.std(ddof=1) / np.sqrt(len(net)))
        out[f"sharpe{tk}"] = net.mean() / net.std(ddof=1) * np.sqrt(len(net) / max(yrs, 1))
    net = tr.gross - 4 * TICK * legs
    out["hit2"] = (net > 0).mean()
    out["worst2"] = net.min()
    out["best2"] = net.max()
    out["n_winters"] = yrs
    if B and len(tr) >= 5:
        ws = tr.W.unique(); s = tr.assign(net=net).groupby("W").net.agg(["sum", "count"])
        a, c = s["sum"].to_numpy(), s["count"].to_numpy()
        idx = rng.integers(0, len(ws), (B, len(ws)))
        bs = a[idx].sum(1) / c[idx].sum(1)
        out["lo2"], out["hi2"] = np.percentile(bs, [2.5, 97.5])
    return out


def grid(p, name):
    rows = []
    for fam, k, thr, h in itertools.product(FAMILIES, KS, THRS, HOLDS):
        tr = simulate(p, name, fam, k, thr, h)
        for wn, w in (("choose", CHOOSE), ("check", CHECK)):
            rows.append({"ins": name, "fam": fam, "k": k, "thr": thr, "h": h, "window": wn,
                         **stats(win(tr, w), name)})
    return pd.DataFrame(rows)


def pick(g, fam, min_n):
    c = g[(g.fam == fam) & (g.window == "choose") & (g.n >= min_n)]
    if c.empty:
        c = g[(g.fam == fam) & (g.window == "choose") & (g.n >= 10)]
    return c.sort_values("t2", ascending=False).iloc[0][["k", "thr", "h"]].to_dict()


def main(B=2000):
    rng = np.random.default_rng(5)
    daily = pd.read_pickle(CACHE / "daily.pkl")
    allg, sel, by_w, trades_all = [], [], [], []
    for name, min_n in (("second", 30), ("hj", 20)):
        p = prep(daily, name)
        g = grid(p, name)
        allg.append(g)
        for fam in ("PREM", "WPREM", "PREM_UP", "WPREM_UP", "RAW", "RAW_UP", "NEWS"):
            prm = pick(g, fam, min_n)
            k, thr, h = int(prm["k"]), float(prm["thr"]), int(prm["h"])
            comps = [(fam, fam)]
            trs = {}
            if fam in ("PREM", "WPREM"):
                comps.append((fam, "RAW"))          # naive benchmark at the SAME parameters
            if fam in ("PREM_UP", "WPREM_UP"):
                comps.append((fam, "RAW_UP"))
            for chosen_for, run_fam in comps:
                tr = simulate(p, name, run_fam, k, thr, h)
                tr["ins"], tr["rule"] = name, f"{run_fam}@{chosen_for}"
                trades_all.append(tr)
                trs[run_fam] = tr
                sd_ = signal_days(p, name, run_fam, k, thr, h)
                for wn, w in (("choose", CHOOSE), ("check", CHECK), ("all", (CHOOSE[0], CHECK[1]))):
                    dd = sd_[(sd_.W >= w[0]) & (sd_.W <= w[1])]
                    mo, to = hac_mean(dd, h)
                    extra = {"n_sigdays": len(dd), "sig_mean2": mo, "sig_t_hac": to}
                    if run_fam != chosen_for:
                        o, lo, hi = paired_diff(win(trs[chosen_for], w), win(tr, w), name, B, rng)
                        extra.update({"split_minus_raw": o, "diff_lo": lo, "diff_hi": hi})
                    sel.append({"ins": name, "chosen_for": chosen_for, "run": run_fam, "k": k, "thr": thr, "h": h,
                                "window": wn, **stats(win(tr, w), name, B, rng), **extra})
                legs = 2 if name == "hj" else 1
                tr = tr.assign(net2=tr.gross - 4 * TICK * legs)
                for W, d in tr.groupby("W"):
                    by_w.append({"ins": name, "rule": f"{run_fam}@{chosen_for}", "W": W, "n": len(d),
                                 "net2_sum": d.net2.sum(), "net2_mean": d.net2.mean(), "worst": d.net2.min()})
    pd.concat(allg).to_csv(CACHE / "step3_grid.csv", index=False, float_format="%.4g")
    pd.DataFrame(sel).to_csv(CACHE / "step3_selected.csv", index=False, float_format="%.4g")
    pd.DataFrame(by_w).to_csv(CACHE / "step3_by_winter.csv", index=False, float_format="%.4g")
    pd.concat(trades_all).to_csv(CACHE / "step3_trades.csv", index=False, float_format="%.6g")
    print("done step3")


if __name__ == "__main__":
    main()
