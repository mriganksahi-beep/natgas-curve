"""Step 3: cross-sectional delta-hedged straddle trade.

Positions.  At each rebalance date r0 (default: the first trading day after each monthly option
expiry, so ~1 per month) every contract with 10 <= tdays <= 130 on the surface is a candidate.
For each candidate we open a LONG ATM straddle at K = F(r0) (price from atm_iv; strike-grid offset
ignored), delta-hedge it daily at the settle with futures, and hold it to the next rebalance r1
(closing at the r1 settle) or to option expiry if earlier (payoff |F - K|, no exit cost).
Marks between: Black-76 at the day's surface smile interpolated linearly in k = ln(K/F)
over {-0.2,-0.1,0,+0.1,+0.2} (flat outside), last iv carried if the row is missing.
Costs (per straddle, $/MMBtu): option half-spread c * premium (base c = 1.5%, min $0.002 per option
leg = $0.004 per straddle) at entry and at exit if not expired; futures 2 ticks ($0.002) per unit of
|change in hedge| at every rebalance incl. the initial hedge and the final unwind.
Sizing: vega-neutral, each side scaled to 1.00 vol of vega at entry (1/n per name / vega0), so P&L is
in vol units per unit vega per side (x100 = vol points).

Options data come only from data/processed/v5/surface.csv (cut at 2021-12-31): the last period is
truncated and marked at 2021-12-31.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

from . import data as D
from .forecast import run as fc_run
from .stats import sharpe_ci

KG = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
KC = ["iv_m20", "iv_m10", "atm_iv", "iv_p10", "iv_p20"]
TMIN, TMAX = 10, 130
START = pd.Timestamp("2011-07-01")


def black_straddle(F, K, T, s):
    v = s * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    c = F * ndtr(d1) - K * ndtr(d1 - v)
    p = c - (F - K)
    from scipy.stats import norm
    return c + p, 2 * ndtr(d1) - 1, 2 * F * norm.pdf(d1) * np.sqrt(T)


def smile_iv(row, k):
    v = row[KC].values.astype(float)
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return np.nan
    return float(np.interp(k, KG[ok], v[ok]))


def build_positions(rebal, force=False, tag="m"):
    f = D.CACHE / f"s3_positions_{tag}.pkl"
    if f.exists() and not force:
        return pd.read_pickle(f)
    s = D.surface()
    px, days, oexp = D.futures()
    days = days[days <= D.CUT]
    sx = s.set_index(["date", "cid"]).sort_index()
    pos = []
    for i, r0 in enumerate(rebal):
        r1 = rebal[i + 1] if i + 1 < len(rebal) else D.CUT
        if r0 not in sx.index.get_level_values(0):
            continue
        cand = sx.loc[r0]
        cand = cand[(cand.tdays >= TMIN) & (cand.tdays <= TMAX) & cand.atm_iv.notna()]
        for cid, row in cand.iterrows():
            e = oexp[cid]
            end = min(r1, e)
            path = days[(days >= r0) & (days <= end)]
            K = row.F
            F = px.loc[path, cid].values
            V, De = np.empty(len(path)), np.empty(len(path))
            iv = row.atm_iv
            for j, t in enumerate(path):
                ntd = days.searchsorted(e) - days.searchsorted(t)
                if t == e or ntd <= 0:
                    V[j], De[j] = abs(F[j] - K), np.sign(F[j] - K)
                    continue
                if (t, cid) in sx.index:
                    rr = sx.loc[(t, cid)]
                    v = smile_iv(rr, np.log(K / F[j]))
                    iv = v if np.isfinite(v) else iv
                V[j], De[j], _ = black_straddle(F[j], K, ntd / 252.0, iv)
            expired = path[-1] == e
            dF = np.diff(F)
            daily = np.diff(V) - De[:-1] * dF                     # long straddle, hedged at t
            hedge_trades = np.abs(np.diff(np.concatenate([[0.0], De[:-1], [0.0]])))
            pos.append({"r0": r0, "r1": end, "cid": cid, "tdays": int(row.tdays), "F0": K, "iv0": row.atm_iv,
                        "P0": V[0], "vega0": row.vega, "Pend": V[-1], "expired": bool(expired),
                        "gross": float(daily.sum()), "hedge_units": float(np.sum(hedge_trades)),
                        "dates": path[1:], "daily": daily})
    out = pd.DataFrame(pos)
    out.to_pickle(f)
    return out


def costs(p, c=0.015, fut=0.002):
    if c == 0 and fut == 0:                      # gross: no costs at all (no minimum tick either)
        return pd.Series(0.0, index=p.index)
    opt = np.maximum(c * p.P0, 0.004) + np.where(p.expired, 0.0, np.maximum(c * p.Pend, 0.004))
    return opt + fut * p.hedge_units


def monthly_rebal():
    s = D.surface()
    px, days, oexp = D.futures()
    days = days[days <= D.CUT]
    r = sorted({days[days.searchsorted(e) + 1] for e in oexp if START <= e < D.CUT and days.searchsorted(e) + 1 < len(days)})
    return [d for d in r if d >= START]


def weekly_rebal():
    px, days, oexp = D.futures()
    days = days[(days >= START) & (days <= D.CUT)]
    return list(days[::5])


# ----------------------------------------------------------------------------- strategies
def add_signals(p):
    b, x, p20, pl = fc_run()
    key = b[["date", "cid"]].copy()
    sig = pd.DataFrame({"date": b.date, "cid": b.cid,
                        "S_life": x.liv - pl["HAR1"], "S_20": x.liv - p20["HAR1"], "S_ewma": x.liv - x.lew,
                        "sdev": b.sdev})
    return p.merge(sig, left_on=["r0", "cid"], right_on=["date", "cid"], how="left").drop(columns="date")


def weights(g, rule, n=2, rng=None):
    """Return weights (per unit vega per side) for positions in one rebalance group."""
    w = pd.Series(0.0, index=g.index)
    v = g.vega0
    if rule == "short_all":
        w[:] = -1.0 / len(g) / v
        return w
    if rule == "long_all":
        w[:] = 1.0 / len(g) / v
        return w
    if rule == "storage_sign":          # post-hoc (informed by step-1 map): long vol if storage below 5y
        sd = g.sdev.iloc[0]
        if not np.isfinite(sd):
            return w
        w[:] = (1.0 if sd < 0 else -1.0) / len(g) / v
        return w
    if len(g) < 2 * n:
        n = len(g) // 2
    if n == 0:
        return w
    if rule == "term":                   # short front n, long back n
        o = g.tdays.sort_values().index
    elif rule == "random":
        o = pd.Index(rng.permutation(g.index.values))
    else:                                # signal: richest first
        gg = g.dropna(subset=[rule])
        if len(gg) < 2 * n:
            n = len(gg) // 2
            if n == 0:
                return w
        o = gg[rule].sort_values(ascending=False).index
    w[o[:n]] = -1.0 / n / v[o[:n]]
    w[o[-n:]] = 1.0 / n / v[o[-n:]]
    return w


def evaluate(p, rule, c=0.015, fut=0.002, rng=None, n=2):
    p = p.copy()
    p["cost"] = costs(p, c, fut)
    p["w"] = 0.0
    for r0, g in p.groupby("r0"):
        p.loc[g.index, "w"] = weights(g, rule, n=n, rng=rng)
    p["net"] = p.w * p.gross - p.w.abs() * p.cost
    per = p.groupby("r0").agg(gross=("gross", lambda s: (p.loc[s.index, "w"] * s).sum()),
                              net=("net", "sum"), turnover=("w", lambda s: (s.abs() * p.loc[s.index, "vega0"]).sum()))
    return p, per


def daily_series(p):
    rows = []
    for r in p[p.w != 0].itertuples():
        d = pd.Series(r.w * r.daily, index=r.dates)
        d.iloc[0] -= abs(r.w) * r.cost          # charge all costs on the first day (conservative for DD)
        rows.append(d)
    if not rows:
        return pd.Series(dtype=float)
    return pd.concat(rows).groupby(level=0).sum().sort_index()


def summarise(per, daily, label):
    per = per.copy()
    per["gy"] = D.gas_year(per.index)
    yrs = (per.index.max() - per.index.min()).days / 365.25
    ppy = len(per) / max(yrs, 0.5)
    out = {"rule": label}
    for win, sel in (("choose", per.index.year <= 2016), ("check", per.index.year >= 2017), ("all", np.ones(len(per), bool))):
        q = per[sel]
        sh, lo, hi = sharpe_ci(q.net.values * 100, q.gy.values, ppy, B=1000)
        dd = daily[(daily.index.year <= 2016) if win == "choose" else (daily.index.year >= 2017) if win == "check" else slice(None)]
        cum = dd.cumsum()
        out.update({f"{win}_mean_vp": q.net.mean() * 100, f"{win}_gross_vp": q.gross.mean() * 100,
                    f"{win}_sharpe": sh, f"{win}_sh_lo": lo, f"{win}_sh_hi": hi,
                    f"{win}_maxdd_vp": (cum - cum.cummax()).min() * 100})
    out["turnover"] = per.turnover.mean()
    for name, m in (("jan14", "2014-01"), ("nov18", "2018-11"), ("feb21", "2021-02")):
        out[name] = daily[m].sum() * 100 if len(daily[m]) else np.nan
    return out


RULES = ["S_life", "S_20", "S_ewma", "short_all", "long_all", "term", "storage_sign"]


def main(force=False):
    pd.set_option("display.width", 250)
    reb = monthly_rebal()
    p = build_positions(reb, force=force, tag="m")
    p = add_signals(p)
    res, keep = [], {}
    for rule in RULES:
        pp, per = evaluate(p, rule)
        dly = daily_series(pp)
        res.append(summarise(per, dly, rule))
        keep[rule] = (per, dly)
    rng = np.random.default_rng(7)
    rs = []
    for k in range(300):
        pp, per = evaluate(p, "random", rng=rng)
        per["gy"] = D.gas_year(per.index)
        rs.append({"choose": per.net[per.index.year <= 2016].mean() / per.net[per.index.year <= 2016].std() * np.sqrt(12.6),
                   "check": per.net[per.index.year >= 2017].mean() / per.net[per.index.year >= 2017].std() * np.sqrt(12.6),
                   "all_mean_vp": per.net.mean() * 100})
    rs = pd.DataFrame(rs)
    rs.to_csv(D.CACHE / "s3_random.csv", index=False)
    t = pd.DataFrame(res)
    t.to_csv(D.CACHE / "s3_summary.csv", index=False)
    print(t.round(2).T.to_string())
    print("random ranking Sharpe pctiles", rs.quantile([0.05, 0.5, 0.95]).round(2).to_string())
    pd.to_pickle(keep, D.CACHE / "s3_keep.pkl")
    return p, t, rs, keep


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)


def quarterly_rebal():
    return monthly_rebal()[::3]


def variants(force=False):
    """Cost sensitivity, gross CIs, and holding-period variants (quarterly, weekly)."""
    from .stats import block_mean_ci
    rows = []
    sets = {"monthly": ("m", monthly_rebal), "quarterly": ("q", quarterly_rebal), "weekly": ("w", weekly_rebal)}
    for hp, (tag, fn) in sets.items():
        p = add_signals(build_positions(fn(), force=force, tag=tag))
        rules = RULES if hp != "weekly" else ["S_ewma", "S_20", "short_all", "term"]
        for rule in rules:
            for c in (0.0, 0.01, 0.015, 0.02):
                fut = 0.0 if c == 0 else 0.002
                pp, per = evaluate(p, rule, c=c, fut=fut)
                gy = D.gas_year(per.index)
                for win, sel in (("choose", per.index.year <= 2016), ("check", per.index.year >= 2017)):
                    m, lo, hi, k = block_mean_ci(per.net[sel] * 100, gy[sel])
                    sd = per.net[sel].std() * 100
                    ppy = {"monthly": 12.6, "quarterly": 4.2, "weekly": 50.4}[hp]
                    rows.append({"hold": hp, "rule": rule, "c": c, "win": win, "mean_vp": m, "lo": lo, "hi": hi,
                                 "sharpe": m / sd * np.sqrt(ppy) if sd > 0 else np.nan, "n": int(sel.sum())})
        print(hp, "done", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(D.CACHE / "s3_variants.csv", index=False)
    return t


def regime_winter(c=0.015):
    """POST-HOC (informed by the step-1 map): equal-vega book of LONG winter-delivery (Z,F,G,H)
    straddles in the 10-130 tday universe at each monthly rebalance, split by storage regime
    (sdev < -150 tight, > +150 loose).  Per-rebalance mean P&L per unit vega, gas-year CIs."""
    from .stats import block_mean_ci
    p = add_signals(pd.read_pickle(D.CACHE / "s3_positions_m.pkl"))
    p["g"] = p.gross / p.vega0 * 100
    p["n"] = p.g - costs(p, c) / p.vega0 * 100
    p = p[(p.cid % 100).isin(list(D.WINTER_DEL))]
    p["reg"] = pd.cut(p.sdev, [-1e9, -150, 150, 1e9], labels=["tight", "normal", "loose"])
    rows = []
    for (win, reg), g in p.groupby([np.where(p.r0.dt.year <= 2016, "choose", "check"), "reg"], observed=True):
        q = g.groupby("r0")[["g", "n"]].mean()
        gy = D.gas_year(q.index)
        gm, glo, ghi, k = block_mean_ci(q.g, gy)
        nm, nlo, nhi, _ = block_mean_ci(q.n, gy)
        rows.append({"win": win, "regime": reg, "months": len(q), "gas_years": k, "gross": gm, "g_lo": glo,
                     "g_hi": ghi, "net": nm, "n_lo": nlo, "n_hi": nhi})
    t = pd.DataFrame(rows)
    t.to_csv(D.CACHE / "s3_regime_winter.csv", index=False)
    return t
