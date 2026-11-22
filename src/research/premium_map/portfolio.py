"""Systematic premium-harvest portfolio.

Monthly rebalance on the first Friday of each month (d0), hold to the next rebalance or the near-leg
expiry (d1), whichever is first. Universe: the 13 two-leg spreads (strips excluded: 12 legs each).
Cell (pair, L), L = round(months from d0 to near-leg expiry), 1..12.
Selection on trades with d0 <= 2015-12-31 only: cells whose mean monthly step has a gas-year-block
bootstrap |t| >= T_SEL; position sign = sign(mean step) (short spreads that tend to fall).
Sizing: equal ex-ante risk R per spread: units = R / (sd of the instance's weekly changes over the
previous 26 Fridays * sqrt(52/12) * 10,000), weekly sd floored at $0.005.
Costs: TICKS per contract per side on every change in position, 2 contracts per spread unit.
"""
import numpy as np
import pandas as pd

from .data import MULT, TICK_USD
from .spreads import SPREADS, spread_values, gas_year
from .stats import boot_mean

TWO_LEG = [p for p, s in SPREADS.items() if not s["strip"]]
SPLIT = pd.Timestamp("2015-12-31")


def monthly_steps(W, expiry, fridays):
    fridays = pd.DatetimeIndex(fridays)
    reb = fridays.to_series().groupby([fridays.year, fridays.month]).min().values
    reb = pd.DatetimeIndex(reb).append(pd.DatetimeIndex([fridays.max()]))   # last Friday closes Dec-2021
    rows = []
    for name in TWO_LEG:
        spec = SPREADS[name]
        anchors = [k for k in expiry.index if k % 12 == spec["anchor_month"]]
        V, _ = spread_values(W, expiry, name, anchors)
        Vf = V.loc[V.index.isin(fridays)]
        sd = Vf.diff().rolling(26, min_periods=13).std()
        for a in anchors:
            T = expiry[a]
            for i, d0 in enumerate(reb[:-1]):
                if d0 >= T:
                    continue
                L = int(np.round((T - d0).days / 30.4375))
                if L < 1 or L > 12:
                    continue
                d1 = min(reb[i + 1], T)
                v0, v1 = V.at[d0, a], V.at[d1, a]
                if np.isnan(v0) or np.isnan(v1):
                    continue
                s = sd.at[d0, a] if d0 in sd.index else np.nan
                rows.append(dict(pair=name, anchor=a, L=L, d0=d0, d1=d1, step=v1 - v0, sd_wk=s,
                                 gy=gas_year(a), expires=d1 == T))
    return pd.DataFrame(rows)


def select_cells(steps, t_sel=2.0, end=SPLIT):
    tr = steps[steps.d0 <= end]
    out = []
    for (p, L), g in tr.groupby(["pair", "L"]):
        if g.gy.nunique() < 4:
            continue
        m, lo, hi, t = boot_mean(g.step.values, g.gy.values)
        out.append(dict(pair=p, L=L, mean=m, lo=lo, hi=hi, t=t, n=len(g)))
    c = pd.DataFrame(out)
    c["sel"] = c.t.abs() >= t_sel
    c["sign"] = np.sign(c["mean"])          # long spreads that tend to rise, short those that fall
    return c


def run_portfolio(steps, cells, risk=1000.0, ticks=4, floor=0.005):
    """Monthly P&L ($) of the portfolio defined by `cells` (pair, L, sign)."""
    sel = cells[cells.sel][["pair", "L", "sign"]]
    s = steps.merge(sel, on=["pair", "L"], how="inner").copy()
    s = s[s.sd_wk.notna()]
    s["units"] = s["sign"] * risk / (np.maximum(s.sd_wk, floor) * np.sqrt(52 / 12) * MULT)
    s["gross"] = s.units * s.step * MULT
    # costs: change in position of each instance between consecutive months
    s = s.sort_values(["pair", "anchor", "d0"])
    prev_units = s.groupby(["pair", "anchor"]).units.shift(1)
    prev_d1 = s.groupby(["pair", "anchor"]).d1.shift(1)
    contiguous = prev_d1.eq(s.d0)
    entry_change = np.where(contiguous, (s.units - prev_units).abs(), s.units.abs())
    # exit: if the next month is not held contiguously (or the leg expires), the position is closed
    nxt_d0 = s.groupby(["pair", "anchor"]).d0.shift(-1)
    closes = ~nxt_d0.eq(s.d1)
    exit_change = np.where(closes, s.units.abs(), 0.0)
    per_unit_side = 2 * ticks * TICK_USD
    s["cost"] = (entry_change + exit_change) * per_unit_side
    s["net"] = s.gross - s.cost
    s["month"] = s.d0.dt.to_period("M")
    return s


def summarize(pnl_month, label):
    x = pnl_month
    yr = x.groupby(x.index.year).sum()
    sh = x.mean() / x.std(ddof=1) * np.sqrt(12) if x.std() > 0 else np.nan
    # Sharpe CI: bootstrap calendar years
    rng = np.random.default_rng(3)
    years = np.array(sorted(set(x.index.year)))
    byyr = {y: x[x.index.year == y].values for y in years}
    bs = []
    for _ in range(2000):
        v = np.concatenate([byyr[y] for y in rng.choice(years, len(years))])
        if v.std() > 0:
            bs.append(v.mean() / v.std(ddof=1) * np.sqrt(12))
    return dict(label=label, months=len(x), mean_month=x.mean(), sd_month=x.std(ddof=1), sharpe=sh,
                sharpe_lo=np.percentile(bs, 2.5), sharpe_hi=np.percentile(bs, 97.5),
                total=x.sum(), worst_year=yr.min(), worst_year_id=int(yr.idxmin()),
                worst_month=x.min(), worst_month_id=str(x.idxmin()), pos_years=f"{(yr > 0).sum()}/{len(yr)}")


# ---------------------------------------------------------------------------------------------
# Variant B ("hold to expiry"): one entry per spread per year, low turnover.
# For each two-leg pair, on 2010-2015 anchors only, the cumulative drift (entry at lead L, exit at
# expiry) is computed from the monthly steps; the entry lead L* = argmax_L |t| over L = 2..12;
# the pair is traded if |t(L*)| >= T_SEL, sign = -sign(mean drift). Units fixed at entry (equal
# ex-ante risk), costs at entry and exit only.
# ---------------------------------------------------------------------------------------------
def cumulative_from_steps(steps):
    """Per (pair, anchor, L): the sum of steps from the month with lead L to expiry
    (= V(expiry) - V(d0 at lead L)); a short position earns minus this."""
    s = steps.sort_values(["pair", "anchor", "d0"]).copy()
    s["to_exp"] = s.iloc[::-1].groupby(["pair", "anchor"]).step.cumsum().iloc[::-1]
    # keep only instances whose path reaches expiry in-sample
    reach = s.groupby(["pair", "anchor"]).expires.transform("any")
    return s[reach]


def select_pairs(steps, t_sel=2.0, end=SPLIT, Lmin=2):
    cum = cumulative_from_steps(steps)
    # an instance is in the selection sample only if it expired by `end`
    tr = cum[cum.groupby(["pair", "anchor"]).d1.transform("max") <= end]
    rows = []
    for (p, L), g in tr.groupby(["pair", "L"]):
        if L < Lmin or g.gy.nunique() < 4:
            continue
        m, lo, hi, t = boot_mean(-g.to_exp.values, g.gy.values)     # drift = V(t) - V(T)
        rows.append(dict(pair=p, L=L, drift=m, lo=lo, hi=hi, t=t, n=len(g)))
    c = pd.DataFrame(rows)
    best = c.loc[c.groupby("pair").t.apply(lambda x: x.abs().idxmax())].copy()
    best["sel"] = best.t.abs() >= t_sel
    best["sign"] = -np.sign(best.drift)       # positive drift -> short the spread
    return c, best


def run_hold(steps, best, risk=1000.0, ticks=4, floor=0.005, only=None):
    sel = best[best.sel] if only is None else only
    out = []
    for _, r in sel.iterrows():
        g = steps[(steps.pair == r.pair) & (steps.L <= r.L)].sort_values(["anchor", "d0"])
        for a, h in g.groupby("anchor"):
            h = h.copy()
            sd0 = h.sd_wk.iloc[0]
            if np.isnan(sd0):
                continue
            u = r.sign * risk / (max(sd0, floor) * np.sqrt(52 / 12) * MULT)
            h["units"] = u
            h["gross"] = u * h.step * MULT
            h["cost"] = 0.0
            h.iloc[0, h.columns.get_loc("cost")] += abs(u) * 2 * ticks * TICK_USD
            h.iloc[-1, h.columns.get_loc("cost")] += abs(u) * 2 * ticks * TICK_USD
            out.append(h)
    s = pd.concat(out)
    s["net"] = s.gross - s.cost
    s["month"] = s.d0.dt.to_period("M")
    return s


def monthly_series(pf, col="net", start="2010-07-01", end="2021-12-01"):
    m = pf.groupby("month")[col].sum()
    m.index = m.index.to_timestamp()
    return m.reindex(pd.date_range(start, end, freq="MS"), fill_value=0.0)
