"""v4_options data layer. Every loader cuts at 2021-12-31 (hard rule 1); nothing pre-2010 is read.

Winter Y = March (H) / April (J) contracts of delivery year Y (Nov Y-1 .. Feb Y).
S_t   = H_Y - J_Y futures settle spread ($/MMBtu).
G4X March options of year Y are options on S (cm = 3).  Options expire one trading day before
the H futures' last trading day (futures LTD = 3 trading days before 1 March).
Time to expiry T = calendar days / 365.25 (same convention as src/v4/panel.py); r = 0.

Smile for missing strikes: on each (date, Y) we invert Bachelier vols from the out-of-the-money
settle (call for K >= F, put for K < F; F = futures spread), then interpolate the normal vol
linearly in strike (flat beyond the quoted range). If a date has no G4X quotes at all, the last
available smile (by strike) is carried forward and repriced at the day's F and T.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[3]
CUTOFF = pd.Timestamp("2021-12-31")
CACHE = ROOT / "data/processed/research_v4_options"
CACHE.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "reports/agents/figures"
MULT = 10_000.0                 # MMBtu per leg / per spread option
TICK = 10.0                     # $ per tick per leg
YEARS = list(range(2013, 2023))  # G4X winters (2013 = winter 2012-13); 2022 truncated at cut
CHOOSE = [2013, 2014, 2015, 2016, 2017]
CHECK = [2018, 2019, 2020, 2021, 2022]


def wl(Y):
    return f"{Y - 1}-{str(Y)[2:]}"


# ----------------------------------------------------------------------------- Bachelier / Black
def bach_call(F, K, T, s):
    F, K, T, s = map(np.asarray, (F, K, T, s))
    v = s * np.sqrt(np.maximum(T, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (F - K) / v
        out = (F - K) * norm.cdf(d) + v * norm.pdf(d)
    return np.where(v > 0, out, np.maximum(F - K, 0))


def bach_put(F, K, T, s):
    return bach_call(F, K, T, s) - (np.asarray(F) - np.asarray(K))


def black_call(F, K, T, s):
    F, K, T, s = map(np.asarray, (F, K, T, s))
    v = s * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * v * v) / v
    return F * norm.cdf(d1) - K * norm.cdf(d1 - v)


def invert(fn, price, F, K, T, lo, hi, n=60):
    """Vectorised bisection; NaN where price is outside [fn(lo), fn(hi)]."""
    price, F, K, T = (np.asarray(x, float) for x in (price, F, K, T))
    a, b = np.full(price.shape, lo), np.full(price.shape, hi)
    ok = (fn(F, K, T, a) <= price + 1e-12) & (fn(F, K, T, b) >= price)
    for _ in range(n):
        m = 0.5 * (a + b)
        up = fn(F, K, T, m) < price
        a, b = np.where(up, m, a), np.where(up, b, m)
    return np.where(ok, 0.5 * (a + b), np.nan)


# ----------------------------------------------------------------------------- futures
def futures():
    """Settle pivot (date x cid) for all contracts, cut at CUTOFF, plus trading-day calendar."""
    c = pd.read_csv(ROOT / "data/raw/ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    c = c[c.date <= CUTOFF].dropna(subset=["settle"])
    c["cid"] = c.contract_year * 100 + c.contract_month
    px = c.pivot_table(index="date", columns="cid", values="settle").sort_index()
    assert px.index.max() <= CUTOFF
    return px


def expiries(Y, days):
    """(futures LTD, option expiry) of the March Y contract, from the trading calendar."""
    first = pd.Timestamp(Y, 3, 1)
    if first > days[-1]:
        f = first - pd.offsets.BDay(3)
        return f, f - pd.offsets.BDay(1)
    p = days.searchsorted(first)
    return days[p - 3], days[p - 4]


def spread_frame():
    """dict Y -> DataFrame(index date): S, H, J, T (years to option expiry), with attrs fut_exp, opt_exp."""
    f = CACHE / "spreads.pkl"
    if f.exists():
        return pd.read_pickle(f)
    px = futures()
    days = px.index
    out = {}
    for Y in range(2011, 2023):
        H, J = Y * 100 + 3, Y * 100 + 4
        d = pd.DataFrame({"H": px[H], "J": px[J]}).dropna()
        fe, oe = expiries(Y, days)
        d = d[(d.index >= pd.Timestamp(Y - 2, 9, 1)) & (d.index <= fe)]
        d["S"] = d.H - d.J
        d["T"] = (oe - d.index).days / 365.25
        nov1 = pd.Timestamp(Y - 1, 11, 1)
        fri = nov1 + pd.Timedelta(days=(4 - nov1.weekday()) % 7)
        d.attrs.update(fut_exp=fe, opt_exp=oe, entry=d.index[d.index.searchsorted(fri)])
        out[Y] = d
    pd.to_pickle(out, f)
    return out


# ----------------------------------------------------------------------------- options
def g4x_march():
    """Long table of G4X March options, cut, with junk strikes removed (|K - F| > 3)."""
    f = CACHE / "g4x_march.pkl"
    if f.exists():
        return pd.read_pickle(f)
    o = pd.read_csv(ROOT / "data/raw/options/ng_options_near_money.csv.gz", parse_dates=["date"])
    o = o[(o.date <= CUTOFF) & (o.root == "G4X") & (o.cm == 3) & o.cy.isin(YEARS)]
    o["K"] = (o.k_raw / 1000.0).round(3)
    o = o.pivot_table(index=["cy", "date", "K"], columns="cp", values="settle").reset_index()
    sp = spread_frame()
    rows = []
    for Y, g in o.groupby("cy"):
        s = sp[Y]
        g = g[g.date.isin(s.index)].copy()
        g["F"] = s.S.reindex(g.date).values
        g["T"] = s["T"].reindex(g.date).values
        rows.append(g)
    o = pd.concat(rows)
    o = o[(o.K - o.F).abs() <= 3.0]
    o = o[o["T"] > 1 / 365.25]
    # OTM settle -> normal vol
    otm_c = o.K >= o.F
    prem = np.where(otm_c, o.C, o.P)
    o["prem_otm"] = prem
    vol_c = invert(bach_call, o.C.values, o.F.values, o.K.values, o["T"].values, 1e-5, 30.0)
    vol_p = invert(bach_put, o.P.values, o.F.values, o.K.values, o["T"].values, 1e-5, 30.0)
    o["nv"] = np.where(otm_c, vol_c, vol_p)
    o["nv"] = o["nv"].where(np.isfinite(o["nv"]), np.where(otm_c, vol_p, vol_c))
    o["parity_err"] = (o.C - o.P) - (o.F - o.K)
    o.to_pickle(f)
    return o


class Smile:
    """Per-winter callable: price(date, K) of a G4X March call, from settle or smile interpolation."""

    def __init__(self, Y, opts, spread):
        self.Y, self.s = Y, spread
        g = opts[(opts.cy == Y) & opts.nv.notna()]
        self.by_date = {d: (x.K.values, x.nv.values, dict(zip(x.K.values, x.C.values)))
                        for d, x in g.sort_values("K").groupby("date")}
        self.dates = sorted(self.by_date)
        self.n_interp = self.n_carry = self.n_exact = 0

    def vol(self, d, K):
        i = np.searchsorted(self.dates, d, side="right") - 1
        if i < 0:
            return np.nan, None
        dd = self.dates[i]
        ks, nv, _ = self.by_date[dd]
        return float(np.interp(K, ks, nv)), dd

    def call(self, d, K, count=True):
        F, T = self.s.at[d, "S"], self.s.at[d, "T"]
        if T <= 0:
            return max(F - K, 0.0)
        if d in self.by_date and K in self.by_date[d][2] and np.isfinite(self.by_date[d][2][K]):
            self.n_exact += count
            return float(self.by_date[d][2][K])
        v, dd = self.vol(d, K)
        if dd is None:
            return np.nan
        if dd == d:
            self.n_interp += count
        else:
            self.n_carry += count
        return float(bach_call(F, K, T, v))


def on_hj():
    """ON settles for H and J of each winter (for the iv_h / iv_j check)."""
    f = CACHE / "on_hj.pkl"
    if f.exists():
        return pd.read_pickle(f)
    o = pd.read_csv(ROOT / "data/raw/options/ng_options_near_money.csv.gz", parse_dates=["date"])
    o = o[(o.date <= CUTOFF) & (o.root == "ON") & o.cm.isin([3, 4]) & o.cy.isin(range(2011, 2023))]
    o["K"] = (o.k_raw / 1000.0).round(3)
    o = o.pivot_table(index=["cy", "cm", "date", "K"], columns="cp", values="settle").reset_index()
    o.to_pickle(f)
    return o


def panel():
    p = pd.read_csv(ROOT / "data/processed/v4/panel_dev.csv", parse_dates=["date"])
    p = p[p.date <= CUTOFF]
    return p.set_index("date")
