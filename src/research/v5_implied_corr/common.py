"""v5_implied_corr: shared loaders (development data only, dates <= 2021-12-31).

Hard rules honoured here:
  * futures, v4 panel and raw options are filtered to date <= 2021-12-31 on load;
  * options panel read only from data/processed/v5/ (already cut);
  * no *_locked folder is opened.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CUT = pd.Timestamp("2021-12-31")
V5 = ROOT / "data/processed/v5"
CACHE = ROOT / "data/processed/research_v5_implied_corr"
FIG = ROOT / "reports/agents/figures"
CACHE.mkdir(parents=True, exist_ok=True)

TICK2 = 0.002          # futures: 2 ticks per side, $/MMBtu
CSO_HS = 0.04          # CSO half-spread, fraction of premium (3-5% band; sensitivity 3/5/8%)
ON_HS = 0.015          # monthly ATM option half-spread, fraction of premium (min $0.002 per option)
ON_MIN = 0.002
MMBTU = 10_000         # contract size


def next_cid(cid):
    y, m = divmod(int(cid), 100)
    return y * 100 + m + 1 if m < 12 else (y + 1) * 100 + 1


def futures():
    """Settles pivot (date x cid), trading-day index, option expiry per cid (panel rule)."""
    c = pd.read_csv(ROOT / "data/raw/ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    c = c[c.date <= CUT].dropna(subset=["settle"])          # dev cut on load
    c = c[c.settle > 0]
    c["cid"] = c.contract_year * 100 + c.contract_month
    px = c.pivot_table(index="date", columns="cid", values="settle").sort_index()
    days = px.index
    oexp = {}
    for cid in px.columns:
        y, m = divmod(int(cid), 100)
        p = days.searchsorted(pd.Timestamp(y, m, 1))
        if 4 <= p < len(days):
            oexp[cid] = days[p - 4]
        else:  # delivery month beyond the cut: calendar fallback (same as panel)
            e = pd.Timestamp(y, m, 1) - pd.offsets.BDay(3)
            oexp[cid] = e - pd.offsets.BDay(1)
    return px, days, pd.Series(oexp)


def spreads():
    s = pd.read_csv(V5 / "spreads.csv", parse_dates=["date"])
    s = s[s.date <= CUT].copy()
    s["cm"] = s.cid % 100
    s["far"] = s.cid.map(next_cid)
    s["tdays"] = (s["T"] * 252).round().astype(int)
    # exchange default settles: Bachelier vol pinned at ~1.18 $/sqrt(yr) for months at a time
    s["stale"] = s.nv.between(1.15, 1.21)
    return s


def surface():
    s = pd.read_csv(V5 / "surface.csv", parse_dates=["date"])
    return s[s.date <= CUT].copy()


def raw_options(roots=("ON", "G4X")):
    p = CACHE / "raw_opts.pkl"
    if p.exists():
        return pd.read_pickle(p)
    o = pd.read_csv(ROOT / "data/raw/options/ng_options_near_money_v2.csv.gz", parse_dates=["date"],
                    usecols=["date", "root", "cy", "cm", "cp", "k_raw", "settle"])
    o = o[(o.date <= CUT) & o.root.isin(roots) & (o.settle > 0)].copy()      # dev cut on load
    o["cid"] = (o.cy * 100 + o.cm).astype(int)
    o["K"] = (o.k_raw / 1000.0).round(4)
    o = o[["date", "root", "cid", "K", "cp", "settle"]]
    o.to_pickle(p)
    return o


def panel():
    p = pd.read_csv(ROOT / "data/processed/v4/panel_dev.csv", parse_dates=["date"])
    return p[p.date <= CUT].copy()                                          # dev cut on load


def gas_year(d):
    """Winter block id: Apr(y)..Mar(y+1) -> y+1 (so winter 2013-14 -> 2014)."""
    d = pd.DatetimeIndex(d)
    return np.where(d.month >= 4, d.year + 1, d.year)


def boot_mean(df, col, block="blk", n=4000, seed=0):
    """Winter-block bootstrap of the pooled mean (blocks resampled with replacement)."""
    d = df.dropna(subset=[col])
    if d.empty:
        return np.nan, np.nan, np.nan, 0
    g = d.groupby(block)[col].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    s, c = g["sum"].values[idx].sum(1), g["count"].values[idx].sum(1)
    m = s / np.maximum(c, 1)
    return d[col].mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), len(g)


def boot_blockmean(vals, n=4000, seed=0):
    """Bootstrap of the mean of per-block values (equal weight per winter)."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
