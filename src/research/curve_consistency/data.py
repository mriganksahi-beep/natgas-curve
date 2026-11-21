"""Data loading for the curve-consistency study (Q1).

HARD RULE: every loader filters to dates < 2022-01-01 immediately after reading.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "processed" / "research_curve_consistency"
FIG = ROOT / "reports" / "agents" / "figures"
CUTOFF = pd.Timestamp("2022-01-01")
NPOS = 18          # first 18 listed contracts
TICK = 0.001       # $/MMBtu
MULT = 10_000      # MMBtu per contract


def load_curve():
    """Long panel of outright settles, dates < 2022, with curve position and days-to-expiry."""
    d = pd.read_csv(RAW / "ng_curve_daily.csv", parse_dates=["date"])
    d = d[d.date < CUTOFF].copy()
    d = d[d.date >= "2010-06-01"]
    d["k"] = d.contract_year * 12 + d.contract_month - 1          # absolute delivery month index
    dates = np.sort(d.date.unique())
    didx = pd.Series(np.arange(len(dates)), index=dates)
    d["di"] = didx.loc[d.date].values
    # last trading date of each contract, only if it occurs inside the sample
    # (contracts delivering Feb-2022 or later never get close to expiry inside the sample)
    last = d.groupby("k").di.max()
    exp_in_sample = d.groupby("k").date.max() < (CUTOFF - pd.Timedelta(days=5))
    d["dte"] = d.k.map(last) - d.di                                # trading days to last trade
    d.loc[~d.k.map(exp_in_sample).astype(bool), "dte"] = 999
    d = d[d.settle.notna() & (d.settle > 0)]
    # curve position: 1 = nearest unexpired contract
    d = d.sort_values(["di", "k"])
    d["pos"] = d.groupby("di").cumcount() + 1
    d = d[d.pos <= NPOS + 2]
    return d.reset_index(drop=True), pd.DatetimeIndex(dates)


def wide(d, col="settle"):
    """date-index x absolute-month matrix."""
    return d.pivot(index="di", columns="k", values=col)


def storage_deviation(dates):
    """For each trading date: latest *public* storage level minus its calendar-aligned
    5-year average (min 3 prior years). Week ending Friday F is public Thursday F+6, 10:30 ET,
    i.e. before the 14:30 settle, so a date t may use F with F + 6 days <= t."""
    s = pd.read_csv(RAW / "storage.csv", parse_dates=["week_ending"])
    s = s[s.week_ending < CUTOFF].set_index("week_ending").storage_bcf.sort_index()
    daily = s.resample("D").interpolate("time")
    avg, n = [], []
    for F in s.index:
        vals = [daily.get(F - pd.DateOffset(years=y)) for y in range(1, 6)]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        avg.append(np.mean(vals) if len(vals) >= 3 else np.nan)
        n.append(len(vals))
    w = pd.DataFrame({"storage": s.values, "avg5": avg}, index=s.index)
    w["dev"] = w.storage - w.avg5
    w["public"] = w.index + pd.Timedelta(days=6)
    w = w.set_index("public")
    out = w.reindex(pd.DatetimeIndex(dates), method="ffill")
    return out[["storage", "avg5", "dev"]]
