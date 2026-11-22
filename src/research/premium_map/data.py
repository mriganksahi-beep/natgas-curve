"""Data for the premium-map study (agent `premium_map`).

HARD RULE: every loader drops rows dated on/after 2022-01-01 immediately after reading.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "processed" / "research_premium_map"
FIG = ROOT / "reports" / "agents" / "figures"
CUTOFF = pd.Timestamp("2022-01-01")
MULT = 10_000          # MMBtu per contract -> $ per $1/MMBtu
TICK_USD = 10.0        # $ per tick per contract
MONTHS = "JFMAMJJASOND"
MNAME = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load_wide():
    """Daily settle matrix (date x absolute delivery month k = 12*year + month - 1), dates < 2022,
    plus the last trading date of every contract that expired inside the sample."""
    d = pd.read_csv(RAW / "ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    d = d[d.date < CUTOFF]
    assert d.date.max() < CUTOFF
    d = d[d.settle.notna() & (d.settle > 0)]
    d["k"] = d.contract_year * 12 + d.contract_month - 1
    W = d.pivot_table(index="date", columns="k", values="settle", aggfunc="last").sort_index()
    last = d.groupby("k").date.max()
    # A contract expired in-sample if its delivery month starts on/before 2022-01-01
    # (Jan-2022 expired 2021-12-29; Feb-2022 and later are still trading at the cutoff).
    kmax = 2022 * 12 + 0
    expiry = last[last.index <= kmax]
    return W, expiry


def k_label(k):
    return f"{MNAME[k % 12]}{k // 12 % 100:02d}"


def storage_state(dates):
    """For each date t: the latest *public* EIA level (week ending F counts from F+6, Thursday),
    its calendar-aligned 5-year average (min 3 prior years, prior years only), the deviation,
    and the 5-year-average path forward: projected end-March and end-October levels
    = avg5(target date) + current deviation (nothing realised is used)."""
    s = pd.read_csv(RAW / "storage.csv", parse_dates=["week_ending"])
    s = s[s.week_ending < CUTOFF].set_index("week_ending").storage_bcf.sort_index()
    daily = s.resample("D").interpolate("time")

    def avg5_at(date, max_year):
        """mean of the level at the same calendar date in the 5 years before `max_year`+1."""
        vals = []
        for y in range(1, 6):
            dd = date - pd.DateOffset(years=y)
            if dd.year > max_year:
                continue
            v = daily.get(dd)
            if v is not None and not np.isnan(v):
                vals.append(v)
        return np.mean(vals) if len(vals) >= 3 else np.nan

    rows = []
    for F in s.index:
        a5 = avg5_at(F, F.year - 1)
        # target dates for the 5-year path: next end-March (31 Mar) and next end-October
        nm = pd.Timestamp(F.year if F.month <= 3 else F.year + 1, 3, 31)
        no = pd.Timestamp(F.year if F.month <= 10 else F.year + 1, 10, 31)
        # avg5 at a future target uses only years that are complete before F's year
        a5m = _avg_years(daily, nm, F)
        a5o = _avg_years(daily, no, F)
        rows.append((F, s[F], a5, a5m, a5o))
    w = pd.DataFrame(rows, columns=["week", "storage", "avg5", "avg5_endmar", "avg5_endoct"]).set_index("week")
    w["dev"] = w.storage - w.avg5
    w["proj_endmar"] = w.avg5_endmar + w.dev
    w["proj_endoct"] = w.avg5_endoct + w.dev
    w.index = w.index + pd.Timedelta(days=6)          # public date
    out = w.reindex(pd.DatetimeIndex(dates), method="ffill")
    return out


def _avg_years(daily, target, now):
    """5-yr average of the level at `target`'s calendar date, using only years whose
    target date is already in the past at `now`."""
    vals = []
    for y in range(1, 8):
        dd = target - pd.DateOffset(years=y)
        if dd >= now:
            continue
        v = daily.get(dd)
        if v is not None and not np.isnan(v):
            vals.append(v)
        if len(vals) == 5:
            break
    return np.mean(vals) if len(vals) >= 3 else np.nan
