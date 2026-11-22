"""Data for the storage-space-option study.

HARD RULE: every loader drops rows dated >= 2022-01-01 immediately after reading.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "processed" / "research_storage_space_option"
FIG = ROOT / "reports" / "agents" / "figures"
CUTOFF = pd.Timestamp("2022-01-01")
MULT = 10_000  # MMBtu per contract
TICK_USD = 10.0
YEARS = range(2010, 2022)


def load_settles():
    d = pd.read_csv(RAW / "ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    d = d[d.date < CUTOFF]
    d = d[d.settle.notna() & (d.settle > 0)]
    d["k"] = d.contract_year * 12 + d.contract_month - 1
    return d


def wide_settles(d=None):
    d = load_settles() if d is None else d
    return d.pivot_table(index="date", columns="k", values="settle")


def kidx(year, month):
    return year * 12 + month - 1


def last_trade(w, year, month):
    """Last trading date of a contract (its last settle in the sample), if it expired < 2022."""
    k = kidx(year, month)
    if k not in w.columns:
        return None
    s = w[k].dropna()
    if s.empty or s.index.max() >= CUTOFF - pd.Timedelta(days=5):
        return None
    # must be the month before delivery (otherwise the series is truncated by the sample)
    last = s.index.max()
    exp_m = (pd.Timestamp(year, month, 1) - pd.Timedelta(days=1))
    return last if (last.year, last.month) == (exp_m.year, exp_m.month) else None


def spread_series(w, year, near_m, far_y, far_m):
    """Contango = far - near ($/MMBtu), daily, for one autumn."""
    a, b = kidx(year, near_m), kidx(far_y, far_m)
    if a not in w.columns or b not in w.columns:
        return pd.Series(dtype=float)
    return (w[b] - w[a]).dropna()


SPREADS = {  # name: (near month, far-year offset, far month)
    "XV": (10, 0, 11),   # Nov - Oct  (the "V-X" spread, sign flipped: + = contango)
    "VU": (9, 0, 10),    # Oct - Sep
    "FV": (10, 1, 1),    # Jan(next) - Oct
}


def load_storage():
    s = pd.read_csv(RAW / "storage.csv", parse_dates=["week_ending"])
    s = s[s.week_ending < CUTOFF].set_index("week_ending").storage_bcf.sort_index()
    return s


def end_october_week(s, year):
    c = s.index[(s.index >= pd.Timestamp(year, 10, 25)) & (s.index <= pd.Timestamp(year, 10, 31))]
    return c[-1] if len(c) else None


def avg5_at(s, date, min_years=3):
    """Calendar-aligned average of the prior 5 years' levels at `date` (time-interpolated)."""
    daily = s.resample("D").interpolate("time")
    vals = [daily.get(date - pd.DateOffset(years=y)) for y in range(1, 6)]
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    return (np.mean(vals), len(vals)) if len(vals) >= min_years else (np.nan, len(vals))


def five_year_path(s, F, T, min_years=3):
    """Level at T implied by level at F plus the average change F->T in the prior 5 years
    (calendar-aligned, time-interpolated). Uses only data dated <= F."""
    daily = s[s.index <= F].resample("D").interpolate("time")
    ch = []
    for y in range(1, 6):
        a, b = F - pd.DateOffset(years=y), T - pd.DateOffset(years=y)
        if a in daily.index and b in daily.index:
            ch.append(daily[b] - daily[a])
    return s[F] + np.mean(ch) if len(ch) >= min_years else np.nan


def autumn_table():
    """One row per autumn: expiry-date spreads and realised end-October storage."""
    w = wide_settles()
    s = load_storage()
    rows = []
    for y in YEARS:
        T = end_october_week(s, y)
        if T is None:
            continue
        lv = last_trade(w, y, 10)
        lu = last_trade(w, y, 9)
        r = {"year": y, "T": T, "S_T": float(s[T]), "V_last": lv, "U_last": lu}
        a5, n5 = avg5_at(s, T)
        r["avg5_T"], r["n5"] = a5, n5
        r["dev_T"] = r["S_T"] - a5
        r["max_prior"] = float(s[s.index < pd.Timestamp(y, 1, 1)].max())  # walk-forward capacity proxy
        r["head_T"] = r["max_prior"] - r["S_T"]
        for name, (nm, fo, fm) in SPREADS.items():
            sp = spread_series(w, y, nm, y + fo, fm)
            exp_day = lu if name == "VU" else lv
            r[f"{name}_exp"] = float(sp.get(exp_day, np.nan)) if exp_day is not None else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def public_dates(trading_dates, F):
    """First trading date on/after the Thursday release (F + 6 days) of week ending F."""
    rel = F + pd.Timedelta(days=6)
    i = trading_dates.searchsorted(rel)
    return trading_dates[i] if i < len(trading_dates) else None
