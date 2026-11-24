"""v5_vol_rv data layer.

Hard rules applied here:
  * options: read ONLY data/processed/v5/ (already cut at 2021-12-31); asserted.
  * futures: rows <= 2021-12-31 are used for every FEATURE.  Futures settles from 2022-01-01 to
    2022-03-31 are loaded only as OUTCOMES (realised vol) for positions opened <= 2021-12-31 on
    contracts whose option expires by 2022-03-31 (brief rule 1).  Features never see them:
    every feature at date d uses returns dated <= d (enforced by construction + assert).
  * v4 panel and storage: cut at 2021-12-31 on load.  Nothing pre-2010 is read.

Conventions: log returns of each futures contract (cid = YYYYMM) on its own settles;
realised vol is zero-mean, annualised with 252, over trading days (d, end].
Option expiry for cid = trading day before the futures last trade (same rule as options_panel).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CUT = pd.Timestamp("2021-12-31")
OUTCOME_CUT = pd.Timestamp("2022-03-31")
V5 = ROOT / "data/processed/v5"
CACHE = ROOT / "data/processed/research_v5_vol_rv"
CACHE.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "reports/agents/figures"
WINTER_DEL = {12, 1, 2, 3}          # Z F G H delivery
SHOULDER_DEL = {4, 5, 10, 11}       # J K V X
SUMMER_DEL = {6, 7, 8, 9}           # M N Q U


def dclass(m):
    return "winter" if m in WINTER_DEL else ("shoulder" if m in SHOULDER_DEL else "summer")


def gas_year(dates):
    """Block id for the winter bootstrap: gas year Apr..Mar labelled by its winter's March year."""
    d = pd.DatetimeIndex(dates)
    return np.where(d.month >= 4, d.year + 1, d.year)


# ----------------------------------------------------------------------------- options (cut panel)
def surface():
    s = pd.read_csv(V5 / "surface.csv", parse_dates=["date"])
    assert s.date.max() <= CUT
    s["tdays"] = (s["T"] * 252).round().astype(int)
    return s


def dh():
    d = pd.read_csv(V5 / "dh_straddle.csv", parse_dates=["date", "date1"])
    assert d.date.max() <= CUT
    return d


# ----------------------------------------------------------------------------- futures
def futures():
    """Returns (px, days, oexp): settle pivot to OUTCOME_CUT, trading days, option expiry per cid."""
    c = pd.read_csv(ROOT / "data/raw/ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    c = c[(c.date <= OUTCOME_CUT)].dropna(subset=["settle"])
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
        else:
            oexp[cid] = pd.Timestamp(y, m, 1) - pd.offsets.BDay(4)
    return px, days, pd.Series(oexp)


def storage_dev(dates):
    """Latest public Lower-48 storage minus its calendar-aligned mean over the prior up-to-5 years
    (min 2 prior years, so defined from early 2012).  Week ending F is public on F+6 (Thursday)."""
    s = pd.read_csv(ROOT / "data/raw/storage.csv", parse_dates=["week_ending"])
    s = s[s.week_ending <= CUT].set_index("week_ending").storage_bcf.sort_index()
    daily = s.resample("D").interpolate("time")
    avg = []
    for F in s.index:
        v = [daily.get(F - pd.DateOffset(years=y)) for y in range(1, 6)]
        v = [x for x in v if x is not None and np.isfinite(x)]
        avg.append(np.mean(v) if len(v) >= 2 else np.nan)
    w = pd.DataFrame({"storage": s.values, "dev": s.values - np.array(avg)}, index=s.index + pd.Timedelta(days=6))
    return w.reindex(pd.DatetimeIndex(dates), method="ffill")["dev"]


def panel():
    p = pd.read_csv(ROOT / "data/processed/v4/panel_dev.csv", parse_dates=["date"],
                    usecols=["date", "lvl_w1", "lvl_w2", "news_tot", "hj", "px_front"])
    return p[p.date <= CUT].set_index("date")
