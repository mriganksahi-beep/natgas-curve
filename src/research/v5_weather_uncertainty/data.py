"""v5_weather_uncertainty data layer.

Hard rules applied here:
  * options and weather_unc: read ONLY data/processed/v5/ (already cut at 2021-12-31); asserted.
    The panel directory can be redirected with env V5_DIR (frozen locked test) - nothing else changes.
  * futures: every FEATURE at date d uses settles dated <= d (<= 2021-12-31).  Settles from 2022-01-01
    to 2022-03-31 enter only as OUTCOMES (forward realised vol) for rows dated <= 2021-12-31 (rule 1).
  * v4 panel and storage: cut at 2021-12-31 on load.  Nothing pre-2010 is read.

Base table: one row per (date d in the Oct-Apr weather_unc sample, horizon h in {5,10,20}, leg in {1,2}).
  leg 1 = nearest ON contract whose option has MORE than h trading days to expiry (so the forward window
  lies inside the option's life); leg 2 = the next listed contract.  Per row:
    atm_iv, tdays, rv5/rv21/rv63 (trailing, own contract, zero-mean, x252), rvf (forward RV over d+1..d+h),
    unc_* (weather_unc), lvl_w1..3 (v4 panel forecast level, HDD anomaly sums), sdev (storage deviation,
    vol_rv definition, from 2012), month, gy (gas year Apr..Mar), win (choose <= 2016 < check).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.v5_vol_rv import data as VD   # reuse (read-only) futures/storage loaders

ROOT = Path(__file__).resolve().parents[3]
CUT = pd.Timestamp(os.environ.get("V5_CUT", "2021-12-31"))
V5 = Path(os.environ.get("V5_DIR", ROOT / "data/processed/v5"))
CACHE = Path(os.environ.get("V5WU_CACHE", ROOT / "data/processed/research_v5_weather_uncertainty"))
CACHE.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "reports/agents/figures"
HS = (5, 10, 20)
UNC = ["unc_n_w1", "unc_n_w2", "unc_n_w3"]


def window(d):
    d = pd.DatetimeIndex(d)
    return np.where(d.year <= 2016, "choose", "check")


def gas_year(d):
    return VD.gas_year(d)


def weather_unc():
    w = pd.read_csv(V5 / "weather_unc.csv", parse_dates=["date"])
    assert w.date.max() <= CUT
    w = w.set_index("date").sort_index()
    # smoothed / derived (all use runs <= d only)
    w["unc_n_w23"] = w[["unc_n_w2", "unc_n_w3"]].mean(1)
    for c in UNC + ["unc_n_w23"]:
        w[c + "_m3"] = w[c].rolling(3, min_periods=2).mean()
        w["dl_" + c] = np.log(w[c]).diff()              # 1-run change in normalised log disagreement
        w["d5l_" + c] = np.log(w[c + "_m3"]).diff(5)    # 5-day change of the 3-day mean
    gap = w.index.to_series().diff().dt.days > 6        # first row of each winter: no change
    w.loc[gap, [c for c in w.columns if c.startswith("dl_")]] = np.nan
    return w


def surface():
    s = pd.read_csv(V5 / "surface.csv", parse_dates=["date"])
    assert s.date.max() <= CUT
    s["tdays"] = (s["T"] * 252).round().astype(int)
    return s


def dh():
    d = pd.read_csv(V5 / "dh_straddle.csv", parse_dates=["date", "date1"])
    assert d.date.max() <= CUT
    return d


def futures():
    """Settle pivot up to CUT + 3 months (outcomes only), trading days, option expiry per cid."""
    px, days, oexp = VD.futures() if CUT == VD.CUT else _futures_generic()
    return px, days, oexp


def _futures_generic():
    c = pd.read_csv(ROOT / "data/raw/ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    c = c[c.date <= CUT + pd.DateOffset(months=3)].dropna(subset=["settle"])
    c = c[c.settle > 0]
    c["cid"] = c.contract_year * 100 + c.contract_month
    px = c.pivot_table(index="date", columns="cid", values="settle").sort_index()
    days = px.index
    oexp = {}
    for cid in px.columns:
        y, m = divmod(int(cid), 100)
        p = days.searchsorted(pd.Timestamp(y, m, 1))
        oexp[cid] = days[p - 4] if 4 <= p < len(days) else pd.Timestamp(y, m, 1) - pd.offsets.BDay(4)
    return px, days, pd.Series(oexp)


def panel():
    p = pd.read_csv(ROOT / "data/processed/v4/panel_dev.csv", parse_dates=["date"],
                    usecols=["date", "lvl_w1", "lvl_w2", "lvl_w3", "r_front", "front"])
    return p[p.date <= CUT].set_index("date")


def storage_dev(dates):
    """vol_rv definition (latest public storage minus mean of the same date in the prior 2-5 years), cut at CUT."""
    if CUT == VD.CUT:
        return VD.storage_dev(dates)
    s = pd.read_csv(ROOT / "data/raw/storage.csv", parse_dates=["week_ending"])
    s = s[s.week_ending <= CUT].set_index("week_ending").storage_bcf.sort_index()
    daily = s.resample("D").interpolate("time")
    avg = []
    for F in s.index:
        v = [daily.get(F - pd.DateOffset(years=y)) for y in range(1, 6)]
        v = [x for x in v if x is not None and np.isfinite(x)]
        avg.append(np.mean(v) if len(v) >= 2 else np.nan)
    w = pd.DataFrame({"dev": s.values - np.array(avg)}, index=s.index + pd.Timedelta(days=6))
    return w.reindex(pd.DatetimeIndex(dates), method="ffill")["dev"]


def base(force=False):
    f = CACHE / "base.csv"
    if f.exists() and not force:
        return pd.read_csv(f, parse_dates=["date"])
    w = weather_unc()
    s = surface()
    px, days, oexp = futures()
    lr = np.log(px).diff()
    # rows are dated <= CUT; trailing features at d use returns <= d; post-CUT settles enter forward sums only
    r2 = lr ** 2
    cnt = lr.notna().astype(float)
    trail = {}
    for k in (5, 21, 63):
        trail[k] = np.sqrt(252 * r2.rolling(k, min_periods=int(0.8 * k)).mean())
    # forward sums over t+1..t+h
    fwd = {}
    for h in HS:
        s2 = r2.fillna(0).rolling(h).sum().shift(-h)
        n2 = cnt.rolling(h).sum().shift(-h)
        fwd[h] = np.sqrt(252 * s2 / n2).where(n2 >= h - 1)
    st = s[s.date.isin(w.index) & s.atm_iv.notna()].copy()
    rows = []
    for h in HS:
        g = st[st.tdays > h].sort_values(["date", "cid"])
        g["leg"] = g.groupby("date").cumcount() + 1
        g = g[g.leg <= 2].copy()
        g["h"] = h
        rows.append(g)
    b = pd.concat(rows, ignore_index=True)
    di = days.get_indexer(b.date)
    assert (di >= 0).all()
    cols = {c: px.columns.get_loc(c) for c in px.columns}
    ci = b.cid.map(cols).values
    for k in (5, 21, 63):
        b[f"rv{k}"] = trail[k].values[di, ci]
    b["rvf"] = np.nan
    for h in HS:
        m = (b.h == h).values
        b.loc[m, "rvf"] = fwd[h].values[di[m], ci[m]]
    # forward window must end by the outcome cut and (for dev) rows dated <= CUT only
    b = b.join(w, on="date")
    p = panel()
    b = b.join(p[["lvl_w1", "lvl_w2", "lvl_w3"]], on="date")
    sd = storage_dev(pd.DatetimeIndex(sorted(b.date.unique())))
    b["sdev"] = b.date.map(sd)
    b["month"] = b.date.dt.month
    b["gy"] = gas_year(b.date)
    b["win"] = window(b.date)
    b = b[["date", "h", "leg", "cid", "F", "tdays", "atm_iv", "vega", "rv5", "rv21", "rv63", "rvf",
           "sdev", "lvl_w1", "lvl_w2", "lvl_w3", "month", "gy", "win"] + list(w.columns)]
    b.to_csv(f, index=False)
    return b


if __name__ == "__main__":
    b = base(force=True)
    print(b.groupby(["h", "leg"]).agg(n=("rvf", "size"), rvf=("rvf", "count"), iv=("atm_iv", "mean"),
                                       rv=("rvf", "mean"), sdev=("sdev", "count")))
    print(b.date.min(), b.date.max())
