"""Data for v4_widowmaker. Hard cut: every loader drops dates after 2021-12-31 (asserted).

Winter Y = March (H) / April (J) contracts of delivery year Y; S = H_Y - J_Y ($/MMBtu).
Entries (both on the settle of the entry day):
  'nov' : first trading day on/after the first Friday on/after 1 Nov of Y-1 (v3 base trade)
  'dec' : first trading day on/after 1 Dec of Y-1 (runnable unchanged on the locked 2000-10 panel,
          whose H-J exists only from late November)
End = last H_Y settle (expiry) for Y <= 2021; 2021-12-31 for Y = 2022 (truncated, the 2022 expiry is
never read).

Forecast signals (all known at the settle of day d: the GEFS 00Z run of d is public ~06:00 ET):
  L23  = lvl_w2 + lvl_w3   (forecast HDD anomaly, days 6-15 ahead, gas-weighted, + = cold)
  L123 = lvl_w1 + lvl_w2 + lvl_w3 (days 1-15)
  NT   = news_tot (revision of the whole 0-15-day path since the previous run)
Standardisation is WALK-FORWARD: z(d) = x(d) / RMS of x over all Oct-Apr panel days strictly
before d (anomalies are zero-mean by construction against the model climatology; min 20 earlier
days, else NaN -> treated as 'not cold'). Rolling k-day means / sums use days <= d only.
No pre-2010 row is read anywhere.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CUTOFF = pd.Timestamp("2021-12-31")
CACHE = ROOT / "data/processed/research_v4_widowmaker"
CACHE.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "reports/agents/figures"
MULT = 10_000.0
SIDE = {2: 2 * 2 * 10.0, 4: 2 * 4 * 10.0}   # $ per spread per side: 2 legs x ticks x $10
YEARS = list(range(2011, 2023))
TRAIN = list(range(2011, 2018))      # 2010-11 .. 2016-17
CHECK = list(range(2018, 2022))      # 2017-18 .. 2020-21 (complete)
TRUNC = 2022                         # 2021-22, truncated at 2021-12-31


def wl(Y):
    return f"{Y - 1}-{str(Y)[2:]}"


def load_px():
    c = pd.read_csv(ROOT / "data/raw/ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    c = c[(c.date <= CUTOFF) & (c.date >= "2010-06-07") & c.contract_month.isin([3, 4])
          & c.contract_year.between(2011, 2022)]
    px = c.dropna(subset=["settle"]).pivot_table(index="date", columns=["contract_year", "contract_month"],
                                                 values="settle")
    assert px.index.max() <= CUTOFF and px.index.min() >= pd.Timestamp("2010-06-07")
    return px


def load_panel():
    p = pd.read_csv(ROOT / "data/processed/v4/panel_dev.csv", parse_dates=["date"])
    p = p[p.date <= CUTOFF].set_index("date").sort_index()
    assert p.index.max() <= CUTOFF
    return p


def _wf_rms(x: pd.Series, season: pd.Series, min_n=20):
    """RMS of x over season days strictly before each date (walk-forward)."""
    v = x.where(season & x.notna())
    s2 = (v ** 2).fillna(0).cumsum().shift(1)
    n = v.notna().astype(int).cumsum().shift(1)
    r = np.sqrt(s2 / n)
    return r.where(n >= min_n)


def signals():
    p = load_panel()
    season = pd.Series(p.index.month.isin([10, 11, 12, 1, 2, 3, 4]), index=p.index)
    out = pd.DataFrame(index=p.index)
    out["L23"] = p.lvl_w2 + p.lvl_w3
    out["L123"] = p.lvl_w1 + p.lvl_w2 + p.lvl_w3
    out["NT"] = p.news_tot
    out["gefs_system"] = p.gefs_system
    for c in ("L23", "L123"):
        out[f"z{c}"] = out[c] / _wf_rms(out[c], season)
        for k in (1, 3, 5):
            out[f"z{c}_{k}"] = out[f"z{c}"].rolling(k, min_periods=k).mean()
    for k in (2, 5, 10):
        sk = out.NT.rolling(k, min_periods=k).sum()
        out[f"zN_{k}"] = sk / _wf_rms(sk, season)
    # tightness-scaled news (news_channel's weekly H-J form without its fitted slope, which cancels
    # under standardisation): X = sum of news_tot x (max(S_ref, 0) + 0.10), S_ref = H-J at the start
    # of the window. Rolling 5-day (daily) and Friday-to-Friday (in force from the Friday settle).
    tight = p.hj.clip(lower=0) + 0.10
    xr = sum(out.NT.shift(j) for j in range(5)) * tight.shift(5)
    out["zNS_d5"] = xr / _wf_rms(xr, season)
    wk = p.index.to_period("W-FRI")
    last = pd.Series(p.index, index=p.index).groupby(wk).transform("max")
    is_end = pd.Series(p.index == last.values, index=p.index)
    ref = tight.where(is_end).shift(1).ffill()           # tightness at the previous week end
    xw = (out.NT * ref).groupby(wk).cumsum()
    xw_end = xw.where(is_end)
    seas_w = season & is_end
    zw = xw_end / _wf_rms(xw_end, seas_w, min_n=8)
    out["zNS_w"] = zw.ffill(limit=5)
    return out


def winters(entry="nov"):
    f = CACHE / f"winters_{entry}.pkl"
    if f.exists():
        return pd.read_pickle(f)
    px, sg = load_px(), signals()
    out = {}
    for Y in YEARS:
        S = (px[(Y, 3)] - px[(Y, 4)]).dropna()
        S = S[S.index >= pd.Timestamp(Y - 1, 10, 1)]
        if entry == "nov":
            nov1 = pd.Timestamp(Y - 1, 11, 1)
            t0 = nov1 + pd.Timedelta(days=(4 - nov1.weekday()) % 7)
        else:
            t0 = pd.Timestamp(Y - 1, 12, 1)
        ent = S.index[S.index.searchsorted(t0)]
        if Y <= 2021:
            assert S.index.max().month == 2 and S.index.max().year == Y
        d = pd.DataFrame({"S": S}).join(sg, how="left")
        d = d[d.index >= ent].copy()
        d.attrs.update(entry=ent, end=d.index.max(), Y=Y)
        out[Y] = d
    pd.to_pickle(out, f)
    return out
