"""Weekly panel for Q2, winters 2010-11 to Dec 2021 (no data dated 2022-01-01 or later).

One row per EIA storage week t (Friday, Nov-Feb). The price is the settlement on the first
trading day on or after t + 7 days (the Friday after the Thursday report), as in
src/model3_data.py. Spread = March - April of the delivery year Y (the next March).

Scarcity measures (Bcf, relative to the average end-March level of the prior years):
  avg5[Y]   mean end-March level over Y-1..Y-5 (years with data; storage starts Jan 2010,
            so we require >= 2 prior years -> delivery years 2012 onward)
  x_real    realised end-March level - avg5      (ex-post; known only after March)
  x_info    storage at t + average remaining draw from t's calendar date to end-March in
            the prior years (>= 2 years) - avg5  (ex-ante, the "5-year path" view)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CUTOFF = pd.Timestamp("2021-12-31")
MIN_DAYS_BEFORE_EXPIRY = 25


def load_storage():
    s = pd.read_csv(ROOT / "data/raw/storage.csv", parse_dates=["week_ending"])
    s = s[s.week_ending <= CUTOFF]
    return s.set_index("week_ending")["storage_bcf"]


def load_prices():
    c = pd.read_csv(ROOT / "data/raw/ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    c = c[(c.date <= CUTOFF) & c.contract_month.isin([3, 4]) & (c.contract_year <= 2022)]
    return c.dropna(subset=["settle"]).pivot_table(index="date", columns=["contract_year", "contract_month"],
                                                   values="settle")


def march_expiry(Y):
    """Last trading day of the March contract: 3 business days before 1 March (holidays ignored).
    Used for every year, so the 2022 contract's expiry is never read from data."""
    d = pd.Timestamp(Y, 3, 1)
    return d - pd.offsets.BDay(3)


def end_march(s):
    m = s[(s.index.month == 3) & (s.index.day >= 25)]
    return m.groupby(m.index.year).last()


def _nearest(s, d, tol=3):
    pos = s.index.searchsorted(d)
    c = [p for p in (pos - 1, pos) if 0 <= p < len(s)]
    if not c:
        return None
    p = min(c, key=lambda p: abs(s.index[p] - d))
    return s.iloc[p] if abs(s.index[p] - d) <= pd.Timedelta(days=tol) else None


def build():
    s, px = load_storage(), load_prices()
    em = end_march(s)
    days = px.index
    rows = []
    for t in s.index:
        if t.month not in (11, 12, 1, 2) or t < pd.Timestamp("2010-11-01"):
            continue
        Y = t.year + 1 if t.month >= 11 else t.year
        if (Y, 3) not in px.columns or (Y, 4) not in px.columns:
            continue
        exp = march_expiry(Y)
        pos = days.searchsorted(t + pd.Timedelta(days=7))
        if pos >= len(days):
            continue
        obs = days[pos]
        if obs > exp - pd.Timedelta(days=MIN_DAYS_BEFORE_EXPIRY):
            continue
        h, j = px[(Y, 3)], px[(Y, 4)]
        if pd.isna(h.get(obs)) or pd.isna(j.get(obs)):
            continue
        prior = [em[Y - k] for k in range(1, 6) if (Y - k) in em.index]
        if len(prior) < 2:
            continue
        avg5 = float(np.mean(prior))
        draws = []
        for k in range(1, 6):
            if (Y - k) not in em.index:
                continue
            past = _nearest(s, t - pd.DateOffset(years=k))
            if past is not None:
                draws.append(em[Y - k] - past)
        row = {"week": t, "obs_date": obs, "Y": Y, "winter": f"{Y - 1}-{str(Y)[2:]}",
               "spread": h[obs] - j[obs], "april": j[obs], "march": h[obs],
               "weeks_to_expiry": (exp - obs).days / 7, "storage_t": s[t], "avg5": avg5,
               "n_prior": len(prior),
               "x_info": s[t] + np.mean(draws) - avg5 if len(draws) >= 2 else np.nan,
               "em_real": em.get(Y, np.nan) if Y <= 2021 else np.nan}
        row["x_real"] = row["em_real"] - avg5
        for hz in (1, 4):
            p2 = days.searchsorted(obs + pd.Timedelta(days=7 * hz))
            ok = p2 < len(days) and days[p2] <= exp and not pd.isna(h.get(days[p2])) and not pd.isna(j.get(days[p2]))
            row[f"dS_{hz}w"] = (h[days[p2]] - j[days[p2]]) - row["spread"] if ok else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    assert df.obs_date.max() <= CUTOFF
    return df


if __name__ == "__main__":
    d = build()
    print(d.groupby("winter").agg(n=("spread", "size"), x_real=("x_real", "first"),
                                  S=("spread", "mean"), xinfo_na=("x_info", lambda v: v.isna().sum())))
