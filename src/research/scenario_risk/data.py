"""Weekly data for scenario_risk. Everything is cut at 2021-12-31 on load.

Generic weekly March-April series
  For each Friday d, the "front winter" year Y(d) is the first March contract whose last trading
  day (3 business days before 1 March) is after d. Price on d = last settle on or before d.
  dS_h(d) = S_Y(d+7h) - S_Y(d), same contract pair, only if d+7h <= expiry and <= CUTOFF.
  Loss of a short 1-spread position over h weeks = dS_h (in $/MMBtu; x 10,000 for $/contract).

Storage / weather inputs (all public by d):
  Storage for week ending F is released Thursday F+6, so on Friday d the latest storage week is d-7.
  avg5, x_info as in src/research/implied_storage/data.py (reviewed, walk-forward clean).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CUTOFF = pd.Timestamp("2021-12-31")
CACHE = ROOT / "data/processed/research_scenario_risk"
CACHE.mkdir(parents=True, exist_ok=True)


def march_expiry(Y):
    return pd.Timestamp(Y, 3, 1) - pd.offsets.BDay(3)


def load_marapr():
    c = pd.read_csv(ROOT / "data/raw/ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    c = c[(c.date <= CUTOFF) & c.contract_month.isin([3, 4])]
    px = c.dropna(subset=["settle"]).pivot_table(index="date", columns=["contract_year", "contract_month"],
                                                 values="settle")
    assert px.index.max() <= CUTOFF
    return px


def load_storage():
    s = pd.read_csv(ROOT / "data/raw/storage.csv", parse_dates=["week_ending"])
    s = s[s.week_ending <= CUTOFF]
    return s.set_index("week_ending")["storage_bcf"]


def load_hdd():
    w = pd.read_csv(ROOT / "data/raw/weekly_degree_days_own.csv", parse_dates=["week_ending"])
    w = w[w.week_ending <= CUTOFF]
    return w.set_index("week_ending")["hdd"]


def load_model2():
    m = pd.read_csv(ROOT / "data/processed/shared/model2_winter_percentiles_dev.csv",
                    parse_dates=["origin", "target"])
    m = m[m.origin <= CUTOFF].copy()
    # 'actual' is ex-post; never used as an input
    return m.drop(columns=["actual"]).set_index("origin")


def _px_on(px, Y, d):
    """March-April spread and April price of year Y at last settle <= d (within 4 days)."""
    if (Y, 3) not in px.columns or (Y, 4) not in px.columns:
        return np.nan, np.nan
    sub = px.loc[:d, [(Y, 3), (Y, 4)]].dropna()
    if sub.empty or (d - sub.index[-1]).days > 4:
        return np.nan, np.nan
    r = sub.iloc[-1]
    return r[(Y, 3)] - r[(Y, 4)], r[(Y, 4)]


def weekly_panel():
    """One row per Friday from 2010-06-11 to CUTOFF."""
    f = CACHE / "weekly_panel.pkl"
    if f.exists():
        return pd.read_pickle(f)
    px = load_marapr()
    fridays = pd.date_range("2010-06-11", CUTOFF, freq="W-FRI")
    rows = []
    for d in fridays:
        Y = d.year if d < march_expiry(d.year) else d.year + 1
        S, A = _px_on(px, Y, d)
        row = {"date": d, "Y": Y, "S": S, "april": A, "expiry": march_expiry(Y),
               "weeks_to_expiry": (march_expiry(Y) - d).days / 7}
        for h in (1, 4):
            e = d + pd.Timedelta(days=7 * h)
            if e <= march_expiry(Y) and e <= CUTOFF:
                S2, _ = _px_on(px, Y, e)
                row[f"dS_{h}"] = S2 - S
            else:
                row[f"dS_{h}"] = np.nan
        # backward 1-week change, same pair (for the trailing-window models)
        S0, _ = _px_on(px, Y, d - pd.Timedelta(days=7))
        row["dS_back1"] = S - S0
        S4, _ = _px_on(px, Y, d - pd.Timedelta(days=28))
        row["dS_back4"] = S - S4
        rows.append(row)
    df = pd.DataFrame(rows).set_index("date")
    df.to_pickle(f)
    return df


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


def add_storage_state(df):
    """x_info on Friday d uses storage for week d-7 (released Thursday d-1)."""
    s = load_storage()
    em = end_march(s)
    xi, av = [], []
    for d, r in df.iterrows():
        t = d - pd.Timedelta(days=7)
        Y = r.Y
        prior = [em[Y - k] for k in range(1, 6) if (Y - k) in em.index and Y - k < Y]
        # end-March of year Y-k is known on d only if it is before d
        prior = [em[Y - k] for k in range(1, 6) if (Y - k) in em.index
                 and pd.Timestamp(Y - k, 4, 10) < d]
        st = _nearest(s, t, tol=1)
        if len(prior) < 2 or st is None:
            xi.append(np.nan); av.append(np.nan); continue
        a5 = float(np.mean(prior))
        draws = []
        for k in range(1, 6):
            if (Y - k) in em.index and pd.Timestamp(Y - k, 4, 10) < d:
                p = _nearest(s, t - pd.DateOffset(years=k))
                if p is not None:
                    draws.append(em[Y - k] - p)
        xi.append(st + np.mean(draws) - a5 if len(draws) >= 2 else np.nan)
        av.append(a5)
    df = df.copy()
    df["x_info"], df["avg5"] = xi, av
    return df


def winter_label(d):
    y = d.year + 1 if d.month >= 7 else d.year
    return f"{y - 1}-{str(y)[2:]}"


def eval_rows(df, h):
    """Forecast Fridays: 1 Nov - 15 Feb, target window inside the contract life and <= CUTOFF."""
    m = df.index.month
    ok = ((m >= 11) | (m == 1) | ((m == 2) & (df.index.day <= 15)))
    ok &= df[f"dS_{h}"].notna() & df.S.notna()
    out = df[ok].copy()
    out["winter"] = [winter_label(d) for d in out.index]
    return out
