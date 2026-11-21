"""Weekly winter panel for the regimes_weather study.

Every input file is cut at 2021-12-31 as soon as it is loaded (no holdout data).

One row per winter Friday F_t (Nov-Feb). The price observation is the settlement on the
first trading day on or after F_t (same rule as src/model3_data.py). The spread is
March minus April of the delivery year Y whose March is the nearest winter's March
(Y = year of F_t + 1 for Nov/Dec, year of F_t for Jan/Feb), as in model3_data.

Timing inside a row (t):
  dS        = S(F_t) - S(F_{t-1}), the weekly spread change.
  surp5     = storage change for the week ending F_t - 7 (reported Thursday F_t - 1, i.e.
              between the two settlements) minus the same-calendar-week change averaged
              over the available prior years (up to 5; storage.csv starts Jan 2010 so the
              early winters use fewer years; >= 2 required).
  surpW     = weather-adjusted storage surprise: the same week's change minus a walk-forward
              OLS fit chg ~ 1 + HDD + HDD^2 + trend on earlier Nov-Mar weeks (>= 20). HDD of the
              report week is observed before the report, so this is closer to what the
              market did not already know.
  a0        = HDD anomaly of the week ending F_t (weather during the price interval)
  a1, a2    = HDD anomaly of the weeks ending F_t+7 and F_t+14 (future weather: a stand-in
              for perfect 1-2 week forecasts; NOT known at t)
  aR        = HDD anomaly of the report week (week ending F_t - 7)
  wx2       = a1 + a2  (primary "forward weather news")
  fwd_h     = S(F_{t+h}) - S(F_t), h = 1..4, only if F_{t+h} settles before March expiry
              and is <= 2021-12-31.
HDD normal = mean over the prior 10 years (>= 5 available) of the same calendar week.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw"
CACHE = ROOT / "data/processed/research_regimes_weather"
CUTOFF = pd.Timestamp("2021-12-31")


def _same_week(s: pd.Series, years: int, min_years: int) -> pd.Series:
    """Mean of s at the same calendar week (nearest date within 3 days) in each of the
    previous `years` years; NaN unless at least `min_years` of them exist."""
    idx = s.index
    out = pd.Series(np.nan, index=idx)
    for t in idx:
        vals = []
        for k in range(1, years + 1):
            target = t - pd.DateOffset(years=k)
            pos = idx.searchsorted(target)
            cands = [p for p in (pos - 1, pos) if 0 <= p < len(idx)]
            if not cands:
                continue
            best = min(cands, key=lambda p: abs(idx[p] - target))
            if abs(idx[best] - target) <= pd.Timedelta(days=3) and not np.isnan(s.iloc[best]):
                vals.append(s.iloc[best])
        if len(vals) >= min_years:
            out[t] = np.mean(vals)
    return out


def load_weekly_fundamentals():
    s = pd.read_csv(RAW / "storage.csv", parse_dates=["week_ending"])
    s = s[s.week_ending <= CUTOFF].set_index("week_ending")["storage_bcf"].astype(float)
    w = pd.read_csv(RAW / "weekly_degree_days_own.csv", parse_dates=["week_ending"])
    w = w[w.week_ending <= CUTOFF].set_index("week_ending")[["hdd", "cdd"]]

    w["hdd_norm"] = _same_week(w["hdd"], 10, 5)
    w["anom"] = w["hdd"] - w["hdd_norm"]

    st = pd.DataFrame({"level": s, "chg": s.diff()})
    st["chg_norm5"] = _same_week(st["chg"], 5, 2)
    st["surp5"] = st["chg"] - st["chg_norm5"]

    # weather-adjusted surprise, walk-forward OLS on all earlier weeks
    # fitted on earlier Nov-Mar weeks only (heating season), with a linear time trend
    # for demand growth (LNG exports, power burn); >= 20 earlier winter weeks required
    j = st.join(w[["hdd", "cdd"]], how="inner").dropna(subset=["chg"])
    j = j[j.index.month.isin([11, 12, 1, 2, 3])]
    yrs = (j.index - pd.Timestamp("2010-01-01")).days / 365.25
    X = np.column_stack([np.ones(len(j)), j.hdd, j.hdd ** 2 / 100, yrs])
    y = j["chg"].values
    pred = np.full(len(j), np.nan)
    for i in range(20, len(j)):
        b, *_ = np.linalg.lstsq(X[:i], y[:i], rcond=None)
        pred[i] = X[i] @ b
    st["surpW"] = pd.Series(y - pred, index=j.index)
    return st, w


def load_spread_daily():
    c = pd.read_csv(RAW / "ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    c = c[(c.date <= CUTOFF) & c.contract_month.isin([3, 4])].dropna(subset=["settle"])
    import sys
    sys.path.insert(0, str(ROOT))
    from src.prices import spread_panel
    sp = spread_panel(c)  # date, year, spread, near, far
    expiry = c[c.contract_month == 3].groupby("contract_year")["date"].max()
    sp["h_last"] = sp["year"].map(expiry)
    return sp


def build(force=False):
    f = CACHE / "panel.csv"
    if f.exists() and not force:
        return pd.read_csv(f, parse_dates=["F", "obs_date", "h_last"])
    st, w = load_weekly_fundamentals()
    sp = load_spread_daily()
    days = np.sort(sp["date"].unique())
    piv = sp.pivot(index="date", columns="year", values="spread")
    apr = sp.pivot(index="date", columns="year", values="far")
    h_last = sp.groupby("year")["h_last"].first()

    def obs_day(F):
        p = days.searchsorted(np.datetime64(F))
        return pd.Timestamp(days[p]) if p < len(days) else None

    rows = []
    fridays = pd.date_range("2010-10-01", CUTOFF, freq="W-FRI")
    for F in fridays:
        if F.month not in (11, 12, 1, 2):
            continue
        Y = F.year + 1 if F.month >= 11 else F.year
        if Y not in piv.columns:
            continue
        hl = h_last[Y]
        d0, dp = obs_day(F), obs_day(F - pd.Timedelta(days=7))
        if d0 is None or d0 >= hl or dp is None:  # skip the expiry day itself
            continue
        S0, Sp = piv.at[d0, Y], piv.at[dp, Y]
        if np.isnan(S0) or np.isnan(Sp) or dp == d0:
            continue
        r = {"F": F, "obs_date": d0, "year": Y, "winter": f"{Y - 1}-{str(Y)[2:]}", "h_last": hl,
             "S": S0, "S_prev": Sp, "dS": S0 - Sp, "april": apr.at[d0, Y],
             "weeks_to_expiry": (hl - d0).days / 7}
        for h in range(1, 5):
            Fh = F + pd.Timedelta(days=7 * h)
            dh = obs_day(Fh) if Fh <= CUTOFF else None
            ok = dh is not None and dh < hl and dh <= CUTOFF and not np.isnan(piv.at[dh, Y])
            r[f"fwd_{h}"] = piv.at[dh, Y] - S0 if ok else np.nan
        rep = F - pd.Timedelta(days=7)
        for k, col in (("surp5", "surp5"), ("surpW", "surpW"), ("storage_chg", "chg"), ("storage_level", "level")):
            r[k] = st[col].get(rep, np.nan)
        for k, wk in (("aR", rep), ("a0", F), ("a1", F + pd.Timedelta(days=7)), ("a2", F + pd.Timedelta(days=14)),
                      ("a3", F + pd.Timedelta(days=21)), ("a4", F + pd.Timedelta(days=28)),
                      ("a5", F + pd.Timedelta(days=35)), ("a6", F + pd.Timedelta(days=42))):
            r[k] = w["anom"].get(wk, np.nan) if wk <= CUTOFF else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    df["wx2"] = df["a1"] + df["a2"]
    assert df.obs_date.max() <= CUTOFF
    df.to_csv(f, index=False)
    return df


if __name__ == "__main__":
    d = build(force=True)
    print(d.groupby("winter").agg(n=("dS", "size"), first=("F", "min"), last=("F", "max"),
                                  sd_dS=("dS", "std"), n_s5=("surp5", "count"), n_wx=("wx2", "count")).to_string())
    print(d[["dS", "surp5", "surpW", "a0", "a1", "a2", "wx2"]].describe().round(3).to_string())
