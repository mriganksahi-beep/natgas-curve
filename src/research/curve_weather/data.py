"""Weekly winter panel for curve_weather: curve moves and the HDD anomalies around them.

Every input file is cut at 2021-12-31 on load (no holdout data).

One row per winter Friday F_t (Nov-Feb), winters 2010-11 to Nov-Dec 2021. Prices are the
settlement on the first trading day on or after F_t (same rule as model3_data /
regimes_weather), so the Thursday storage report is already public.

Instruments (changes between the settles for F_{t-1} and F_t, same contracts at both dates):
  r_front  = 100 * dlog price of the front contract (earliest contract with >= 2 trading days
             left after the F_t settle), in %
  r_second = 100 * dlog price of the next contract, in %
  r_strip  = 100 * dlog of the mean price of the target winter's Dec-Mar contracts that are live
             (>= 2 trading days to expiry) at F_t, in %
  dS       = change in the target winter's March - April spread, $/MMBtu (target winter = the
             nearest March, as in regimes_weather; rows at/after March expiry dropped)
  S_prev   = March - April spread at F_{t-1} (tightness)

Weather (gas-weighted HDD, 17 cities; anomaly = HDD minus the mean of the same calendar week in
the prior 10 years, >= 5 required):
  aR = report week (ending F_t-7), a0 = week ending F_t (during the price interval),
  a1, a2 = weeks ending F_t+7 and F_t+14.  wx2 = a1 + a2 is the target ("next 2 weeks").
  a3..a6 used only as controls in the reversal tests.
Forward changes fwd_<inst>_h (h = 1, 2, 4) use the same contracts as at F_t.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw"
CACHE = ROOT / "data/processed/research_curve_weather"
CACHE.mkdir(parents=True, exist_ok=True)
CUTOFF = pd.Timestamp("2021-12-31")


def same_week(s: pd.Series, years: int, min_years: int) -> pd.Series:
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


def load_weather():
    w = pd.read_csv(RAW / "weekly_degree_days_own.csv", parse_dates=["week_ending"])
    w = w[w.week_ending <= CUTOFF].set_index("week_ending")[["hdd", "cdd"]]
    w["hdd_norm10"] = same_week(w["hdd"], 10, 5)
    w["anom"] = w["hdd"] - w["hdd_norm10"]
    return w


def load_storage():
    s = pd.read_csv(RAW / "storage.csv", parse_dates=["week_ending"])
    return s[s.week_ending <= CUTOFF].set_index("week_ending")["storage_bcf"].astype(float)


def load_curve():
    c = pd.read_csv(RAW / "ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    c = c[c.date <= CUTOFF].dropna(subset=["settle"])
    c["cid"] = c.contract_year * 100 + c.contract_month
    px = c.pivot_table(index="date", columns="cid", values="settle")
    days = px.index
    # expiry = 3rd trading day before the first day of the delivery month (trading calendar
    # from the data; for delivery after the cutoff, business days)
    exp = {}
    for cid in px.columns:
        y, m = divmod(cid, 100)
        first = pd.Timestamp(y, m, 1)
        if first <= CUTOFF:
            pos = days.searchsorted(first)
            exp[cid] = days[pos - 3]
        else:
            exp[cid] = first - pd.offsets.BDay(3)
    return px, pd.Series(exp)


def build(force=False):
    f = CACHE / "panel.csv"
    if f.exists() and not force:
        return pd.read_csv(f, parse_dates=["F", "d0", "dp"])
    w = load_weather()
    st = load_storage()
    px, exp = load_curve()
    days = px.index
    tdi = pd.Series(np.arange(len(days)), index=days)

    def obs_day(F):
        p = days.searchsorted(F)
        return days[p] if p < len(days) else None

    def tdays_left(cid, d):
        e = exp[cid]
        if e > days[-1]:
            return 99
        return tdi[e] - tdi[d]

    rows = []
    for F in pd.date_range("2010-10-29", CUTOFF, freq="W-FRI"):
        if F.month not in (11, 12, 1, 2):
            continue
        d0, dp = obs_day(F), obs_day(F - pd.Timedelta(days=7))
        if d0 is None or dp is None or dp == d0:
            continue
        Y = F.year + 1 if F.month >= 11 else F.year
        r = {"F": F, "d0": d0, "dp": dp, "year": Y, "winter": f"{Y - 1}-{str(Y)[2:]}"}
        fwd_days = {h: (obs_day(F + pd.Timedelta(days=7 * h)) if F + pd.Timedelta(days=7 * h) <= CUTOFF else None)
                    for h in (1, 2, 4)}
        live = [cid for cid in sorted(px.columns) if exp[cid] >= d0 and tdays_left(cid, d0) >= 2]
        front, second = live[0], live[1]

        def lchg(cids, a, b):
            if b is None:
                return np.nan
            pa, pb = px.loc[a, cids], px.loc[b, cids]
            if pa.isna().any() or pb.isna().any():
                return np.nan
            return 100 * (np.log(pb.mean()) - np.log(pa.mean()))

        def ok_until(cids, d):  # all contracts still trade (not expired) at d
            return d is not None and all(exp[c] > d for c in cids)

        strip = [cid for cid in live if cid in (Y * 100 + 3, Y * 100 + 2, Y * 100 + 1, (Y - 1) * 100 + 12)]
        for name, cids in (("front", [front]), ("second", [second]), ("strip", strip)):
            r[f"r_{name}"] = lchg(cids, dp, d0) if cids else np.nan
            for h, dh in fwd_days.items():
                r[f"fwd_{name}_{h}"] = lchg(cids, d0, dh) if (cids and ok_until(cids, dh)) else np.nan
        r["front_cid"], r["second_cid"] = front, second
        r["front_px_prev"] = px.at[dp, front]  # used only for $ conversions
        # March - April spread of the target winter
        h_, j_ = Y * 100 + 3, Y * 100 + 4
        if h_ in px.columns and j_ in px.columns and exp[h_] > d0:
            S0 = px.at[d0, h_] - px.at[d0, j_]
            Sp = px.at[dp, h_] - px.at[dp, j_]
            r["S"], r["S_prev"], r["dS"] = S0, Sp, S0 - Sp
            for h, dh in fwd_days.items():
                r[f"fwd_dS_{h}"] = (px.at[dh, h_] - px.at[dh, j_] - S0) if (dh is not None and exp[h_] > dh) else np.nan
        for k, off in (("aR", -7), ("a0", 0), ("a1", 7), ("a2", 14), ("a3", 21), ("a4", 28), ("a5", 35), ("a6", 42)):
            wk = F + pd.Timedelta(days=off)
            r[k] = w["anom"].get(wk, np.nan) if wk <= CUTOFF else np.nan
            if k in ("a0", "a1"):
                r["hdd_" + k] = w["hdd"].get(wk, np.nan) if wk <= CUTOFF else np.nan
                r["norm10_" + k] = w["hdd_norm10"].get(wk, np.nan) if wk <= CUTOFF else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    df["wx2"] = df["a1"] + df["a2"]
    df["pers"] = 2 * df["a0"]  # naive persistence forecast of wx2
    assert df.d0.max() <= CUTOFF
    df.to_csv(f, index=False)
    return df


if __name__ == "__main__":
    d = build(force=True)
    print(d.groupby("winter").agg(n=("F", "size"), nS=("dS", "count"), nwx=("wx2", "count"),
                                  sd_front=("r_front", "std"), sd_dS=("dS", "std"), sd_wx=("wx2", "std")).round(2).to_string())
    print(d[["r_front", "r_second", "r_strip", "dS", "a0", "a1", "a2", "wx2"]].corr().round(2).to_string())
