"""Weekly winter panel for the fear-vs-physics study. No data dated 2022-01-01 or later is read.

Definitions
-----------
Delivery year Y: March (H) and April (J) of year Y; winter = Nov Y-1 .. Feb Y.
S = H - J ($/MMBtu). One spread = 10,000 MMBtu per leg, so $1 of spread = $10,000.
Origin F: an EIA storage week (Friday) with F in 1 Nov .. 14 Feb. Storage for F is public
    Thursday F+6, so the decision is the settle of the first trading day >= F+7 ("obs").
Outcomes: S_exp = H-J on the H last trading day (Y <= 2021 only);
    S_4w = spread at the first settle >= obs+28 d, or at expiry if sooner; NaN past 2021-12-31.

Fear (all from prices up to and including the obs settle):
    rv2, rv4, rv8: s.d. of daily changes of S over the trailing 10 / 20 / 40 trading days ($/day)
    maxrise: largest 5-day rise of S within the trailing 20 trading days ($)
    runup: S minus its trailing 20-day minimum ($)
    fvol: front-month 20-day realised vol of log returns, annualised (front = nearest contract
          with >= 2 trading days left; returns are computed within a contract, so no roll jumps)

Physics (storage known at F, weather history before F only):
    dev_now: storage at F minus the mean of the same calendar week in prior years (1..5 years;
        storage starts Jan 2010 so Nov-Dec 2010 has none -> NaN; Jan 2011 has 1, ... 2015+ has 5)
    avg5[Y]: mean end-March storage over the prior 1..5 years (2011 uses 2010 only).
    Weather-analog projection (primary physics): fit weekly change = a + b*HDD on Nov-Mar weeks of the
        trailing 3 years known at F; for each complete prior winter A (2004-05 .. Y-1 with HDD for every
        remaining week), x_A = storage_F + sum_k (a + b*HDD_{A,k}) - avg5[Y], weeks aligned by distance
        to the end-March week. P_tail = mean_A Phi((THR - x_A) / (sigma_resid * sqrt(n_rem))),
        THR = -500 Bcf (fixed a priori: the payout curve of widowmaker_insurance starts to bite there).
        x_an = mean_A x_A (projected end-March deviation).
    Storage-draw analog (secondary): x'_A = storage_F + (prior winter A's draw over the same remaining
        weeks) - avg5[Y]; available for A >= 2011 (Nov origins) since storage starts Jan 2010.
    Model 2 (Nov 2016+): P(x < THR) from p1..p99 re-centred so the median = 5-year-path benchmark
        (bias-robust, as widowmaker_insurance), and raw.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[3]
CUTOFF = pd.Timestamp("2021-12-31")
CACHE = ROOT / "data/processed/research_fear_vs_physics"
CACHE.mkdir(parents=True, exist_ok=True)
MULT = 10_000
COST_TICKS = 4
ROUND_TRIP = 2 * 2 * COST_TICKS * 10.0      # 2 contracts x 2 sides x 4 ticks x $10 = $160 per spread
THR = -500.0


def _cut(df, col):
    df = df[df[col] <= CUTOFF].copy()
    assert df[col].max() <= CUTOFF
    return df


def load_storage():
    s = _cut(pd.read_csv(ROOT / "data/raw/storage.csv", parse_dates=["week_ending"]), "week_ending")
    return s.set_index("week_ending")["storage_bcf"].sort_index()


def load_hdd():
    h = _cut(pd.read_csv(ROOT / "data/raw/weekly_degree_days_own.csv", parse_dates=["week_ending"]), "week_ending")
    return h.set_index("week_ending")["hdd"].sort_index()


def end_march_week(Y):
    for d in range(25, 32):
        t = pd.Timestamp(Y, 3, d)
        if t.weekday() == 4:
            return t


def avg5_table():
    s = load_storage()
    em = {Y: s.get(end_march_week(Y), np.nan) for Y in range(2010, 2022)}
    rows = []
    for Y in range(2011, 2023):
        prior = [em[Y - k] for k in range(1, 6) if (Y - k) in em and np.isfinite(em[Y - k])]
        rows.append({"Y": Y, "avg5": np.mean(prior), "n_prior": len(prior), "end_march": em.get(Y, np.nan)})
    t = pd.DataFrame(rows).set_index("Y")
    t["x"] = t.end_march - t.avg5
    return t


def load_curve():
    c = pd.read_csv(ROOT / "data/raw/ng_curve_daily.csv", parse_dates=["date"],
                    usecols=["date", "contract_year", "contract_month", "settle"])
    c = _cut(c.dropna(subset=["settle"]), "date")
    return c.pivot_table(index="date", columns=["contract_year", "contract_month"], values="settle").sort_index()


def spreads(px):
    out = {}
    for Y in range(2011, 2023):
        if (Y, 3) not in px.columns or (Y, 4) not in px.columns:
            continue
        sp = (px[(Y, 3)] - px[(Y, 4)]).dropna()
        out[Y] = sp[sp.index >= pd.Timestamp(Y - 1, 8, 1)]
    return out


def front_vol(px):
    """20-day realised vol (annualised) of the front contract's own daily log returns."""
    lr = np.log(px).diff()
    last = px.apply(lambda c: c.last_valid_index())
    dates = px.index
    pos = pd.Series(np.arange(len(dates)), index=dates)
    # front on day d: earliest-expiring contract whose last day is >= 2 trading days after d
    exp_pos = {k: pos[v] for k, v in last.items() if v < CUTOFF}  # contracts that expired in sample
    keys = sorted(px.columns)
    r = pd.Series(np.nan, index=dates)
    for i, d in enumerate(dates):
        for k in keys:
            ep = exp_pos.get(k, 10 ** 9)
            if ep - i >= 2 and np.isfinite(px.iat[i, px.columns.get_loc(k)]):
                r.iat[i] = lr.iat[i, px.columns.get_loc(k)]
                break
    return r.rolling(20).std() * np.sqrt(252)


def analog_projection(F, Y, s, hdd, d_s, a5):
    T = end_march_week(Y)
    n_rem = int(round((T - F).days / 7))
    win = d_s[(d_s.index <= F) & (d_s.index > F - pd.Timedelta(days=3 * 365)) & d_s.index.month.isin([11, 12, 1, 2, 3])]
    j = pd.concat([win, hdd], axis=1, join="inner").dropna()
    j.columns = ["ds", "hdd"]
    b, a = np.polyfit(j.hdd, j.ds, 1)
    sig = np.std(j.ds - (a + b * j.hdd), ddof=2)
    xs, xs_st = [], []
    for A in range(2005, Y):
        TA = end_march_week(A)
        if TA >= F:
            continue
        wk = [TA - pd.Timedelta(days=7 * (n_rem - k)) for k in range(1, n_rem + 1)]
        if all(w in hdd.index for w in wk):
            xs.append(s[F] + np.sum(a + b * hdd[wk].values) - a5)
        st0 = TA - pd.Timedelta(days=7 * n_rem)
        if st0 in s.index and TA in s.index:
            xs_st.append(s[F] + (s[TA] - s[st0]) - a5)
    xs = np.array(xs)
    sd = sig * np.sqrt(max(n_rem, 1))
    return {"n_rem": n_rem, "beta_hdd": b, "alpha": a, "sig": sig, "n_an": len(xs),
            "x_an": xs.mean(), "x_an_min": xs.min(), "p_tail": norm.cdf((THR - xs) / sd).mean(),
            "p_tail_raw": (xs < THR).mean(),
            "p_tail_300": norm.cdf((-300 - xs) / sd).mean(), "p_tail_800": norm.cdf((-800 - xs) / sd).mean(),
            "n_st": len(xs_st), "x_st": np.mean(xs_st) if xs_st else np.nan,
            "p_tail_st": np.mean(np.array(xs_st) < THR) if len(xs_st) >= 2 else np.nan}


def calendar_dev(F, s):
    prior = []
    for k in range(1, 6):
        t = F - pd.DateOffset(years=k)
        near = s[(s.index >= t - pd.Timedelta(days=3)) & (s.index <= t + pd.Timedelta(days=3))]
        if len(near):
            prior.append(near.iloc[0])
    return (s[F] - np.mean(prior), len(prior)) if prior else (np.nan, 0)


def load_m2(a5):
    m = _cut(pd.read_csv(ROOT / "data/processed/shared/model2_winter_percentiles_dev.csv",
                         parse_dates=["origin", "target"]), "origin")
    P = m[[f"p{k}" for k in range(1, 100)]].values
    Y = m.target.dt.year.values
    av = a5.avg5.reindex(Y).values[:, None]
    rec = P - m[["p50"]].values + m[["benchmark"]].values
    m["p_tail_m2"] = ((rec - av) < THR).mean(1)
    m["p_tail_m2raw"] = ((P - av) < THR).mean(1)
    m["x_m2"] = m.benchmark - av[:, 0]
    return m.set_index("origin")[["p_tail_m2", "p_tail_m2raw", "x_m2"]]


def build_panel(refresh=False):
    f = CACHE / "panel.pkl"
    if f.exists() and not refresh:
        return pd.read_pickle(f)
    px = load_curve()
    sp = spreads(px)
    fv = front_vol(px)
    s, hdd = load_storage(), load_hdd()
    d_s = s.diff().dropna()
    a5 = avg5_table()
    m2 = load_m2(a5)
    rows = []
    for F in s.index:
        if F.month not in (11, 12, 1, 2) or (F.month == 2 and F.day > 14):
            continue
        Y = F.year + 1 if F.month >= 11 else F.year
        if Y not in sp:
            continue
        S = sp[Y]
        ltd = S.index[-1] if Y <= 2021 else pd.NaT
        pos = S.index.searchsorted(F + pd.Timedelta(days=7))
        if pos >= len(S):
            continue
        obs = S.index[pos]
        if obs > CUTOFF or (pd.notna(ltd) and obs >= ltd):
            continue
        dS = S.diff()
        h = S.iloc[: pos + 1]
        dh = dS.iloc[: pos + 1]
        r = {"origin": F, "obs": obs, "Y": Y, "winter": f"{Y - 1}-{str(Y)[2:]}", "S": S.iloc[pos],
             "rv2": dh.iloc[-10:].std(), "rv4": dh.iloc[-20:].std(), "rv8": dh.iloc[-40:].std(),
             "maxrise": (h - h.shift(5)).iloc[-20:].max(), "runup": h.iloc[-1] - h.iloc[-20:].min(),
             "fvol": fv.get(obs, np.nan), "storage": s[F]}
        ltd_nom = ltd if pd.notna(ltd) else pd.Timestamp(Y, 3, 1) - pd.offsets.BDay(3)
        r["wte"] = (ltd_nom - obs).days / 7
        r["dev_now"], r["n_dev"] = calendar_dev(F, s)
        r["avg5"], r["n_avg5"] = a5.avg5[Y], a5.n_prior[Y]
        r.update(analog_projection(F, Y, s, hdd, d_s, a5.avg5[Y]))
        if F in m2.index:
            r.update(m2.loc[F].to_dict())
        r["S_exp"] = S.iloc[-1] if pd.notna(ltd) else np.nan
        p4 = S.index.searchsorted(obs + pd.Timedelta(days=28))
        if p4 < len(S) and S.index[p4] <= CUTOFF:
            r["S_4w"], r["date_4w"] = S.iloc[p4], S.index[p4]
        elif pd.notna(ltd):
            r["S_4w"], r["date_4w"] = S.iloc[-1], S.index[-1]
        else:
            r["S_4w"], r["date_4w"] = np.nan, pd.NaT
        # worst mark-to-market rise for a short held to expiry (path risk), in $ of spread
        fut = S.iloc[pos:] if pd.notna(ltd) else S.iloc[pos:]
        r["max_adverse_exp"] = fut.max() - S.iloc[pos]
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("obs").reset_index(drop=True)
    df["dS_exp"] = df.S_exp - df.S
    df["dS_4w"] = df.S_4w - df.S
    assert df.obs.max() <= CUTOFF and df.origin.max() <= CUTOFF
    assert (df.date_4w.dropna() <= CUTOFF).all()
    df.to_pickle(f)
    # daily spread paths (for figures / 2014 table), cached separately
    pd.to_pickle({Y: v for Y, v in sp.items()}, CACHE / "spreads.pkl")
    pd.to_pickle(fv, CACHE / "fvol.pkl")
    return df


if __name__ == "__main__":
    import time
    t0 = time.time()
    p = build_panel(refresh=True)
    print(p.shape, time.time() - t0)
    pd.set_option("display.width", 250)
    print(p.groupby("winter")[["S", "rv4", "fvol", "dev_now", "x_an", "p_tail", "p_tail_st", "p_tail_m2", "dS_exp", "n_an"]].mean().round(3))
