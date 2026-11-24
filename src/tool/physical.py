"""Physical ("real-world") as-of distribution of end-March storage deviation x.

    from src.tool.physical import physical_dist
    d = physical_dist("2026-11-24")
    d["quantiles"][0.5], d["prob_below"](-400), d["samples"]   # x in Bcf

Target
------
Y = the March of the winter that contains the as-of date (Aug-Dec -> next year).
T = last Friday of March Y (the last EIA storage week of March, as Model 2).
x = storage(T) - baseline(Y), where baseline(Y) is the mean end-March level of the five prior
years Y-1..Y-5 (weekly last-March-week level from 2010; for earlier years the EIA monthly
end-March working gas, NG.N5020US2.M, plus the mean weekly-minus-monthly offset over years with
both, about -20 Bcf). Pass `baseline=` to use another definition (e.g. the storage_map agent's);
x then shifts by a constant.

Model (default "analog", variant an_f_tr_mu chosen on winters 2011-12..2015-16;
benchmarked in reports/tool/physical_benchmark.md; defaults: prior="trend", balance="fall",
mundlak=True)
-----------------------------------------------------------------------------
Intuition: end-March storage = storage we already know + the draws still to come; draws are
weather (HDD) times a slope, plus a winter-specific "balance" (production, LNG exports, power
burn) that drifts from winter to winter, plus week-to-week noise.

1. Known: the last reported storage week F0 (week ending Friday F0 is public Thursday F0+6),
   realised gas-weighted HDD (17 cities) for every day F0+1 .. as_of-1.
2. Days as_of .. +15: the latest GEFS 00Z run (init <= as_of) ensemble-mean HDD; days without
   realised weather before that run's init (live data lag) use that day's own lead-0 forecast.
   Each scenario adds a past forecast-error path: realised - forecast, for the run initialised on
   the same calendar date k years earlier, same leads (keeps bias and lead-to-lead correlation).
3. Later days: the realised HDD of the same calendar day k years earlier (analog year k,
   2000 .. Y-1), optionally scaled by a linear trend in winter HDD (option `hdd_trend`).
4. Weekly draw regression (EIA weeks, Nov-Mar, all weeks known at as_of):
       chg_w = a_winter + b1 HDD_w + b2 HDD_w^2/100 + c holiday_w + d1 sin + d2 cos + e_w
   with one intercept a_j per winter (fixed effects). The current winter's intercept gets a
   prior from past winters (mean of the last 3, or a linear trend: option `prior`) with variance
   tau^2 (shrunk toward tau0 = 5 Bcf/week with 3 pseudo-winters), updated with the current
   winter's weeks so far (normal-normal, using the long-run residual variance).
5. Scenarios = analog years x residual paths. Each residual path draws an intercept from its
   posterior and 4-week blocks of past within-winter residuals.

Information rule: every input is filtered to what was public at the as-of date's settle, so the
same call works on 2011 dates, locked 2022-26 dates and today, without edits.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.model1 import holiday_in_week

ROOT = Path(__file__).resolve().parents[2]
RELEASE_LAG = 6          # days from week ending (Fri) to EIA release (Thu)
WINTER = (11, 12, 1, 2, 3)
QS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99)
TAU0, TAU0_WEIGHT = 5.0, 3.0
DEFAULTS = dict(prior="trend", balance="fall", mundlak=True, hdd_trend=False, use_forecast=True, n_resid=200, block=4)


# ----------------------------------------------------------------------------- data (cached, uncut)
@lru_cache(maxsize=1)
def _weekly_storage() -> pd.Series:
    s = pd.read_csv(ROOT / "data/raw/storage.csv", parse_dates=["week_ending"])
    return s.set_index("week_ending")["storage_bcf"].astype(float).sort_index()


@lru_cache(maxsize=1)
def _monthly_storage() -> pd.Series:
    m = pd.read_csv(ROOT / "data/raw/eia/storage_monthly_series_bulk.csv")
    m = m[m.series_id == "NG.N5020US2.M"]
    idx = pd.to_datetime(m.month.astype(str), format="%Y%m")
    return pd.Series(m.value.values / 1000.0, index=idx).sort_index()  # Bcf, month start label


@lru_cache(maxsize=1)
def _daily_hdd() -> pd.Series:
    from src.v4.forecasts import realised_hdd
    return realised_hdd().sort_index()


@lru_cache(maxsize=1)
def _gefs() -> pd.DataFrame:
    """init_date x lead (0..15) table of gas-weighted ensemble-mean HDD."""
    from src.v4.forecasts import gefs_hdd
    return gefs_hdd().pivot_table(index="init_date", columns="lead", values="hdd_ens").sort_index()


@lru_cache(maxsize=1)
def _weekly_cdd() -> pd.Series:
    from src.fetch import BASE_F, CITIES
    d = pd.read_csv(ROOT / "data/weather/daily_temps_2000.csv", parse_dates=["date"])
    n = d.groupby("date").city.transform("size")
    d = d[(n == len(CITIES)) & d.temp_f.notna()]
    cw = pd.Series({c: v[3] for c, v in CITIES.items()})
    c = ((d.temp_f - BASE_F).clip(lower=0) * d.city.map(cw)).groupby(d.date).sum() / cw.sum()
    wk = c.resample("W-FRI").agg(["sum", "count"])
    return wk.loc[wk["count"] == 7, "sum"]


@lru_cache(maxsize=1)
def _weekly_hdd() -> pd.Series:
    r = _daily_hdd()
    wk = r.resample("W-FRI").agg(["sum", "count"])
    return wk.loc[wk["count"] == 7, "sum"]


# ----------------------------------------------------------------------------- calendar / target
def winter_of(d) -> int:
    d = pd.Timestamp(d)
    return d.year + 1 if d.month >= 8 else d.year


def target_week(Y: int) -> pd.Timestamp:
    d = pd.Timestamp(Y, 3, 31)
    while d.weekday() != 4:
        d -= pd.Timedelta(days=1)
    return d


def storage_asof(as_of) -> pd.Series:
    as_of = pd.Timestamp(as_of)
    s = _weekly_storage()
    return s[s.index + pd.Timedelta(days=RELEASE_LAG) <= as_of]


def _monthly_asof(as_of) -> pd.Series:
    m = _monthly_storage()
    return m[m.index + pd.DateOffset(months=2) <= pd.Timestamp(as_of)]   # monthly data lag ~2 months


def end_march_level(Y: int, as_of):
    """End-March level of year Y as public at as_of (weekly from 2010, else monthly + offset)."""
    s = storage_asof(as_of)
    T = target_week(Y)
    if T in s.index:
        return float(s[T])
    m = _monthly_asof(as_of)
    key = pd.Timestamp(Y, 3, 1)
    if key not in m.index:
        return np.nan
    off = [s[target_week(y)] - m[pd.Timestamp(y, 3, 1)] for y in range(2010, Y + 30)
           if target_week(y) in s.index and pd.Timestamp(y, 3, 1) in m.index]
    return float(m[key] + (np.mean(off) if off else 0.0))


def baseline(Y: int, as_of) -> float:
    v = [end_march_level(Y - k, as_of) for k in range(1, 6)]
    return float(np.mean(v)) if np.all(np.isfinite(v)) else np.nan


def x_realised(Y: int, as_of):
    """Realised x for winter Y, or NaN if end-March Y storage is not public at as_of."""
    s = storage_asof(as_of)
    T = target_week(Y)
    return float(s[T] - baseline(Y, as_of)) if T in s.index else np.nan


def bench5(as_of) -> float:
    """5-year path: last known level + mean change to end-March over the prior <=5 years (as v5)."""
    s = storage_asof(as_of)
    F0, Y = s.index[-1], winter_of(as_of)
    T = target_week(Y)
    rem = []
    for k in range(1, 6):
        a0, a1 = F0 - pd.DateOffset(years=k), T - pd.DateOffset(years=k)
        i0, i1 = s.index.searchsorted(a0), s.index.searchsorted(a1)
        if i0 < len(s) and i1 < len(s) and abs((s.index[i0] - a0).days) <= 3 and abs((s.index[i1] - a1).days) <= 3:
            rem.append(s.iloc[i1] - s.iloc[i0])
    return float(s[F0] + np.mean(rem)) if rem else np.nan


# ----------------------------------------------------------------------------- weather scenarios
def _shift_years(idx: pd.DatetimeIndex, k: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([d - pd.DateOffset(years=k) for d in idx])


def weather_paths(as_of, F0, T, use_forecast=True, hdd_trend=False, weather_asof=None):
    """Daily HDD scenarios for days F0+1..T: array (n_analog, n_days) and metadata.
    weather_asof (ablation only): treat realised weather as known only before this date."""
    as_of = pd.Timestamp(as_of)
    r_all = _daily_hdd()
    cut = as_of if weather_asof is None else min(as_of, pd.Timestamp(weather_asof))
    r = r_all[r_all.index < cut]
    days = pd.date_range(F0 + pd.Timedelta(days=1), T)
    known = days.isin(r.index)
    base = np.where(known, r.reindex(days).values, np.nan)

    # forecast assignment: (init, lead) per unknown day
    init_of = np.full(len(days), np.datetime64("NaT"), dtype="datetime64[ns]")
    lead_of = np.full(len(days), -1)
    fc_init = None
    if use_forecast:
        g = _gefs()
        g = g[g.index <= as_of]
        if len(g) and (as_of - g.index[-1]).days <= 3:
            fc_init = g.index[-1]
            for i, u in enumerate(days):
                if known[i]:
                    continue
                if fc_init <= u <= fc_init + pd.Timedelta(days=15):
                    init_of[i], lead_of[i] = fc_init, (u - fc_init).days
                elif u < fc_init and u in g.index:
                    init_of[i], lead_of[i] = u, 0
            fmask = lead_of >= 0
            vals = [g.at[pd.Timestamp(init_of[i]), lead_of[i]] for i in np.flatnonzero(fmask)]
            base[fmask] = vals
    fmask = lead_of >= 0
    amask = ~known & ~fmask

    # linear trend in Nov-Mar HDD totals (complete winters before as_of)
    scale = {}
    if hdd_trend:
        tot = {}
        for y in range(2001, winter_of(as_of)):
            w = r[(r.index >= pd.Timestamp(y - 1, 11, 1)) & (r.index <= pd.Timestamp(y, 3, 31))]
            if len(w) >= 150:
                tot[y] = w.sum()
        ys = np.array(sorted(tot))
        coef = np.polyfit(ys, [tot[y] for y in ys], 1) if len(ys) >= 5 else None

    g = _gefs() if use_forecast else None
    paths, years = [], []
    Y = winter_of(as_of)
    for k in range(1, 60):
        dk = _shift_years(days, k)
        if dk[0] < r_all.index[0]:
            break
        p = base.copy()
        if amask.any():
            av = r.reindex(dk[amask]).values
            if np.isnan(av).any():
                continue
            if hdd_trend and coef is not None:
                av = av * np.polyval(coef, Y) / np.polyval(coef, Y - k)
            p[amask] = av
        if fmask.any():
            ik = _shift_years(pd.DatetimeIndex(init_of[fmask]), k)
            gk = g.reindex(ik)
            if gk.isna().all(axis=1).any():
                continue
            fk = gk.values[np.arange(len(ik)), lead_of[fmask]]
            rk = r.reindex(dk[fmask]).values
            err = rk - fk
            if np.isnan(err).any():
                continue
            p[fmask] = p[fmask] + err
        paths.append(np.maximum(p, 0.0))
        years.append(Y - k)
    meta = {"forecast_init": fc_init, "n_known_days": int(known.sum()), "n_forecast_days": int(fmask.sum()),
            "n_analog_days": int(amask.sum()), "analog_years": years}
    return np.array(paths), days, meta


# ----------------------------------------------------------------------------- draw regression
def _design(weeks: pd.DatetimeIndex, hdd):
    doy = weeks.dayofyear.values
    hol = np.array([int(holiday_in_week(t)) for t in weeks], float)
    return hdd, np.column_stack([hol, np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)])


def _prior(a: pd.Series, Y: int, how: str) -> float:
    """Prior mean of winter Y's intercept from past winters' intercepts a (index = winter)."""
    if how == "trend" and len(a) >= 3:
        return float(np.polyval(np.polyfit(a.index.values.astype(float), a.values, 1), Y))
    return float(a.iloc[-3:].mean())


def fit_draws(as_of, prior="last3", balance="winter", mundlak=False):
    """Weekly draw regression on all weeks public at as_of (see module docstring, step 4).

    balance="fall" also fits Sep-Oct weeks (own constant, seasonal shape and a CDD term) that
    share the season's intercept, so the current season's balance is already partly known in Nov.
    """
    s = storage_asof(as_of)
    chg = s.diff()
    ok = (s.index.to_series().diff().dt.days == 7).values
    df = pd.DataFrame({"chg": chg[ok]}).join(_weekly_hdd().rename("hdd"), how="inner")
    df = df.join(_weekly_cdd().rename("cdd"), how="left").dropna()
    months = WINTER + ((9, 10) if balance == "fall" else ())
    df = df[df.index.month.isin(months)]
    df["wy"] = [d.year + 1 if d.month >= 8 else d.year for d in df.index]
    win = df.index.month.isin(WINTER)
    Y = winter_of(as_of)
    hdd, Z = _design(df.index, df.hdd.values)
    cols = [hdd * win, hdd ** 2 / 100 * win, Z[:, 0], Z[:, 1] * win, Z[:, 2] * win]
    if balance == "fall":
        fl = (~win).astype(float)
        cols += [fl, fl * Z[:, 1], fl * Z[:, 2], fl * df.cdd.values, fl * hdd]
    wys = sorted(df.wy.unique())
    D = (df.wy.values[:, None] == np.array(wys)[None, :]).astype(float)
    X = np.column_stack(cols + [D])
    nb = len(cols)
    beta, *_ = np.linalg.lstsq(X, df.chg.values, rcond=None)
    e = df.chg.values - X @ beta
    a = pd.Series(beta[nb:], index=wys)
    n_w = df[win].groupby("wy").size()
    past = [j for j in wys if j < Y and n_w.get(j, 0) >= 8]
    # between-winter weather effect (option mundlak): a_j = c0 + c1 j + gamma * h_j, where h_j is
    # winter j's mean weekly HDD anomaly; cold winters draw less than the weekly slope implies
    # (price-driven supply/demand response). Needs >= 5 past winters, else gamma = 0.
    norm = weekly_hdd_normal(as_of)
    hw = df.hdd[win] - norm.reindex(_woy(df.index[win])).values
    h = hw.groupby(df.wy[win]).mean()
    gamma = 0.0
    if mundlak and len(past) >= 5:
        Xg = np.column_stack([np.ones(len(past)), np.array(past, float), h[past].values])
        gamma = max(0.0, float(np.linalg.lstsq(Xg, a[past].values, rcond=None)[0][2]))
    astar = a - gamma * h.reindex(a.index).fillna(0.0)
    pr = _prior(astar[past], Y, prior)
    errs = [astar[past[i]] - _prior(astar[past[:i]], past[i], prior) for i in range(1, len(past))]
    tau2 = (TAU0_WEIGHT * TAU0 ** 2 + np.sum(np.square(errs))) / (TAU0_WEIGHT + len(errs))
    # residual pool: Nov-Mar weeks of complete past winters; long-run variance from 4-week sums
    mask_p = np.isin(df.wy.values, past) & win
    ep = e[mask_p]
    dof = len(e) / max(len(e) - X.shape[1], 1)
    ep = (ep - ep.mean()) * np.sqrt(dof)
    wy_p = df.wy.values[mask_p]
    sums = [ep[np.flatnonzero(wy_p == j)][i:i + 4].sum() for j in past
            for i in range(0, (wy_p == j).sum() - 3, 4)]
    s2_lr = max(np.var(sums) / 4, np.var(ep)) if len(sums) > 3 else np.var(ep)
    starts = [i for i in range(len(ep) - 3) if wy_p[i] == wy_p[i + 3]]
    n_cur = int((df.wy == Y).sum())
    if n_cur:
        prec = 1 / tau2 + n_cur / s2_lr
        post_m, post_v = (pr / tau2 + n_cur * astar[Y] / s2_lr) / prec, 1 / prec
    else:
        post_m, post_v = pr, tau2
    return {"b": beta[:5], "a_prior": pr, "tau2": tau2, "a_post": post_m, "a_post_var": post_v,
            "n_cur": n_cur, "resid": ep, "starts": np.array(starts), "s2_lr": s2_lr,
            "a_hist": a, "n_train": len(df), "gamma": gamma,
            "h_cur": float(h.get(Y, 0.0)), "n_cur_win": int(n_w.get(Y, 0)), "norm": norm}


def _woy(idx):
    return np.minimum((pd.DatetimeIndex(idx).dayofyear.values - 1) // 7, 51)


def weekly_hdd_normal(as_of) -> pd.Series:
    """Mean weekly HDD by week-of-year over complete weeks before as_of (2000+)."""
    h = _weekly_hdd()
    h = h[h.index < pd.Timestamp(as_of)]
    return h.groupby(_woy(h.index)).mean()


# ----------------------------------------------------------------------------- main entry
def _simulate_levels(as_of, prior="last3", balance="winter", mundlak=False, hdd_trend=False, use_forecast=True, n_resid=200, block=4,
                     seed=None, weather_asof=None):
    as_of = pd.Timestamp(as_of)
    s = storage_asof(as_of)
    F0, Y = s.index[-1], winter_of(as_of)
    T = target_week(Y)
    if T <= F0:
        raise ValueError("as-of date is past the target week")
    A, days, meta = weather_paths(as_of, F0, T, use_forecast=use_forecast, hdd_trend=hdd_trend,
                                  weather_asof=weather_asof)
    n = len(days) // 7
    H = A.reshape(len(A), n, 7).sum(2)                         # (K, n) weekly HDD
    weeks = pd.date_range(F0 + pd.Timedelta(days=7), T, freq="7D")
    fit = fit_draws(as_of, prior=prior, balance=balance, mundlak=mundlak)
    b = fit["b"]
    _, Z = _design(weeks, np.zeros(n))
    cal = Z @ b[2:5]                                            # (n,)
    wx = b[0] * H + b[1] * H ** 2 / 100 + cal[None, :]          # (K, n)
    rng = np.random.default_rng(int(as_of.strftime("%Y%m%d")) if seed is None else seed)
    R = np.empty((n_resid, n))
    for i in range(n_resid):
        f = 0
        while f < n:
            j = rng.choice(fit["starts"])
            t = min(block, n - f)
            R[i, f:f + t] = fit["resid"][j:j + t]
            f += t
    a = rng.normal(fit["a_post"], np.sqrt(fit["a_post_var"]), n_resid)
    rsum = R.sum(1) + n * a                                     # (M,)
    if fit["gamma"] > 0:     # scenario's whole-winter mean HDD anomaly shifts the intercept
        win = weeks.month.isin(WINTER)
        an = (H - fit["norm"].reindex(_woy(weeks)).values[None, :])[:, win]
        hs = (fit["h_cur"] * fit["n_cur_win"] + an.sum(1)) / (fit["n_cur_win"] + win.sum())
        wx = wx + fit["gamma"] * hs[:, None]
    lv = float(s[F0]) + wx.sum(1)[:, None] + rsum[None, :]
    meta.update({"as_of": as_of, "Y": Y, "target_week": T, "storage_week": F0, "storage_level": float(s[F0]),
                 "weeks_to_go": n, "a_prior": fit["a_prior"], "a_post": fit["a_post"],
                 "a_post_sd": float(np.sqrt(fit["a_post_var"])), "tau": float(np.sqrt(fit["tau2"])),
                 "hdd_slope": float(b[0]), "n_cur_weeks": fit["n_cur"], "gamma": fit["gamma"]})
    return lv.ravel(), meta


def climatology_samples(as_of, first_year=1980):
    """Walk-forward climatology of x: x of every past winter (EIA monthly end-March, 5-yr mean)."""
    m = _monthly_asof(as_of)
    mar = m[m.index.month == 3]
    mar.index = mar.index.year
    xs = [mar[y] - mar[[y - k for k in range(1, 6)]].mean() for y in mar.index
          if y >= first_year and y < winter_of(as_of) and all((y - k) in mar.index for k in range(1, 6))]
    return np.array(xs, float)


def summarise(x: np.ndarray, **meta) -> dict:
    x = np.sort(np.asarray(x, float))
    q = dict(zip(QS, np.quantile(x, QS)))
    return {"samples": x, "quantiles": q, "prob_below": lambda x0, _x=x: float(np.searchsorted(_x, x0) / len(_x)),
            **meta}


def physical_dist(as_of_date, model="analog", baseline_bcf=None, centre=None, **kw) -> dict:
    """Physical as-of distribution of end-March storage deviation x (Bcf).

    model: "analog" (default, see module docstring) or "climatology".
    baseline_bcf: override the 5-year end-March baseline (x = level - baseline).
    centre: None, or "bench5" to recentre the median on the 5-year path (v5 convention; worse
            calibrated in the benchmark, kept only as a sensitivity).
    Returns dict(samples, quantiles, prob_below(x0) -> P(x < x0), plus metadata).
    """
    as_of = pd.Timestamp(as_of_date)
    Y = winter_of(as_of)
    base = baseline(Y, as_of) if baseline_bcf is None else float(baseline_bcf)
    if model == "climatology":
        return summarise(climatology_samples(as_of), model=model, as_of=as_of, Y=Y, baseline=base)
    if model != "analog":
        raise ValueError(model)
    opts = {**DEFAULTS, **kw}
    lv, meta = _simulate_levels(as_of, **opts)
    if centre == "bench5":
        lv = lv - np.median(lv) + bench5(as_of)
    return summarise(lv - base, model=model, baseline=base, options=opts, centre=centre, **meta)
