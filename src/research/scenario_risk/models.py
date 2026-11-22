"""Risk models for the loss of a short 1 x March-April spread over h weeks (loss = dS, $/MMBtu).

Every model returns, for each forecast Friday d, a quantile function of the loss on a fixed grid
U = (i - 0.5)/1000. VaR_a = Q(a); ES_0.975 = mean of Q(u) for u > 0.975. All inputs are dated <= d
(prices) or <= d-7 (storage week, released d-1; Model 2 origin d-7). Specifications were fixed
before any evaluation; nothing is tuned on the evaluation sample.

(a1) HS-1y   trailing 52 weekly changes (h=1) / 52 overlapping 4-week changes (h=4) of the generic
             front-winter March-April spread
(a2) HS-3y   same, 156 weeks (fewer before mid-2013, data start Jun 2010)
(a3) HS-win  seasonal HS: Nov-Feb weeks of the last 3 winters plus this winter so far (added
             benchmark: a trader would not use summer weeks for a winter book)
(b)  EWMA    RiskMetrics lambda=0.94 on weekly changes, zero mean, normal, sigma_h = sigma*sqrt(h)
(c)  GARCH   GARCH(1,1), zero mean, Student-t, refit on the expanding weekly history each Friday;
             h=4 by simulation (4,000 paths)
(d)  PHYS    weather-shock scenario model:
               loss = (S - a) * (exp(-c * dE / 1000) - 1) + eta * sqrt(h) * z
             (S - a)  spread above its ample-storage floor (market-anchored; floored at 0.005)
             c        curvature of the convex map spread = a + b exp(-c x/1000), x = expected
                      end-March storage minus 5-yr avg (x_info), fitted on earlier winters only
             dE       h-week change in expected end-March storage (Bcf):
               d-M2   Model 2 scenarios (origin d-7): (p_k - mean) * sqrt(h / weeks_to_go)
               d-WX   weather climatology: -beta * (h-week HDD anomaly), sampled from the same
                      calendar window (+-2 weeks) in all earlier winters since 2004-05;
                      beta (Bcf/HDD) from earlier winters' weekly storage change on HDD
             eta      non-weather noise: robust s.d. (1.4826*MAD) of weekly dS in earlier winters
(e)  HYB     EWMA (0.94) on scaled changes z = dS/(S_prev - a) (floor 0.02), forecast
             sigma = (S_d - a) * sigma_z * sqrt(h), normal: EWMA scaled by the physical sensitivity.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

from src.research.scenario_risk import data as D

warnings.filterwarnings("ignore")
U = (np.arange(1000) + 0.5) / 1000
ZU = stats.norm.ppf(U)
RNG = np.random.default_rng(20261122)
NSIM = 4000
A_LO, A_HI = -0.2, 0.1   # same bounds as implied_storage (reviewed)
FLOOR_PHYS, FLOOR_HYB = 0.005, 0.02


def qgrid_from_sample(x):
    return np.quantile(x, U)


# ---------------------------------------------------------------- (a) historical simulation
def hs(df, d, h, weeks):
    col = "dS_back1" if h == 1 else "dS_back4"
    past = df.loc[:d, col].dropna().iloc[-weeks:]
    return qgrid_from_sample(past.values) if len(past) >= 26 else None


def hs_winter(df, d, h):
    col = "dS_back1" if h == 1 else "dS_back4"
    past = df.loc[:d]
    m = past.index.month
    win = past[((m >= 11) | (m <= 2)) & (past.index >= d - pd.DateOffset(years=3, months=1))][col].dropna()
    return qgrid_from_sample(win.values) if len(win) >= 26 else None


# ---------------------------------------------------------------- (b) EWMA
def ewma_sigma(x, lam=0.94, init=26):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < init + 1:
        return np.nan
    v = np.mean(x[:init] ** 2)
    for r in x[init:]:
        v = lam * v + (1 - lam) * r ** 2
    return np.sqrt(v)


def ewma(df, d, h):
    s = ewma_sigma(df.loc[:d, "dS_back1"].values)
    return None if np.isnan(s) else s * np.sqrt(h) * ZU


# ---------------------------------------------------------------- (c) GARCH-t
def garch(df, d, h):
    from arch import arch_model
    y = df.loc[:d, "dS_back1"].dropna().values * 100.0
    if len(y) < 52:
        return None
    res = arch_model(y, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    if h == 1:
        sig = np.sqrt(res.forecast(horizon=1, reindex=False).variance.values[0, 0])
        nu = res.params["nu"]
        q = sig * stats.t.ppf(U, nu) * np.sqrt((nu - 2) / nu)
    else:
        f = res.forecast(horizon=h, method="simulation", simulations=NSIM, reindex=False,
                         random_state=np.random.RandomState(1))
        q = qgrid_from_sample(f.simulations.values[0].sum(axis=1))
    return q / 100.0


# ---------------------------------------------------------------- map and physical inputs
def fmap(x, a, b, c):
    return a + b * np.exp(-c * x / 1000.0)


def fit_maps(df):
    """Walk-forward convex map per delivery year Y, fitted on Nov-Feb rows of winters < Y."""
    w = df[df.index.month.isin([11, 12, 1, 2]) & df.x_info.notna() & df.S.notna()
           & (df.weeks_to_expiry >= 25 / 7)]
    out = {}
    for Y in sorted(df.Y.unique()):
        tr = w[w.Y < Y]
        if tr.Y.nunique() < 2:
            continue
        p, _ = curve_fit(fmap, tr.x_info.values, tr.S.values, p0=[0.0, 0.1, 1.5],
                         bounds=([A_LO, 1e-5, 0.05], [A_HI, 20, 15]), maxfev=20000)
        dS = tr.dS_back1.dropna().values  # weekly changes within earlier winters
        eta = 1.4826 * np.median(np.abs(dS - np.median(dS)))
        out[Y] = dict(a=p[0], b=p[1], c=p[2], eta=eta, n_winters=tr.Y.nunique())
    return out


def hdd_beta(Y):
    """Bcf per HDD from weekly storage changes on HDD, Dec-Feb weeks of winters ending before Y."""
    s, w = D.load_storage(), D.load_hdd()
    ch = s.diff().dropna()
    j = pd.concat([ch.rename("ch"), w.rename("hdd")], axis=1, join="inner").dropna()
    j = j[j.index.month.isin([12, 1, 2]) & (j.index < pd.Timestamp(Y - 1, 11, 1))]
    if len(j) < 20:
        return np.nan
    X = np.c_[np.ones(len(j)), j.hdd.values]
    coef = np.linalg.lstsq(X, j.ch.values, rcond=None)[0]
    return -coef[1]  # positive: Bcf of extra withdrawal per HDD


def hdd_anomalies(d, h, Y):
    """Climatological h-week HDD anomalies for the window starting at d, from winters < Y-1."""
    w = D.load_hdd()
    sums = []
    for yr in range(1, 30):
        for off in (-14, -7, 0, 7, 14):
            start = d - pd.DateOffset(years=yr) + pd.Timedelta(days=off)
            if start + pd.Timedelta(days=7 * h) >= pd.Timestamp(Y - 1, 4, 1) or start < w.index[0]:
                continue
            idx = w.index.searchsorted(start + pd.Timedelta(days=4))
            if idx + h > len(w):
                continue
            seg = w.iloc[idx: idx + h]
            if len(seg) == h and (seg.index[-1] - seg.index[0]).days == 7 * (h - 1):
                sums.append(seg.sum())
    sums = np.array(sums)
    return sums - sums.mean() if len(sums) >= 20 else None


def phys(S, mp, dE, h, noise=True):
    k = max(S - mp["a"], FLOOR_PHYS)
    dE = RNG.choice(dE, NSIM, replace=True)
    loss = k * (np.exp(-mp["c"] * dE / 1000.0) - 1.0)
    if noise:
        loss = loss + mp["eta"] * np.sqrt(h) * RNG.standard_normal(NSIM)
    return qgrid_from_sample(loss)


def m2_dE(m2, d, h):
    o = d - pd.Timedelta(days=7)
    if o not in m2.index:
        return None
    r = m2.loc[o]
    p = r[[f"p{k}" for k in range(1, 100)]].values.astype(float)
    return (p - p.mean()) * np.sqrt(min(h / r.weeks_to_go, 1.0))


def hyb(df, d, h, a):
    x = df.loc[:d]
    prevS = x.S.shift(1)
    same = x.Y == x.Y.shift(1)
    z = (x.dS_back1 / np.maximum(prevS - a, FLOOR_HYB)).where(same)
    s = ewma_sigma(z.values)
    k = max(x.S.iloc[-1] - a, FLOOR_HYB)
    return None if np.isnan(s) else k * s * np.sqrt(h) * ZU


# ---------------------------------------------------------------- driver
def run_all():
    f = D.CACHE / "forecasts.pkl"
    if f.exists():
        return pd.read_pickle(f)
    df = D.add_storage_state(D.weekly_panel())
    maps = fit_maps(df)
    m2 = D.load_model2()
    betas = {Y: hdd_beta(Y) for Y in maps}
    recs = []
    for h in (1, 4):
        ev = D.eval_rows(df, h)
        ev = ev[ev.index >= "2011-11-01"]
        for d, r in ev.iterrows():
            fc = {"HS-1y": hs(df, d, h, 52), "HS-3y": hs(df, d, h, 156), "HS-win": hs_winter(df, d, h),
                  "EWMA": ewma(df, d, h), "GARCH-t": garch(df, d, h)}
            mp = maps.get(r.Y)
            if mp is not None:
                fc["HYB"] = hyb(df, d, h, mp["a"])
                an = hdd_anomalies(d, h, r.Y)
                if an is not None and not np.isnan(betas[r.Y]):
                    dE = -betas[r.Y] * an
                    fc["PHYS-WX"] = phys(r.S, mp, dE, h)
                dm = m2_dE(m2, d, h)
                if dm is not None:
                    fc["PHYS-M2"] = phys(r.S, mp, dm, h)
                    fc["PHYS-M2-pure"] = phys(r.S, mp, dm, h, noise=False)
            for k, q in fc.items():
                if q is not None:
                    recs.append(dict(date=d, h=h, model=k, winter=r.winter, S=r.S, loss=r[f"dS_{h}"], q=q,
                                     a=mp["a"] if mp else np.nan, c=mp["c"] if mp else np.nan,
                                     eta=mp["eta"] if mp else np.nan,
                                     beta=betas.get(r.Y, np.nan), x_info=r.x_info))
    out = pd.DataFrame(recs)
    out.to_pickle(f)
    pd.DataFrame(maps).T.assign(beta=pd.Series(betas)).to_csv(D.CACHE / "maps.csv")
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    fc = run_all()
    print(fc.groupby(["h", "model"]).size().unstack(0), f"{time.time() - t0:.0f}s")
    print(pd.read_csv(D.CACHE / "maps.csv", index_col=0).round(3))
