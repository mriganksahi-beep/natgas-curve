"""Composite (fallback / cross-check) risk-neutral density of the March-April spread at the March option expiry.

    S_T = F_Mar(T) - F_Apr(T),  T = March ON / G4X option expiry (trading day before the H futures LTD).

Route: two outright ON smiles + a copula, instead of the G4X spread-option strikes (the direct engine).

1. Legs. Per date, Black-76 IVs (r = 0) from OUT-of-the-money ON settles >= 0.002 of the March (H) and April (J)
   contracts of delivery year Y; raw SVI fitted to total variance w(k) = iv^2 T in log-moneyness k = ln(K/F)
   (>= 5 quotes; quadratic in k with 3-4; else no reading). Arbitrage checks: Lee wing bound
   b(1+|rho|) <= 2 (enforced in the fit), Gatheral g(k) >= 0 (butterfly; flagged), the BL density's negative
   mass (clipped, renormalised, flagged).
   * March leg: its options expire at T, so its marginal is Breeden-Litzenberger on the fitted smile.
   * April leg: its options expire ~1 month later (T_J). The distribution of F_J at T is obtained by removing
     the April contract's expected variance over its last month [T, T_J]:
         sigma_h^2 T = sigma_J^2 T_J - (q sigma_J)^2 (T_J - T)        (ATM forward variance)
     Default q = 1 (Q_MODE; April's vol over [0, T] = its ATM IV to its own expiry). Option Q_MODE = "wf":
     q = walk-forward median, over COMPLETED past winters and the same tenor bucket, of
     IV_J,atm(on the March option expiry day) / IV_J,atm(d)  (the implied vol of the April contract's last
     month, relative to today's); q = 1 if no past winter. On 2014-16 the walk-forward q (1.0-1.3) made the
     G4X ATM straddle infeasible (cheaper than comonotone legs) on 26% of dense days against 3% with q = 1,
     with no gain in tail fit, so q = 1 was chosen. Floor sigma_h >= 0.5 sigma_J (flagged).
     The smile shape is carried in standardised moneyness: w_h(k) = w_J(k * sqrt(W_J/W_h)) * W_h / W_J,
     W = ATM total variance. This is an APPROXIMATION (smile shape over [0,T] assumed equal to that over [0,T_J]).
   Each marginal is shifted to have mean exactly F (martingale), the shift is reported.
2. Dependence. Copula on the legs' uniform scores; fixed common random numbers (200k antithetic pairs, seed 0):
   'gauss' (Gaussian), 't4' (Student-t, nu = 4: tail DEPENDENCE, i.e. joint extremes), 'mix' (decoupling mixture:
   with prob P_MIX the legs follow a Gaussian copula with RHO_LO, otherwise rho; rho is the free parameter).
   Correlation source (corr_mode):
     'cso'   rho calibrated so the composite's ATM spread straddle E|S - F| equals the G4X March ATM straddle
             (Bachelier nv from the strike nearest the parity F; valid only if the previous trading day's
             near-money G4X open interest > 0 and nv is not the exchange default ~1.18, else unavailable);
     'hist'  EWMA (halflife 20 trading days, last 250 days) of daily log-return correlation of H and J futures,
             minus a walk-forward correlation-premium offset c = median over past winters' 'cso' days of
             (rho_hist - rho_cso)  (implied rho is lower than realised: v5_implied_corr);
     'fixed' walk-forward median of past winters' 'cso' rho (Nov-Feb, same tenor bucket); 0.97 if none;
     'auto'  cso -> hist -> fixed.
   A float may be passed as corr_mode to force rho.
3. Output dict, same keys as the direct engine (src/tool/rnd_direct.spread_density): grid, pdf, cdf (arrays),
   cdf_f(x), prob_above(k), quantile(p) (callables), F, T, method, flags (= quality), usable, n_quotes; plus
   prob_above_table {0.10, 0.25, 0.50: P(S_T > F + x)}, quantiles, rho, rho_source, legs.
   F here is the futures spread H - J (the direct engine uses the G4X parity forward; same within 0.5c
   except blow-up days).
4. select(direct, composite): the recommended route choice (direct if usable with >= 3 open-interest quotes).

Data discipline: every read is filtered to rows dated <= the query date (options, futures) and <= CFG['cut'].
Point CFG at the locked files with configure(options=..., futures=..., cut=...) to run unchanged later; caches
are tagged with the cut so dev caches are never reused for another sample.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay,
                                    USMartinLutherKingJr, USMemorialDay, USPresidentsDay,
                                    USThanksgivingDay, nearest_workday)
from pandas.tseries.offsets import CustomBusinessDay
from scipy.optimize import brentq, least_squares
from scipy.special import ndtr, stdtr

ROOT = Path(__file__).resolve().parents[2]
CFG = {
    "options": Path(os.environ.get("COMPOSITE_OPTIONS", ROOT / "data/raw/options/ng_options_near_money_v2.csv.gz")),
    "futures": Path(os.environ.get("COMPOSITE_FUTURES", ROOT / "data/raw/ng_curve_daily.csv")),
    "cut": pd.Timestamp(os.environ.get("COMPOSITE_CUT", os.environ.get("RND_DATA_CUTOFF", "2021-12-31"))),
    "cache": Path(os.environ.get("COMPOSITE_CACHE", ROOT / "data/processed/research_tool_composite")),
}
THRESH = (0.10, 0.25, 0.50)
NSIM = 60_000             # x2 antithetic
P_MIX, RHO_LO = 0.05, 0.30   # decoupling-mixture copula (chosen on 2014-16 dense days, 9-cell grid, see report)
EWMA_HL, EWMA_WIN = 20, 250
DEFAULT_RHO = 0.97
MIN_PREV_OI = 1
Q_MODE = 1.0              # April last-month vol ratio: 1.0 (chosen on 2014-16: fewer Frechet-infeasible CSO days)
                          # or "wf" (walk-forward implied ratio q_hat)


def configure(options=None, futures=None, cut=None, cache=None):
    if options is not None:
        CFG["options"] = Path(options)
    if futures is not None:
        CFG["futures"] = Path(futures)
    if cut is not None:
        CFG["cut"] = pd.Timestamp(cut)
    if cache is not None:
        CFG["cache"] = Path(cache)
    for f in (_opts, _opts_by_day, _futs, _q_samples):
        f.cache_clear()


def _cache_path(name):
    d = CFG["cache"]
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}_cut{CFG['cut']:%Y%m%d}.pkl"


# ----------------------------------------------------------------------------------------------- calendar
class _Nymex(AbstractHolidayCalendar):
    rules = [Holiday("NewYear", month=1, day=1, observance=nearest_workday), USMartinLutherKingJr,
             USPresidentsDay, GoodFriday, USMemorialDay, Holiday("July4", month=7, day=4, observance=nearest_workday),
             USLaborDay, USThanksgivingDay, Holiday("Christmas", month=12, day=25, observance=nearest_workday)]


BDAY = CustomBusinessDay(calendar=_Nymex())


def expiries(Y):
    """H/J futures last trade (3 business days before the delivery month) and option expiry (1 bday before)."""
    h_ltd = pd.Timestamp(Y, 3, 1) - 3 * BDAY
    j_ltd = pd.Timestamp(Y, 4, 1) - 3 * BDAY
    return {"H_ltd": h_ltd, "H_oexp": h_ltd - BDAY, "J_ltd": j_ltd, "J_oexp": j_ltd - BDAY}


def tdays(d0, d1):
    """Trading days in (d0, d1]."""
    if d1 <= d0:
        return 0
    return len(pd.date_range(d0 + BDAY, d1, freq=BDAY))


# ----------------------------------------------------------------------------------------------- data
@lru_cache(maxsize=1)
def _opts():
    """ON (cm 3, 4) and G4X (cm 3) rows of the Mar/Apr contracts, Sep..Mar of each winter, dated <= cut."""
    p = _cache_path("opts_hj")
    if p.exists():
        return pd.read_pickle(p)
    out = []
    for ch in pd.read_csv(CFG["options"], chunksize=1_000_000, parse_dates=["date"]):
        ch = ch[ch.date <= CFG["cut"]]                                  # cut on load
        ch = ch[(ch.root == "ON") & ch.cm.isin([3, 4]) | (ch.root == "G4X") & (ch.cm == 3)]
        m, y = ch.date.dt.month, ch.date.dt.year
        out.append(ch[((m >= 9) & (ch.cy == y + 1)) | ((m <= 3) & (ch.cy == y))])
    o = pd.concat(out)
    o = o[o.settle > 0].copy()
    o["K"] = o.k_raw / 1000.0
    o["Y"] = o.cy.astype(int)
    o = o[["date", "root", "Y", "cm", "cp", "K", "settle", "volume", "oi"]].sort_values(["date", "root", "cm", "K"])
    assert o.date.max() <= CFG["cut"]
    o.to_pickle(p)
    return o


@lru_cache(maxsize=1)
def _opts_by_day():
    o = _opts()
    return {k: g for k, g in o.groupby(["date", "Y"])}


@lru_cache(maxsize=1)
def _futs():
    """H and J settles per delivery year Y, dated <= cut: dict Y -> DataFrame[H, J]."""
    p = _cache_path("fut_hj")
    if p.exists():
        return pd.read_pickle(p)
    c = pd.read_csv(CFG["futures"], parse_dates=["date"], usecols=["date", "contract_year", "contract_month", "settle"])
    c = c[(c.date <= CFG["cut"]) & c.contract_month.isin([3, 4])].dropna(subset=["settle"])
    px = c.pivot_table(index="date", columns=["contract_year", "contract_month"], values="settle").sort_index()
    out = {}
    for Y in sorted({y for y, _ in px.columns}):
        if (Y, 3) in px.columns and (Y, 4) in px.columns:
            d = pd.DataFrame({"H": px[(Y, 3)], "J": px[(Y, 4)]}).dropna()
            out[Y] = d[d.index <= expiries(Y)["H_ltd"]]
    pd.to_pickle(out, p)
    return out


# ----------------------------------------------------------------------------------------------- Black / smiles
def black(F, K, w, call=True):
    sw = np.sqrt(np.maximum(w, 1e-12))
    d1 = (np.log(F / K) + 0.5 * w) / sw
    c = F * ndtr(d1) - K * ndtr(d1 - sw)
    return np.where(call, c, c - (F - K))


def black_iv(px, F, K, T, call, lo=0.01, hi=6.0, n=55):
    a, b = np.full(len(px), lo), np.full(len(px), hi)
    for _ in range(n):
        m = 0.5 * (a + b)
        up = black(F, K, m * m * T, call) < px
        a, b = np.where(up, m, a), np.where(up, b, m)
    iv = 0.5 * (a + b)
    iv[(iv > hi * 0.999) | (iv < lo * 1.001)] = np.nan
    return iv


def svi_w(p, k):
    a, b, r, m, s = p
    return a + b * (r * (k - m) + np.sqrt((k - m) ** 2 + s * s))


WING_PAD, WING_IV_SLOPE = 0.02, 0.5


def _wing_capped(p, kmin, kmax, T):
    """SVI inside the quoted range (+ WING_PAD); outside, IV linear in k with the boundary IV-slope clipped to
    |d iv/dk| <= WING_IV_SLOPE, so the wings cannot explode where no quote exists (degenerate SVI fits to 1-2 tick
    wing settles otherwise create large negative densities)."""
    kl, ku = kmin - WING_PAD, kmax + WING_PAD
    eps = 1e-4
    iv = lambda x: np.sqrt(np.maximum(svi_w(p, x), 1e-10) / T)
    vl, vu = iv(kl), iv(ku)
    sl = np.clip((iv(kl + eps) - iv(kl - eps)) / (2 * eps), -WING_IV_SLOPE, 0.0)
    su = np.clip((iv(ku + eps) - iv(ku - eps)) / (2 * eps), 0.0, WING_IV_SLOPE)

    def f(x):
        x = np.asarray(x, float)
        v = iv(np.clip(x, kl, ku))
        v = np.where(x < kl, vl + sl * (x - kl), v)
        v = np.where(x > ku, vu + su * (x - ku), v)
        return np.maximum(v, 0.05) ** 2 * T
    return f


def svi_g(p, k):
    """Gatheral's butterfly function g(k) (>= 0 <=> no butterfly arbitrage)."""
    a, b, r, m, s = p
    x = k - m
    sq = np.sqrt(x * x + s * s)
    w = a + b * (r * x + sq)
    w1 = b * (r + x / sq)
    w2 = b * s * s / sq ** 3
    w = np.maximum(w, 1e-12)
    return (1 - k * w1 / (2 * w)) ** 2 - w1 ** 2 / 4 * (1 / w + 0.25) + w2 / 2


def leg_quotes(day, cm, F, T):
    q = day[(day.root == "ON") & (day.cm == cm)]
    q = q[((q.cp == "C") & (q.K >= F)) | ((q.cp == "P") & (q.K < F))]
    q = q[q.settle >= 0.002]
    if len(q) == 0:
        return np.array([]), np.array([]), q
    K = q.K.values
    iv = black_iv(q.settle.values, F, K, T, (q.cp == "C").values)
    ok = np.isfinite(iv)
    return np.log(K[ok] / F), iv[ok], q[ok]


def fit_smile(k, iv, T):
    """Two-pass fit: drop quotes whose IV misses the first fit by > max(4 MAD, 0.03) (stale/off-market settles)."""
    s = _fit_smile(k, iv, T)
    if s is None or len(k) < 6:
        return s
    r = np.sqrt(s["f"](k) / T) - iv
    mad = np.median(np.abs(r - np.median(r))) * 1.4826
    keep = np.abs(r) <= max(4 * mad, 0.03)
    if keep.all() or keep.sum() < 5:
        s["n_drop"] = 0
        return s
    s2 = _fit_smile(k[keep], iv[keep], T)
    s2["n_drop"] = int((~keep).sum())
    return s2


def _fit_smile(k, iv, T):
    """Fit total variance w(k). Returns dict(kind, f(k)->w, params, rmse_iv, flags)."""
    n = len(k)
    if n < 3:
        return None
    w_obs = iv * iv * T
    # ATM variance by linear interpolation in k (or nearest)
    o = np.argsort(k)
    k, iv, w_obs = k[o], iv[o], w_obs[o]
    w_atm = float(np.interp(0.0, k, w_obs))
    sd = np.sqrt(w_atm)
    wt = 0.3 + np.exp(-0.5 * (k / max(2.5 * sd, 0.05)) ** 2)
    if n >= 5:
        # parameters (a, bf, rho, m, s) with b = bf * 2 / (1 + |rho|): Lee's wing bound b(1+|rho|) <= 2 by construction
        def to_p(z):
            a, bf, r, m, s_ = z
            return np.array([a, bf * 2.0 / (1 + abs(r)), r, m, s_])

        kg = np.linspace(k.min() - 0.1, k.max() + 0.1, 41)

        def res(z):
            pz = to_p(z)
            w = svi_w(pz, k)
            r = wt * (np.sqrt(np.maximum(w, 1e-10) / T) - iv)
            return np.concatenate([r, 0.05 * np.minimum(svi_g(pz, kg), 0.0)])   # butterfly penalty
        best = None
        for r0, s0 in ((-0.2, 0.1), (0.3, 0.1), (0.0, 0.3)):
            b0 = 0.5 * w_atm / max(s0, 0.05)
            z0 = [max(w_atm - b0 * s0, 1e-6), min(b0 * (1 + abs(r0)) / 2, 0.9), r0, 0.0, s0]
            lo = [-1.0, 1e-6, -0.999, -1.5, 1e-3]
            hi = [2.0, 1.0, 0.999, 1.5, 3.0]
            z0 = np.clip(z0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
            try:
                sol = least_squares(res, z0, bounds=(lo, hi), max_nfev=400, loss="soft_l1", f_scale=0.01)
            except Exception:
                continue
            pp = to_p(sol.x)
            a, b, r, m, s_ = pp
            pen = sol.cost + 1e3 * max(0.0, -(a + b * s_ * np.sqrt(1 - r * r)))
            if best is None or pen < best[0]:
                best = (pen, pp)
        if best is not None:
            p = best[1]
            kk = np.linspace(k.min() - WING_PAD, k.max() + WING_PAD, 201)
            g = svi_g(p, kk)
            fit = np.sqrt(np.maximum(svi_w(p, k), 1e-10) / T)
            return {"kind": "svi", "params": p, "f": _wing_capped(p, k.min(), k.max(), T),
                    "rmse_iv": float(np.sqrt(np.mean((fit - iv) ** 2))), "w_atm": float(svi_w(p, 0.0)),
                    "lee_ok": bool(p[1] * (1 + abs(p[2])) <= 2.0 + 1e-6),
                    "bfly_ok": bool(g.min() >= -1e-6), "n": n, "kmin": float(k.min()), "kmax": float(k.max())}
    # quadratic in k for iv, flat extrapolation outside the quoted range
    c = np.polyfit(k, iv, 2 if n >= 3 else 0, w=wt)
    kl, ku = k.min(), k.max()
    f = lambda x, c=c, kl=kl, ku=ku: np.maximum(np.polyval(c, np.clip(x, kl, ku)), 0.02) ** 2 * T
    fit = np.sqrt(f(k) / T)
    return {"kind": "quad", "params": c, "n_drop": 0, "f": f, "rmse_iv": float(np.sqrt(np.mean((fit - iv) ** 2))),
            "w_atm": float(f(np.array([0.0]))[0]), "lee_ok": True, "bfly_ok": True, "n": n,
            "kmin": float(kl), "kmax": float(ku)}


def marginal(F, wfun, w_atm):
    """BL marginal of F_T from total-variance function w(k). Returns (x grid, cdf, flags)."""
    sd = np.sqrt(max(w_atm, 1e-8))
    kl, ku = -max(7 * sd, 0.25), max(9 * sd, 0.35)
    k = np.linspace(kl, ku, 3001)
    K = F * np.exp(k)
    C = black(F, K, wfun(k), True)
    dC = np.gradient(C, K)
    cdf_raw = 1.0 + dC
    neg = float(np.clip(-np.diff(cdf_raw), 0, None).sum())          # mass of negative density
    cdf = np.clip(np.maximum.accumulate(np.clip(cdf_raw, 0, 1)), 0, 1)
    cdf[0], cdf[-1] = 0.0, 1.0                                       # truncate (tiny) residual tails onto grid
    cdf = cdf + np.linspace(0, 1e-9, len(cdf))                       # strictly increasing for inversion
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    return K, cdf, {"neg_mass": neg}


# ----------------------------------------------------------------------------------------------- copulas
@lru_cache(maxsize=1)
def _crn():
    rng = np.random.default_rng(0)
    z1 = rng.standard_normal(NSIM)
    z2 = rng.standard_normal(NSIM)
    chi = rng.chisquare(4, NSIM)
    reg = rng.random(NSIM)
    z1, z2 = np.concatenate([z1, -z1]), np.concatenate([z2, -z2])
    chi, reg = np.concatenate([chi, chi]), np.concatenate([reg, reg])
    return z1, z2, np.sqrt(4.0 / chi), reg


def _uniforms(rho, family, n=None):
    z1, z2, wt, reg = _crn()
    if n is not None:      # calibration subset (first n draws and their antithetic partners)
        h = len(z1) // 2
        ix = np.r_[0:n // 2, h:h + n // 2]
        z1, z2, wt, reg = z1[ix], z2[ix], wt[ix], reg[ix]
    if family == "mix":
        r = np.where(reg < P_MIX, RHO_LO, rho)
        x2 = r * z1 + np.sqrt(1 - r * r) * z2
        return ndtr(z1), ndtr(x2)
    x2 = rho * z1 + np.sqrt(max(1 - rho * rho, 0.0)) * z2
    if family == "t4":
        return stdtr(4, z1 * wt), stdtr(4, x2 * wt)
    return ndtr(z1), ndtr(x2)


def simulate(legH, legJ, rho, family, n=None):
    u1, u2 = _uniforms(rho, family, n)
    xh = np.interp(u1, legH["cdf"], legH["x"]) + legH["shift"]
    xj = np.interp(u2, legJ["cdf"], legJ["x"]) + legJ["shift"]
    return xh - xj


def calibrate_rho(legH, legJ, target_absdev, family):
    """rho such that E|S - E S| = target (the Bachelier ATM straddle value nv*sqrt(2T/pi))."""
    def f(r):
        s = simulate(legH, legJ, r, family, n=40_000)
        return np.mean(np.abs(s - s.mean())) - target_absdev
    hi, lo = 0.99999, -0.5
    fhi = f(hi)
    if fhi > 0:
        return hi, "rho_cap"       # CSO cheaper than comonotone legs (the 'rho > 1' case)
    if f(lo) < 0:
        return lo, "rho_floor"
    return brentq(f, lo, hi, xtol=2e-4), "ok"


# ----------------------------------------------------------------------------------------------- per-date pieces
def _bach_straddle_nv(px, T):
    return px / np.sqrt(2 * T / np.pi)


def cso_quote(Y, date, F_spread, T):
    """G4X March ATM Bachelier vol at the strike nearest the parity F; validity by previous-day OI."""
    g = _opts_by_day().get((date, Y))
    out = {"nv": np.nan, "valid": False, "n_quotes": 0, "n_oi_prev": 0, "default": False, "reason": "no_quotes"}
    if g is None:
        return out
    g = g[g.root == "G4X"]
    if len(g) == 0:
        return out
    c = g[g.cp == "C"].groupby("K").settle.first()
    p = g[g.cp == "P"].groupby("K").settle.first()
    j = c.index.intersection(p.index)
    out["n_quotes"] = int(len(g))
    if len(j) == 0:
        out["reason"] = "no_pair"
        return out
    diff = c[j] - p[j]
    k0 = diff.abs().idxmin()
    Fp = k0 + diff[k0]
    # straddle at k0, re-centred to ATM in Bachelier: solve nv from C+P at k0
    st = c[k0] + p[k0]
    lo_, hi_ = 1e-4, 20.0
    for _ in range(60):
        m = 0.5 * (lo_ + hi_)
        v = m * np.sqrt(T)
        d = (Fp - k0) / v
        val = 2 * ((Fp - k0) * ndtr(d) + v * np.exp(-0.5 * d * d) / np.sqrt(2 * np.pi)) - (Fp - k0)
        lo_, hi_ = (m, hi_) if val < st else (lo_, m)
    nv = 0.5 * (lo_ + hi_)
    out.update(nv=nv, F_par=Fp)
    # previous trading day's near-money OI (any strike within 0.30 of F)
    days = sorted(d for (d, y) in _opts_by_day() if y == Y and d < date)
    prev = days[-1] if days else None
    noi = 0
    if prev is not None:
        gp = _opts_by_day()[(prev, Y)]
        gp = gp[(gp.root == "G4X") & ((gp.K - F_spread).abs() <= 0.30)]
        noi = int((gp.oi.fillna(0) > 0).sum())
    out["n_oi_prev"] = noi
    out["default"] = bool(1.15 <= nv <= 1.21)
    out["valid"] = bool(noi >= MIN_PREV_OI and not out["default"] and T >= 2 / 252)
    out["reason"] = "ok" if out["valid"] else ("default" if out["default"] else "no_prev_oi")
    return out


def _bucket(td):
    return 0 if td <= 21 else 1 if td <= 42 else 2 if td <= 63 else 3 if td <= 126 else 4


def legs(date, Y, q=None):
    """Both marginals at the March option expiry, or None. q: April last-month vol ratio (None -> walk-forward)."""
    date = pd.Timestamp(date)
    fut = _futs().get(Y)
    if fut is None or date not in fut.index:
        return None
    ex = expiries(Y)
    tdH, tdJ = tdays(date, ex["H_oexp"]), tdays(date, ex["J_oexp"])
    if tdH < 1:
        return None
    TH, TJ = tdH / 252.0, tdJ / 252.0
    FH, FJ = float(fut.H[date]), float(fut.J[date])
    day = _opts_by_day().get((date, Y))
    if day is None:
        return None
    kH, ivH, _ = leg_quotes(day, 3, FH, TH)
    kJ, ivJ, _ = leg_quotes(day, 4, FJ, TJ)
    sH, sJ = fit_smile(kH, ivH, TH), fit_smile(kJ, ivJ, TJ)
    j_proxy = False
    if sH is None:
        return None
    if sJ is None or sJ["n"] < 5:
        # April quotes missing (e.g. Oct-Dec 2020 gap in the raw file): April smile = March smile shape in
        # standardised moneyness, ATM vol = walk-forward median IV_J/IV_H ratio x IV_H (flagged j_proxy)
        ratio, _ = jh_ratio(Y, tdH)
        WH = sH["w_atm"]
        WJs = (ratio ** 2) * WH / TH * TJ
        sJ = {"kind": "proxy_from_H", "f": lambda x, f=sH["f"], sc=np.sqrt(WH / WJs), r=WJs / WH: f(x * sc) * r,
              "w_atm": WJs, "rmse_iv": np.nan, "lee_ok": sH["lee_ok"], "bfly_ok": sH["bfly_ok"], "n": 0,
              "n_drop": 0, "ratio": ratio}
        j_proxy = True
    # April: forward-variance removal of its last month [T_H, T_J]
    qsrc = "given"
    if q is None:
        q, qsrc = q_hat(Y, tdH) if Q_MODE == "wf" else (float(Q_MODE), "fixed_q")
    WJ = sJ["w_atm"]
    sigJ2 = WJ / TJ
    Wh = WJ - q * q * sigJ2 * (TJ - TH)
    floored = Wh < 0.25 * sigJ2 * TH
    Wh = max(Wh, 0.25 * sigJ2 * TH)
    scale = np.sqrt(WJ / Wh)
    wJh = lambda x, f=sJ["f"], sc=scale, r=Wh / WJ: f(x * sc) * r
    xH, cH, fH = marginal(FH, sH["f"], sH["w_atm"])
    xJ, cJ, fJ = marginal(FJ, wJh, Wh)
    out = {"date": date, "Y": Y, "TH": TH, "TJ": TJ, "tdH": tdH, "FH": FH, "FJ": FJ,
           "H": {"x": xH, "cdf": cH, "smile": sH, "F": FH, **fH},
           "J": {"x": xJ, "cdf": cJ, "smile": sJ, "F": FJ, "W_h": Wh, "floored": bool(floored), "proxy": j_proxy, **fJ},
           "q": q, "q_src": qsrc, "iv_H_atm": np.sqrt(sH["w_atm"] / TH), "iv_J_atm": np.sqrt(WJ / TJ),
           "iv_J_atm_h": np.sqrt(Wh / TH)}
    # martingale shift
    z1, _, _, _ = _crn()
    u = ndtr(z1)
    for leg, F in (("H", FH), ("J", FJ)):
        m = np.interp(u, out[leg]["cdf"], out[leg]["x"]).mean()
        out[leg]["shift"] = F - m
    return out


# ----------------------------------------------------------------------------------------------- walk-forward tables
def _winter_days(Y):
    fut = _futs().get(Y)
    if fut is None:
        return []
    ex = expiries(Y)
    idx = fut.index[(fut.index >= pd.Timestamp(Y - 1, 10, 1)) & (fut.index < ex["H_oexp"]) & (fut.index <= CFG["cut"])]
    return list(idx)


def atm_table(Y):
    """ATM IV of H (to its expiry) and J (to its expiry) per date for winter Y (Oct 1 .. H option expiry), cached."""
    p = _cache_path(f"atm_{Y}")
    if p.exists():
        return pd.read_pickle(p)
    fut = _futs().get(Y)
    ex = expiries(Y)
    rows = []
    if fut is not None:
        idx = fut.index[(fut.index >= pd.Timestamp(Y - 1, 10, 1)) & (fut.index <= ex["H_oexp"])]
        for d in idx:
            day = _opts_by_day().get((d, Y))
            if day is None:
                continue
            tdH = tdays(d, ex["H_oexp"])
            r = {"date": d, "tdH": tdH, "ivH": np.nan, "ivJ": np.nan}
            for cm, col, F, oe in ((3, "ivH", fut.H[d], ex["H_oexp"]), (4, "ivJ", fut.J[d], ex["J_oexp"])):
                T = tdays(d, oe) / 252.0
                if T <= 0:
                    continue
                k, iv, _ = leg_quotes(day, cm, float(F), T)
                if len(k) >= 2:
                    o = np.argsort(k)
                    r[col] = float(np.interp(0.0, k[o], iv[o]))
            rows.append(r)
    t = pd.DataFrame(rows, columns=["date", "tdH", "ivH", "ivJ"])
    if Y in _futs() and ex["H_oexp"] <= CFG["cut"]:
        t.to_pickle(p)
    return t


def jh_ratio(Y, tdH):
    """Walk-forward median of IV_J,atm / IV_H,atm (raw, each to its own expiry) over winters < Y, same bucket."""
    b = _bucket(tdH)
    rs = []
    for y in range(Y - 15, Y):
        if y not in _futs() or expiries(y)["H_oexp"] > CFG["cut"]:
            continue
        t = atm_table(y)
        t = t[(t.tdH > 0) & (t.tdH.map(_bucket) == b)]
        rs.append((t.ivJ / t.ivH).dropna().values)
    rs = np.concatenate(rs) if rs else np.array([])
    return (float(np.median(rs)), "walk_forward") if len(rs) >= 10 else (0.85, "default")


@lru_cache(maxsize=64)
def _q_samples(Y):
    """Ratios IV_J(H oexp)/IV_J(d) of winter Y (needs Y's H option expiry inside the data)."""
    ex = expiries(Y)
    if ex["H_oexp"] > CFG["cut"]:
        return None
    t = atm_table(Y)
    t = t[t.ivJ.notna()]
    last = t[t.date == ex["H_oexp"]]
    if len(last) == 0:              # expiry-day settle missing: use the last available day <= 2 tdays before
        last = t[t.tdH <= 2].tail(1)
    if len(last) == 0:
        return None
    t = t[(t.tdH > 0) & t.ivJ.notna()].copy()
    t["ratio"] = float(last.ivJ.iloc[0]) / t.ivJ
    t["bucket"] = t.tdH.map(_bucket)
    return t[["bucket", "ratio"]]


def q_hat(Y, tdH):
    """Walk-forward q: median ratio over completed winters < Y, same tenor bucket."""
    b = _bucket(tdH)
    rs = []
    for y in range(Y - 15, Y):
        s = _q_samples(y)
        if s is not None:
            rs.append(s.ratio[s.bucket == b].values)
    rs = np.concatenate(rs) if rs else np.array([])
    if len(rs) < 10:
        return 1.0, "default_q1"
    return float(np.clip(np.median(rs), 0.6, 2.5)), "walk_forward"


def ewma_corr(Y, date):
    fut = _futs().get(Y)
    if fut is None:
        return np.nan
    f = fut[fut.index <= date].tail(EWMA_WIN + 1)
    if len(f) < 40:
        return np.nan
    r = np.log(f).diff().dropna()
    lam = 0.5 ** (1 / EWMA_HL)
    w = lam ** np.arange(len(r))[::-1]
    a, b = r.H.values, r.J.values
    cab, caa, cbb = (w * a * b).sum(), (w * a * a).sum(), (w * b * b).sum()
    return float(cab / np.sqrt(caa * cbb))


def cso_rho_table(Y, family):
    """Daily rho_cso (where valid) and rho_ewma for winter Y, Oct..expiry, dated <= cut. Cached per winter."""
    p = _cache_path(f"rho_{family}_{Y}")
    if p.exists():
        return pd.read_pickle(p)
    rows = []
    for d in _winter_days(Y):
        L = legs(d, Y)
        if L is None:
            continue
        cq = cso_quote(Y, d, L["FH"] - L["FJ"], L["TH"])
        r_c = np.nan
        if cq["valid"]:
            r_c, _ = calibrate_rho(L["H"], L["J"], cq["nv"] * np.sqrt(2 * L["TH"] / np.pi), family)
        rows.append({"date": d, "tdH": L["tdH"], "rho_cso": r_c, "rho_ewma": ewma_corr(Y, d)})
    t = pd.DataFrame(rows, columns=["date", "tdH", "rho_cso", "rho_ewma"])
    if expiries(Y)["H_oexp"] <= CFG["cut"]:      # only completed winters are written (no partial caches)
        t.to_pickle(p)
    return t


def _past_rho(Y, family, tdH):
    """Walk-forward stats from winters < Y: (median rho_cso in bucket, median(ewma - cso) offset)."""
    b = _bucket(tdH)
    rc, off = [], []
    for y in range(Y - 15, Y):
        if y not in _futs() or expiries(y)["H_oexp"] > CFG["cut"]:
            continue
        t = cso_rho_table(y, family)
        t = t[t.date.dt.month.isin([11, 12, 1, 2]) & t.rho_cso.notna()]
        if len(t) == 0:
            continue
        tb = t[t.tdH.map(_bucket) == b]
        rc.append(tb.rho_cso.values)
        off.append((t.rho_ewma - t.rho_cso).dropna().values)
    rc = np.concatenate(rc) if rc else np.array([])
    off = np.concatenate(off) if off else np.array([])
    return (float(np.median(rc)) if len(rc) >= 5 else np.nan, float(np.median(off)) if len(off) >= 10 else np.nan)


# ----------------------------------------------------------------------------------------------- main entry
def _density_from_sample(s, F, n=401):
    ss = np.sort(s)
    lo, hi = ss[int(0.0005 * len(ss))], ss[int(0.9995 * len(ss)) - 1]
    pad = 0.15 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, n)
    cdf = np.searchsorted(ss, grid, side="right") / len(ss)
    h = grid[1] - grid[0]
    edges = np.concatenate([[grid[0] - h / 2], grid + h / 2])
    pdf, _ = np.histogram(s, bins=edges)
    pdf = pdf / (len(s) * h)
    ker = np.exp(-0.5 * (np.arange(-4, 5) / 1.5) ** 2)
    pdf = np.convolve(pdf, ker / ker.sum(), mode="same")
    qs = {p: float(np.quantile(ss, p)) for p in (0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99)}
    pa = {x: float((s > F + x).mean()) for x in THRESH}
    return grid, pdf, cdf, qs, pa


def spread_density_composite(date, winter_year, corr_mode="auto", family="gauss", q=None, return_sample=False):
    """Composite RND of S_T = F_H(T) - F_J(T) at the March option expiry of delivery year `winter_year`.

    corr_mode: 'auto' | 'cso' | 'hist' | 'fixed' | float.   family: 'gauss' | 't4' | 'mix'.
    Reads only rows dated <= date. Returns None if no reading is possible (no futures/ON quotes that day).
    """
    date = pd.Timestamp(date)
    Y = int(winter_year)
    assert date <= CFG["cut"], "query date beyond the configured data cut"
    L = legs(date, Y, q=q)
    if L is None:
        return None
    F = L["FH"] - L["FJ"]
    T = L["TH"]
    cq = cso_quote(Y, date, F, T)
    rho, src, cal = np.nan, None, "n/a"
    modes = [corr_mode] if corr_mode != "auto" else ["cso", "hist", "fixed"]
    for m in modes:
        if isinstance(m, (float, int)) and not isinstance(m, bool):
            rho, src = float(m), "given"
        elif m == "cso" and cq["valid"]:
            rho, cal = calibrate_rho(L["H"], L["J"], cq["nv"] * np.sqrt(2 * T / np.pi), family)
            src = "cso"
        elif m == "hist":
            e = ewma_corr(Y, date)
            _, off = _past_rho(Y, family, L["tdH"])
            if np.isfinite(e):
                rho = e - (off if np.isfinite(off) else 0.0)
                src = "hist" + ("" if np.isfinite(off) else "_nooffset")
        elif m == "fixed":
            med, _ = _past_rho(Y, family, L["tdH"])
            rho, src = (med, "fixed") if np.isfinite(med) else (DEFAULT_RHO, "fixed_default")
        if src is not None:
            break
    if src is None:
        return None
    rho = float(np.clip(rho, -0.5, 0.99999))
    s = simulate(L["H"], L["J"], rho, family)
    grid, pdf, cdf, qs, pa = _density_from_sample(s, F)
    quality = {
        "n_H": L["H"]["smile"]["n"], "n_J": L["J"]["smile"]["n"],
        "smile_H": L["H"]["smile"]["kind"], "smile_J": L["J"]["smile"]["kind"],
        "rmse_iv_H": L["H"]["smile"]["rmse_iv"], "rmse_iv_J": L["J"]["smile"]["rmse_iv"],
        "bfly_ok": bool(L["H"]["smile"]["bfly_ok"] and L["J"]["smile"]["bfly_ok"]),
        "lee_ok": bool(L["H"]["smile"]["lee_ok"] and L["J"]["smile"]["lee_ok"]),
        "neg_mass": max(L["H"]["neg_mass"], L["J"]["neg_mass"]),
        "mart_shift_H": L["H"]["shift"], "mart_shift_J": L["J"]["shift"],
        "april_floored": L["J"]["floored"], "april_proxy": L["J"]["proxy"],
        "n_drop": L["H"]["smile"].get("n_drop", 0) + L["J"]["smile"].get("n_drop", 0), "q": L["q"], "q_src": L["q_src"],
        "cso_valid": cq["valid"], "cso_reason": cq["reason"], "cso_nv": cq["nv"], "cso_n_oi_prev": cq["n_oi_prev"],
        "rho_calib": cal, "usable": True,
    }
    rm = np.nanmax([quality["rmse_iv_H"], quality["rmse_iv_J"]])
    quality["usable"] = bool(quality["n_H"] >= 5 and (quality["n_J"] >= 5 or quality["april_proxy"])
                             and quality["neg_mass"] < 0.05 and rm < 0.05)
    out = {"grid": grid, "pdf": pdf, "cdf": cdf, "F": F, "T": T, "tdays": L["tdH"],
           "method": f"composite:{family}:{src}", "quality": quality, "flags": quality, "usable": quality["usable"],
           "prob_above_table": pa, "quantiles": qs,
           "rho": rho, "rho_source": src, "date": date, "Y": Y, "n_quotes": quality["n_H"] + quality["n_J"],
           "legs": {"FH": L["FH"], "FJ": L["FJ"], "iv_H_atm": L["iv_H_atm"], "iv_J_atm": L["iv_J_atm"],
                    "iv_J_atm_at_T": L["iv_J_atm_h"]}}
    _attach(out)
    if return_sample:
        out["sample"] = s
    return out


def _attach(out):
    """Callables with the direct engine's (src/tool/rnd_direct.py) signatures: cdf_f(x), prob_above(k), quantile(p)."""
    grid, cdf = out["grid"], out["cdf"]
    cdf_f = lambda x: np.interp(x, grid, cdf, left=0.0, right=1.0)  # noqa: E731
    out["cdf_f"] = cdf_f
    out["prob_above"] = lambda k: 1.0 - cdf_f(k)
    cs = cdf + np.linspace(0, 1e-9, len(cdf))
    out["quantile"] = lambda p: np.interp(np.atleast_1d(np.asarray(p, float)), cs, grid)
    return out


# ----------------------------------------------------------------------------------------------- blend rule
MIN_OI_QUOTES_DIRECT = 3


def n_oi_quotes(direct):
    """G4X March quotes the direct engine used that had open interest the previous quote day."""
    if direct is None:
        return 0
    f = direct.get("flags", {}) or {}
    if f.get("no_data") or f.get("default_settle"):
        return 0
    zs = f.get("zero_oi_share", 0.0)
    zs = 0.0 if zs is None or not np.isfinite(zs) else zs
    return int(round(direct.get("n_quotes", 0) * (1 - zs)))


def select(direct, composite, min_oi=MIN_OI_QUOTES_DIRECT):
    """RECOMMENDED rule (reports/tool/rnd_composite.md): the direct G4X density when the direct engine is usable,
    not a default-settle day and has >= min_oi quotes with previous-day open interest; otherwise the composite
    (corr_mode 'auto': CSO-calibrated rho -> walk-forward EWMA rho -> walk-forward fixed rho). No weighting:
    on 2014-21 every weighted blend scored worse (CRPS) than this switch. Returns (dict, route)."""
    ok_direct = (direct is not None and bool(direct.get("usable")) and len(direct.get("grid", [])) > 0
                 and n_oi_quotes(direct) >= min_oi)
    if ok_direct:
        return direct, "direct"
    if composite is not None and composite.get("usable", True):
        return composite, "composite"
    if direct is not None and len(direct.get("grid", [])) > 0:
        return direct, "direct_low_quality"
    return composite, ("composite_low_quality" if composite is not None else "none")


def blend(direct, composite, w):
    """CDF mixture w * direct + (1 - w) * composite (kept for sensitivity work; not the recommended rule)."""
    lo = min(direct["grid"][0], composite["grid"][0])
    hi = max(direct["grid"][-1], composite["grid"][-1])
    grid = np.linspace(lo, hi, 801)
    cdf = w * direct["cdf_f"](grid) + (1 - w) * composite["cdf_f"](grid)
    out = {"grid": grid, "pdf": np.gradient(cdf, grid), "cdf": cdf, "F": direct["F"], "T": direct["T"],
           "method": f"blend:w_direct={w:.2f}", "usable": True,
           "flags": {"w_direct": w, "direct": direct.get("flags"), "composite": composite.get("flags")}}
    return _attach(out)
