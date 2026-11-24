"""Risk-neutral density of the March-April spread at option expiry (S_T) from G4X March CSO settles.

Link (A) of the Winter Squeeze-Risk Monitor (reports/tool/SPEC.md). Investigation: reports/tool/rnd_engine.md.

Usage
-----
    from src.tool.rnd_direct import spread_density
    d = spread_density("2018-01-10", 2018)          # winter_year = March delivery year
    d["prob_above"](d["F"] + 0.25), d["cdf"](0.0), d["grid"], d["pdf"], d["flags"]

Data rule (hard)
----------------
Options and futures are read from the raw files and filtered on load to date <= DATA_CUTOFF (default
2021-12-31, the development cut). Inside a call, only rows dated <= min(date, DATA_CUTOFF) are used: the
day's own settles plus the previous quote day's open interest. To run on the locked sample later, the
orchestrator sets the environment variable RND_DATA_CUTOFF (e.g. 2026-12-31) - no code change.

Underlying, timing
------------------
S = H - J futures settle ($/MMBtu). G4X cm = 3 options settle on S at option expiry = the trading day before
the H futures last trade day (LTD = 3 business days before 1 March). T = NYMEX trading days to that
expiry / 252. F = the put-call-parity forward of the day's G4X settles (weighted to near-the-money pairs); the
H-J futures settle is the fallback (no C/P pair, or a gap > 15c). They agree within 0.5c except blow-up days
(Nov 2018: 6.5c), when the option settles follow their own parity.

Methods (method=...)
--------------------
  "sln"     DEFAULT. Mixture of shifted (displaced) lognormals with a common shift and mean F: S_T + lam ~
            sum_i w_i LN(G_i, sigma_i^2 T); 2 components on >= 5 quotes, 1 below. Prices by Black on (G_i, K + lam);
            the shift keeps the support below every quoted strike. Proper density by construction (calls monotone,
            convex, pdf >= 0); right skew and tails rising with strike come from the lognormal shape; extrapolation
            beyond quoted strikes is the model's own tail. Analytic pdf/CDF.
  "svi"     Normal (Bachelier) implied total variance w(x) = sigma_N(K)^2 T, x = K - F, fitted with
            the SVI form  w = a + b(rho (x-m) + sqrt((x-m)^2 + s^2)).  Its wings grow linearly in |x|, which
            in Bachelier terms means exponential density tails, so extrapolation beyond the quoted strikes is
            built in and smooth. Fewer than 5 quotes: falls back to "sln" with one component. Calls C(K) = Bachelier(F, K, sqrt(w(K))); CDF = 1 + dC/dK; pdf = d2C/dK2
            (finite differences on a 0.005 grid).
            A soft no-arbitrage penalty (negative density on the grid) is added to the fit; any residual
            negative density is clipped and renormalised (flag arb_clipped).
  "quad"    Quadratic normal vol sigma(x) = a + b x + c x^2 inside the quoted range, linear (C1) extension
            outside, floored at 0.25 x ATM vol.
  "mixture" Baseline (v5_implied_storage): 1-3 component normal mixture, mean F, fitted to OTM prices.
  "convex"  Model-free Breeden-Litzenberger: weighted least-squares projection of the call curve onto
            {decreasing, convex, slopes in [-1, 0], above intrinsic}; CDF knots at strike midpoints, PCHIP
            (monotone) interpolation, exponential tails matched to the last call/put price and tail prob.

Quotes and quality
------------------
Per strike the out-of-the-money settle (calls K >= F, puts K < F), |K - F| <= 2, settle >= 0.002 plus at most
one 0.001 settle per wing. Weights 1/(price + 0.003) (relative error). Strikes with no open interest (either
side) on the previous quote day are exchange marks, not trades: they are DROPPED when >= MIN_OI_QUOTES quotes
with open interest remain AND their wing keeps >= MIN_OI_SIDE such quotes (default; ZERO_OI_WEIGHT = 0);
otherwise they are kept at full weight (flag marks_kept).
Quotes that are both pinned at the exchange-default normal vol band [1.15, 1.21] and zero-OI are always dropped.
Flags: default_settle (>= half of usable quotes pinned, or flat vols ~1.18 - the v5_implied_storage rule),
zero_oi (no used strike had open interest the previous quote day), zero_oi_share, extrapolated_share
(Q-mass outside the quoted strike range), low_info (< 3 quotes), arb_clipped, rmse, mean_err.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares, minimize
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[2]
DATA_CUTOFF = pd.Timestamp(os.environ.get("RND_DATA_CUTOFF", "2021-12-31"))
CACHE = ROOT / "data/processed/research_tool_rnd"
OPT_FILE = ROOT / "data/raw/options/ng_options_near_money_v2.csv.gz"
FUT_FILE = ROOT / "data/raw/ng_curve_daily.csv"

DEFAULT_METHOD = "sln"        # chosen on 2014-16, checked 2017-21 (reports/tool/rnd_engine.md)
ZERO_OI_WEIGHT = 0.0          # 0 = drop zero-OI marks (chosen 2014-16: LOSO on OI>0 strikes 0.40c vs 0.79c)
MIN_OI_QUOTES = 5             # ...but only if >= 5 quotes with open interest remain; else keep all, weight 1
MIN_OI_SIDE = 2               # ...and per wing (puts K<F / calls K>=F): marks kept on a wing with < 2 OI quotes
SLN_MAX_SDT = 1.5             # cap on each lognormal component's sigma*sqrt(T)
SLN_MIN_GAP = 0.15            # support floor at least 15c (+2c) below the lowest quoted strike
W_EPS = 0.003
KMAX = 2.0
PIN_LO, PIN_HI = 1.15, 1.21
GRID_LO, GRID_HI, GRID_DK = -3.0, 10.0, 0.005  # grid relative to F (Nov 2018 had $2-strike calls worth $0.9)
SQ2PI = np.sqrt(2 * np.pi)
USABLE_RMSE = 0.02                 # usable if price RMSE <= max(USABLE_RMSE, 5% of the ATM option price)


# ----------------------------------------------------------------------------------------------- data
def _cut_tag(cut):
    return pd.Timestamp(cut).strftime("%Y%m%d")


@lru_cache(maxsize=2)
def load_options(cut=None):
    """G4X cm=3 rows dated <= cut (default DATA_CUTOFF). Filtered chunk by chunk on load."""
    cut = DATA_CUTOFF if cut is None else pd.Timestamp(cut)
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"g4x_cm3_le{_cut_tag(cut)}.pkl"
    if p.exists():
        return pd.read_pickle(p)
    parts = []
    for ch in pd.read_csv(OPT_FILE, chunksize=500_000, parse_dates=["date"]):
        ch = ch[(ch.date <= cut) & (ch.root == "G4X") & (ch.cm == 3)]
        if len(ch):
            parts.append(ch)
    o = pd.concat(parts)
    o = o[o.settle > 0].copy()
    o["Y"] = o.cy.astype(int)
    o["K"] = o.k_raw / 1000.0
    o = o[["date", "Y", "cp", "K", "settle", "volume", "oi"]].sort_values(["Y", "date", "cp", "K"])
    o = o.reset_index(drop=True)
    assert o.date.max() <= cut
    o.to_pickle(p)
    return o


@lru_cache(maxsize=2)
def load_futures(cut=None):
    """Daily H-J spread settles per delivery year Y, rows dated <= cut."""
    cut = DATA_CUTOFF if cut is None else pd.Timestamp(cut)
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"hj_le{_cut_tag(cut)}.pkl"
    if p.exists():
        return pd.read_pickle(p)
    c = pd.read_csv(FUT_FILE, parse_dates=["date"], usecols=["date", "contract_year", "contract_month", "settle"])
    c = c[(c.date <= cut) & c.contract_month.isin([3, 4])].dropna(subset=["settle"])
    px = c.pivot_table(index=["contract_year", "date"], columns="contract_month", values="settle").dropna()
    px.columns = ["H", "J"]
    px["S"] = px.H - px.J
    px.index.names = ["Y", "date"]
    assert px.index.get_level_values("date").max() <= cut
    px.to_pickle(p)
    return px


# ------------------------------------------------------------------------------------------- calendar
def _holidays(y0, y1):
    from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, USMartinLutherKingJr,
                                        USPresidentsDay, USMemorialDay, USLaborDay, USThanksgivingDay,
                                        Holiday, nearest_workday)

    class NymexCal(AbstractHolidayCalendar):
        rules = [Holiday("NewYear", month=1, day=1, observance=nearest_workday), USMartinLutherKingJr,
                 USPresidentsDay, GoodFriday, USMemorialDay,
                 Holiday("July4", month=7, day=4, observance=nearest_workday), USLaborDay, USThanksgivingDay,
                 Holiday("Christmas", month=12, day=25, observance=nearest_workday)]
    return NymexCal().holidays(pd.Timestamp(y0, 1, 1), pd.Timestamp(y1, 12, 31)).values.astype("datetime64[D]")


def option_expiry(Y):
    """G4X March expiry: trading day before the H LTD (3 business days before 1 March of Y)."""
    hol = _holidays(Y, Y)
    ltd = np.busday_offset(np.datetime64(f"{Y}-03-01"), -3, roll="forward", holidays=hol)
    return pd.Timestamp(np.busday_offset(ltd, -1, roll="backward", holidays=hol))


def trading_T(date, Y):
    """Trading days from date to option expiry / 252 (floored at 1 day)."""
    hol = _holidays(Y - 2, Y)
    n = np.busday_count(np.datetime64(pd.Timestamp(date).date()), np.datetime64(option_expiry(Y).date()),
                        holidays=hol)
    return max(int(n), 1) / 252.0


# ------------------------------------------------------------------------------------------ Bachelier
def bach_call(F, K, v):
    """Bachelier call, v = sigma_N * sqrt(T)."""
    v = np.maximum(v, 1e-9)
    d = (F - K) / v
    return (F - K) * ndtr(d) + v * np.exp(-0.5 * d * d) / SQ2PI


def bach_otm(F, K, v, isc):
    c = bach_call(F, K, v)
    return np.where(isc, c, c - (F - K))


def bach_iv(px, F, K, T, isc, lo=1e-4, hi=20.0, n=60):
    """Normal vol (per sqrt yr) from OTM prices, vectorised bisection."""
    px, K, isc = np.atleast_1d(px).astype(float), np.atleast_1d(K).astype(float), np.atleast_1d(isc)
    a, b = np.full(len(px), lo), np.full(len(px), hi)
    sT = np.sqrt(T)
    for _ in range(n):
        m = 0.5 * (a + b)
        up = bach_otm(F, K, m * sT, isc) < px
        a, b = np.where(up, m, a), np.where(up, b, m)
    out = 0.5 * (a + b)
    out[out > hi * 0.999] = np.nan
    return out


# ------------------------------------------------------------------------------------------- quotes
def parity_F(day):
    c = day[day.cp == "C"].groupby("K").settle.first()
    p = day[day.cp == "P"].groupby("K").settle.first()
    j = c.index.intersection(p.index)
    if len(j) == 0:
        return np.nan
    est = j.values + (c[j].values - p[j].values)
    wt = 1.0 / (np.abs(c[j].values - p[j].values) + 0.01)     # strikes nearest the money count most
    return float(np.sum(est * wt) / np.sum(wt))


def prepare_quotes(day, F, T, prev=None, zero_oi_weight=None):
    """OTM quote set for one day. `day`: rows (cp, K, settle[, oi]); `prev`: previous quote day's rows (for OI).
    Returns dict of arrays K, px, isc, C (call-equivalent), w, oi_ok, pinned, plus day-level flags."""
    zw = ZERO_OI_WEIGHT if zero_oi_weight is None else zero_oi_weight
    day = day[(np.abs(day.K - F) <= KMAX) & (day.settle > 0)]
    c = day[(day.cp == "C") & (day.K >= F)].groupby("K").settle.first()
    p = day[(day.cp == "P") & (day.K < F)].groupby("K").settle.first()
    K = np.concatenate([c.index.values, p.index.values]).astype(float)
    px = np.concatenate([c.values, p.values]).astype(float)
    isc = np.concatenate([np.ones(len(c), bool), np.zeros(len(p), bool)])
    o = np.argsort(K)
    K, px, isc = K[o], px[o], isc[o]
    # tick-level wings: keep at most the innermost 0.001 settle on each side
    keep = px >= 0.002
    for side in (isc, ~isc):
        idx = np.where(side & (px < 0.002))[0]
        if len(idx):
            inner = idx[np.argmin(np.abs(K[idx] - F))]
            keep[inner] = True
    K, px, isc = K[keep], px[keep], isc[keep]
    # lagged open interest: either side at the strike, previous quote day
    if prev is not None and len(prev) and "oi" in prev:
        oi_prev = prev.groupby("K").oi.sum(min_count=1).fillna(0.0)
        oi_ok = np.array([oi_prev.get(k, 0.0) > 0 for k in K])
        have_oi = True
    else:
        oi_ok = np.ones(len(K), bool)
        have_oi = False
    iv = bach_iv(px, F, K, T, isc) if len(K) else np.array([])
    pinned = (iv >= PIN_LO) & (iv <= PIN_HI)
    usable = px >= 0.002
    nu = usable.sum()
    pin_share = float(pinned[usable].mean()) if nu else np.nan
    ivu = iv[usable & np.isfinite(iv)]
    flat = len(ivu) >= 2 and (np.percentile(ivu, 75) - np.percentile(ivu, 25)) < 0.03 \
        and abs(np.median(ivu) - 1.18) < 0.05
    default_settle = bool((nu > 0 and pin_share >= 0.5) or flat)
    drop = pinned & ~oi_ok
    marks_kept = False
    if zw == 0.0:
        if (oi_ok & ~drop).sum() >= MIN_OI_QUOTES:
            for side in (isc, ~isc):                   # drop exchange marks wing by wing
                if (side & oi_ok & ~drop).sum() >= MIN_OI_SIDE:
                    drop = drop | (side & ~oi_ok)
                elif (side & ~oi_ok & ~drop).any():
                    marks_kept = True
            zw = 1.0                                    # marks that survive (thin wing) keep full weight
        else:
            zw, marks_kept = 1.0, bool((~oi_ok & ~drop).any())   # too few genuine quotes: keep marks
    K, px, isc, oi_ok, iv = K[~drop], px[~drop], isc[~drop], oi_ok[~drop], iv[~drop]
    w = 1.0 / (px + W_EPS) * np.where(oi_ok, 1.0, zw)
    C = np.where(isc, px, px + (F - K))
    return {"K": K, "px": px, "isc": isc, "C": C, "w": w, "oi_ok": oi_ok, "iv": iv,
            "default_settle": default_settle, "pin_share": pin_share, "n_dropped": int(drop.sum()),
            "marks_kept": marks_kept,
            "zero_oi": bool(have_oi and len(K) and not oi_ok.any()),
            "zero_oi_share": float((~oi_ok).mean()) if (have_oi and len(K)) else np.nan}


# --------------------------------------------------------------------------------------------- fits
class _Fit:
    """Common interface: call(K) prices, cdf/pdf on grid."""
    method = "?"
    low_info = False

    def otm(self, K, isc, F):
        c = self.call(K)
        return np.where(isc, c, c - (F - K))

    def density(self, grid):           # default: numerical BL from the call function
        h = grid[1] - grid[0]
        ext = np.concatenate([[grid[0] - h], grid, [grid[-1] + h]])
        C = self.call(ext)
        cdf = 1.0 + (C[2:] - C[:-2]) / (2 * h)
        pdf = (C[2:] - 2 * C[1:-1] + C[:-2]) / h ** 2
        return cdf, pdf


def _starts_atm_v(q, F, T):
    i = np.argmin(np.abs(q["K"] - F))
    v = q["iv"][i] if np.isfinite(q["iv"][i]) else 1.0
    return max(v, 0.05) * np.sqrt(T)


# ---- normal-SVI in total variance (default)
class SVIFit(_Fit):
    method = "svi"

    def __init__(self, q, F, T, grid=None, penalty=True):
        self.F, self.T = F, T
        K, px, isc, w = q["K"], q["px"], q["isc"], q["w"]
        v0 = _starts_atm_v(q, F, T)
        n = len(K)
        self.npar = 5 if n >= 5 else (2 if n >= 2 else 1)
        x = K - F
        # SVI vertex m is kept inside the quoted range (no vertex in the extrapolated region)
        self.xc = 0.5 * (x.min() + x.max()) if n else 0.0
        self.xr = max(0.5 * (x.max() - x.min()), 0.05) if n else 0.05
        gpen = F + np.arange(GRID_LO, GRID_HI, 0.04)

        def model(p):
            return bach_otm(F, K, np.sqrt(self._w(p, K - F)), isc)

        def res(p):
            r = (model(p) - px) * w
            if penalty and self.npar == 5:
                h = 0.01
                c = bach_call(F, gpen, np.sqrt(self._w(p, gpen - F)))
                cm = bach_call(F, gpen - h, np.sqrt(self._w(p, gpen - h - F)))
                cp_ = bach_call(F, gpen + h, np.sqrt(self._w(p, gpen + h - F)))
                neg = np.minimum((cp_ - 2 * c + cm) / h ** 2, 0.0)
                r = np.concatenate([r, 100.0 * neg * 0.04])
            return r

        best = None
        if self.npar == 5:
            lo = np.array([np.log(v0 ** 2) - 8, np.log(v0 ** 2) - 10, -2.0, -3.0, -7.0])
            hi = np.array([np.log(v0 ** 2) + 4, np.log(v0 ** 2) + 6, 2.0, 3.0, 1.0])
            m0 = np.clip(-self.xc / self.xr, -0.9, 0.9)      # vertex start near F
            for pb, pr, ps in [(-1.0, 0.3, -1.5), (0.0, 0.6, -1.0), (-2.0, 0.0, -2.0), (-0.5, 1.2, -0.5)]:
                p0 = np.array([np.log(v0 ** 2 * 0.8), pb + np.log(v0 ** 2), pr, np.arctanh(m0), ps])
                p0 = np.clip(p0, lo + 1e-6, hi - 1e-6)
                try:
                    s = least_squares(res, p0, bounds=(lo, hi), method="trf", max_nfev=800, x_scale="jac")
                except Exception:
                    continue
                if best is None or s.cost < best.cost:
                    best = s
        elif self.npar == 2:
            # 2-parameter skewed form: w = v^2 (1 + g x) clipped - level + linear skew in variance
            best = least_squares(res, np.array([np.log(v0 ** 2), 0.5]), method="trf", max_nfev=400)
        else:
            best = least_squares(res, np.array([np.log(v0 ** 2)]), method="trf", max_nfev=200)
        self.p = best.x
        self.low_info = self.npar < 5
        err = model(self.p) - px
        self.rmse = float(np.sqrt(np.mean(err ** 2)))

    def _w(self, p, x):
        if self.npar == 5:
            la, lb, r, pm, ls = p
            b, rho, s = np.exp(lb), 0.95 * np.tanh(r), np.exp(ls)
            m = self.xc + self.xr * np.tanh(pm)
            a = np.exp(la) - b * s * np.sqrt(1 - rho ** 2)
            return np.maximum(a + b * (rho * (x - m) + np.sqrt((x - m) ** 2 + s ** 2)), 1e-8)
        if self.npar == 2:
            lv, g = p
            # smooth positive version of v^2 (1 + g x): softplus keeps variance > 0, growth linear in |x|
            z = g * x
            return np.exp(lv) * (1 + np.logaddexp(0, 4 * z) / 4 - np.log(2) / 4)
        return np.full_like(np.asarray(x, float), np.exp(p[0]))

    def call(self, K):
        K = np.asarray(K, float)
        return bach_call(self.F, K, np.sqrt(self._w(self.p, K - self.F)))


# ---- quadratic normal vol with linear extension
class QuadFit(_Fit):
    method = "quad"

    def __init__(self, q, F, T, grid=None):
        self.F, self.T = F, T
        K, px, isc, w = q["K"], q["px"], q["isc"], q["w"]
        x = K - F
        self.xlo, self.xhi = (x.min(), x.max()) if len(x) else (0.0, 0.0)
        v0 = _starts_atm_v(q, F, T) / np.sqrt(T)
        self.sig_floor = 0.25 * v0
        n = len(K)
        self.npar = 3 if n >= 3 else 1
        self.low_info = self.npar < 3

        def res(p):
            return (bach_otm(F, K, self._sig(p, x) * np.sqrt(T), isc) - px) * w
        p0 = np.zeros(self.npar)
        p0[0] = v0
        s = least_squares(res, p0, method="trf", max_nfev=600)
        self.p = s.x
        self.rmse = float(np.sqrt(np.mean((res(self.p) / w) ** 2)))

    def _poly(self, p, x):
        a = p[0]
        b = p[1] if len(p) > 1 else 0.0
        c = p[2] if len(p) > 2 else 0.0
        return a + b * x + c * x * x, b + 2 * c * x

    def _sig(self, p, x):
        x = np.asarray(x, float)
        xc = np.clip(x, self.xlo, self.xhi)
        val, slope = self._poly(p, xc)
        out = val + slope * (x - xc)          # C1 linear extension outside the quoted range
        return np.maximum(out, self.sig_floor)

    def call(self, K):
        K = np.asarray(K, float)
        return bach_call(self.F, K, self._sig(self.p, K - self.F) * np.sqrt(self.T))


# ---- baseline normal mixture (v5_implied_storage/rnd.py logic)
class MixtureFit(_Fit):
    method = "mixture"

    def __init__(self, q, F, T, grid=None, warm=None, starts=None, seed=0):
        self.F = F
        K, px, isc, w = q["K"], q["px"], q["isc"], q["w"]
        nq = len(K)
        n = 3 if nq >= 15 else (2 if nq >= 5 else 1)
        self.n, self.low_info = n, nq < 3
        s0 = max(_starts_atm_v(q, F, T), 0.02)

        def res(p):
            return (self.otm_th(self._unpack(p, n), K, isc) - px) * w
        rng = np.random.default_rng(seed)
        nst = (1 if n == 1 else 6) if starts is None else starts
        best = None
        cands = []
        if warm is not None and len(warm) == 3 * n - 1:
            cands.append(np.asarray(warm))
        for r in range(nst):
            if n == 1:
                cands.append(np.array([np.log(s0)]))
            else:
                lw = rng.normal(-1.5, 0.8, n - 1)
                m0 = np.concatenate([[F], F + rng.uniform(0.05, 0.8, n - 1)])
                ls = np.log(np.concatenate([[s0 * rng.uniform(0.4, 1.0)], s0 * rng.uniform(1.5, 6, n - 1)]))
                cands.append(np.concatenate([lw, m0, ls]))
        for p0 in cands:
            try:
                s = least_squares(res, p0, method="trf", max_nfev=2000)
            except Exception:
                continue
            if best is None or s.cost < best.cost:
                best = s
        self.p = best.x
        self.theta = self._unpack(best.x, n)
        self.rmse = float(np.sqrt(np.mean((res(self.p) / w) ** 2)))

    def _unpack(self, p, n):
        F = self.F
        if n == 1:
            return np.array([1.0]), np.array([F]), np.array([np.exp(p[0])])
        z = np.concatenate([[0.0], p[: n - 1]])
        w = np.exp(z - z.max())
        w /= w.sum()
        m = p[n - 1: 2 * n - 1]
        m = m - (w @ m) + F
        s = np.exp(p[2 * n - 1: 3 * n - 1])
        return w, m, s

    def otm_th(self, th, K, isc):
        w, m, s = th
        c = bach_call(m[None, :], np.asarray(K, float)[:, None], s[None, :]) @ w
        return np.where(isc, c, c - (self.F - K))

    def call(self, K):
        w, m, s = self.theta
        return bach_call(m[None, :], np.asarray(K, float)[:, None], s[None, :]) @ w

    def density(self, grid):
        w, m, s = self.theta
        z = (grid[:, None] - m[None, :]) / s[None, :]
        return ndtr(z) @ w, (np.exp(-0.5 * z * z) / (SQ2PI * s[None, :])) @ w


# ---- shifted-lognormal mixture
class SLNFit(_Fit):
    method = "sln"

    def __init__(self, q, F, T, grid=None):
        self.F, self.T = F, T
        K, px, isc, w = q["K"], q["px"], q["isc"], q["w"]
        nq = len(K)
        self.n = 2 if nq >= 5 else 1
        self.low_info = nq < 3
        self.kfloor = min(K.min() if nq else F, F) - 0.02
        v0 = _starts_atm_v(q, F, T)

        def res(p):
            return (self.otm(K, isc, F, p) - px) * w
        best = None
        # bounds (added after the check-window audit, see report): support floor >= SLN_MIN_GAP below the lowest strike,
        # component sigma*sqrt(T) <= SLN_MAX_SDT - stops a component collapsing into an atom at the floor
        # when one wing has a single quote (2016-11-11: 74% of mass at the floor).
        vlo, vhi = np.log(0.005), np.log(SLN_MAX_SDT / np.sqrt(T))
        if self.n == 1:
            lo, hi = np.array([np.log(SLN_MIN_GAP), vlo]), np.array([np.log(20.0), vhi])
        else:
            lo = np.array([np.log(SLN_MIN_GAP), -6, -6, -6, vlo, vlo])
            hi = np.array([np.log(20.0), 6, 6, 6, vhi, vhi])
        for lam0 in (0.2, 0.5, 1.5, 3.0):
            if self.n == 1:
                p0 = np.array([np.log(lam0), np.log(v0 / (F - self.kfloor + lam0) / np.sqrt(T))])
            else:
                sg = v0 / (F - self.kfloor + lam0) / np.sqrt(T)
                p0 = np.array([np.log(lam0), -1.5, 0.0, 0.3, np.log(sg * 0.8), np.log(sg * 2.5)])
            p0 = np.clip(p0, lo + 1e-6, hi - 1e-6)
            try:
                s = least_squares(res, p0, bounds=(lo, hi), method="trf", max_nfev=1500)
            except Exception:
                continue
            if best is None or s.cost < best.cost:
                best = s
        self.p = best.x
        self.rmse = float(np.sqrt(np.mean((res(self.p) / w) ** 2)))

    def _th(self, p):
        F = self.F
        shift = -self.kfloor + np.exp(p[0])          # support is S > -shift, below every quoted strike
        if self.n == 1:
            return shift, np.array([1.0]), np.array([F + shift]), np.array([np.exp(p[1])])
        z = np.array([0.0, p[1]])
        wt = np.exp(z - z.max())
        wt /= wt.sum()
        # component forwards (shifted), positive, mean-preserving
        G = F + shift
        r = np.exp(np.array([p[2], p[3]]))
        r = r / (wt @ r)
        return shift, wt, G * r, np.exp(np.array([p[4], p[5]]))

    def _call_p(self, K, p):
        shift, wt, G, sg = self._th(p)
        Kx = np.maximum(np.asarray(K, float)[:, None] + shift, 1e-9)
        sT = sg[None, :] * np.sqrt(self.T)
        d1 = (np.log(G[None, :] / Kx) + 0.5 * sT ** 2) / sT
        c = G[None, :] * ndtr(d1) - Kx * ndtr(d1 - sT)
        Kraw = np.asarray(K, float)[:, None] + shift
        c = np.where(Kraw > 0, c, G[None, :] - Kraw)      # below the support: call = forward - strike
        return c @ wt

    def otm(self, K, isc, F, p=None):
        c = self._call_p(K, self.p if p is None else p)
        return np.where(isc, c, c - (F - K))

    def call(self, K):
        return self._call_p(K, self.p)

    def density(self, grid):
        shift, wt, G, sg = self._th(self.p)
        x = np.maximum(grid[:, None] + shift, 1e-12)
        sT = sg[None, :] * np.sqrt(self.T)
        z = (np.log(x / G[None, :]) + 0.5 * sT ** 2) / sT
        pdf = (np.exp(-0.5 * z * z) / (SQ2PI * sT * x)) @ wt
        cdf = ndtr(z) @ wt
        pdf[grid + shift <= 0] = 0.0
        return cdf, pdf


# ---- model-free BL with convex projection
class ConvexFit(_Fit):
    method = "convex"

    def __init__(self, q, F, T, grid=None):
        self.F, self.T = F, T
        K, C, w = q["K"], q["C"], q["w"]
        n = len(K)
        self.low_info = n < 3
        if n < 2:
            raise ValueError("convex needs >= 2 strikes")
        dK = np.diff(K)
        # params: C1, u_0..u_{n-2} >= 0 ; slopes s_j = -1 + cumsum(u)_j ; C_i = C1 + sum_{j<i} s_j dK_j
        A = np.zeros((n, n))
        A[:, 0] = 1.0
        for i in range(1, n):
            for j in range(i):            # s_j contributes dK_j to C_i ; s_j = -1 + sum_{l<=j} u_l
                A[i, 1: j + 2] += dK[j]
        b0 = np.array([-np.sum(dK[:i]) for i in range(n)])  # the "-1" part of slopes
        intr = np.maximum(F - K, 0.0)
        W = w

        def price(z):
            return A @ z + b0

        def obj(z):
            r = (price(z) - C) * W
            return 0.5 * r @ r

        def grad(z):
            return A.T @ ((price(z) - C) * W * W)
        # initial: isotonic-ish from raw slopes
        s_raw = np.clip(np.diff(C) / dK, -1, 0)
        s_mon = np.maximum.accumulate(s_raw)
        u0 = np.diff(np.concatenate([[-1.0], s_mon]))
        z0 = np.concatenate([[C[0]], np.maximum(u0, 0)])
        cons = [{"type": "ineq", "fun": lambda z: 1.0 - np.sum(z[1:]), "jac": lambda z: np.concatenate([[0], -np.ones(n - 1)])},
                {"type": "ineq", "fun": lambda z: price(z) - intr, "jac": lambda z: A}]
        bnds = [(None, None)] + [(0, 1)] * (n - 1)
        sol = minimize(obj, z0, jac=grad, bounds=bnds, constraints=cons, method="SLSQP",
                       options={"maxiter": 400, "ftol": 1e-14})
        z = sol.x
        Cf = price(z)
        s = np.minimum(-1 + np.cumsum(z[1:]), 0.0)
        self.Kq, self.Cq, self.slopes = K, Cf, s
        self.rmse = float(np.sqrt(np.mean((Cf - C) ** 2)))
        # CDF knots at strike midpoints; tails exponential
        mid = 0.5 * (K[1:] + K[:-1])
        cdf_mid = 1.0 + s
        pR = max(-s[-1], 1e-6)                       # P(S > ~K_n)
        CR = max(Cf[-1], 1e-7)                        # call at K_n
        self.betaR = float(np.clip(CR / pR, 0.005, 3.0))
        PL = max(Cf[0] - (F - K[0]), 1e-7)            # put at K_1
        qL = max(1.0 + s[0], 1e-6)                    # P(S < ~K_1)
        self.betaL = float(np.clip(PL / qL, 0.005, 3.0))
        self.kn, self.k1, self.pR, self.qL = K[-1], K[0], pR, qL
        xs = np.concatenate([[K[0]], mid, [K[-1]]])
        ys = np.concatenate([[min(qL, cdf_mid[0])], cdf_mid, [max(1 - pR, cdf_mid[-1])]])
        ys = np.maximum.accumulate(np.clip(ys, 0, 1))
        xs, idx = np.unique(xs, return_index=True)
        self._pch = PchipInterpolator(xs, ys[idx])

    def cdf_f(self, x):
        x = np.asarray(x, float)
        out = np.empty_like(x)
        lo, hi = x < self.k1, x > self.kn
        mid = ~(lo | hi)
        out[mid] = self._pch(x[mid])
        out[lo] = self.qL * np.exp((x[lo] - self.k1) / self.betaL)
        out[hi] = 1 - self.pR * np.exp(-(x[hi] - self.kn) / self.betaR)
        return np.clip(out, 0, 1)

    def density(self, grid):
        cdf = self.cdf_f(grid)
        pdf = np.gradient(cdf, grid)
        return cdf, np.maximum(pdf, 0)

    def call(self, K):
        K = np.atleast_1d(np.asarray(K, float))
        g = self.F + np.arange(GRID_LO, GRID_HI, GRID_DK)
        sv = 1 - self.cdf_f(g)
        # C(K) = int_K^inf (1 - CDF); integrate on the grid
        cum = np.concatenate([[0], np.cumsum(0.5 * (sv[1:] + sv[:-1]) * np.diff(g))])
        tot = cum[-1] + sv[-1] * self.betaR
        return tot - np.interp(K, g, cum)


METHODS = {"svi": SVIFit, "quad": QuadFit, "mixture": MixtureFit, "convex": ConvexFit, "sln": SLNFit}


def fit_quotes(q, F, T, method=DEFAULT_METHOD, **kw):
    if len(q["K"]) == 0:
        raise ValueError("no quotes")
    if method in ("convex", "svi") and len(q["K"]) < (3 if method == "convex" else 5):
        method = "sln"            # documented fallback on thin days: 1-component displaced lognormal (2 params)
    return METHODS[method](q, F, T, **kw)


def density_from_fit(fit, F, grid=None):
    """CDF from the fit's own 1 + dC/dK (exact even if some mass lies beyond the grid); pdf clipped at 0.
    Negative density, if any, is removed by a monotone (running-max) repair of the CDF and reported."""
    grid = F + np.arange(GRID_LO, GRID_HI + 1e-9, GRID_DK) if grid is None else grid
    cdf_raw, pdf = fit.density(grid)
    h = grid[1] - grid[0]
    neg_mass = float(-np.minimum(pdf, 0).sum() * h)
    cdf = np.maximum.accumulate(np.clip(cdf_raw, 0.0, 1.0))
    pdf = np.maximum(pdf, 0.0)
    if neg_mass > 1e-6:
        pdf = np.gradient(cdf, grid)
    tail_out = float(cdf[0] + 1.0 - cdf[-1])          # mass beyond the grid (Nov 2018: right tail > F+10)
    return grid, pdf, cdf, neg_mass, tail_out


# ---------------------------------------------------------------------------------------------- API
def _day_rows(date, Y, cut):
    o = load_options(DATA_CUTOFF)
    oy = o[(o.Y == Y) & (o.date <= cut)]
    day = oy[oy.date == date]
    prevd = oy.date[oy.date < date]
    prev = oy[oy.date == prevd.max()] if len(prevd) else None
    return day, prev


def spread_density(date, winter_year, quotes=None, method=None, F=None, zero_oi_weight=None, prev_quotes=None):
    """Risk-neutral density of S_T = (H - J at G4X March option expiry) as of `date`.

    date        : settlement date (str or Timestamp). Only data dated <= min(date, DATA_CUTOFF) are read.
    winter_year : March delivery year Y (winter Nov Y-1 .. Feb Y).
    quotes      : optional DataFrame (cp in {'C','P'}, K, settle[, oi]) to use instead of the file.
    method      : 'svi' (default) | 'quad' | 'mixture' | 'convex' | 'sln'.
    F           : optional futures spread; default = H-J settle on date (parity-implied F if missing).
    Returns dict: grid, pdf, cdf (array on grid), cdf_f (callable), prob_above (callable), quantile (callable),
    F, F_parity, T, method, n_quotes, flags {...}, usable (bool).
    """
    date = pd.Timestamp(date).normalize()
    Y = int(winter_year)
    method = DEFAULT_METHOD if method is None else method
    cut = min(date, DATA_CUTOFF)
    T = trading_T(date, Y)
    if quotes is None:
        day, prev = _day_rows(date, Y, cut) if date <= DATA_CUTOFF else (pd.DataFrame(), None)
    else:
        day, prev = quotes, prev_quotes
    F_par = parity_F(day) if len(day) else np.nan
    F_src, F_fut = "given", np.nan
    if F is None:
        fut = load_futures(DATA_CUTOFF)
        try:
            F_fut = float(fut.loc[(Y, date), "S"]) if date <= cut else np.nan
        except KeyError:
            F_fut = np.nan
        # option settles are internally consistent with their own parity forward; on blow-up days (Nov 2018)
        # the futures settle differs by up to 6.5c. Use parity unless missing or implausibly far away.
        if np.isfinite(F_par) and (not np.isfinite(F_fut) or abs(F_par - F_fut) < 0.15):
            F, F_src = F_par, "parity"
        else:
            F, F_src = F_fut, "futures"
    out = {"date": date, "Y": Y, "F": F, "F_parity": F_par, "F_futures": F_fut, "F_source": F_src, "T": T,
           "method": method,
           "n_quotes": 0, "usable": False, "flags": {}}
    if not np.isfinite(F) or len(day) == 0:
        out["flags"] = {"no_data": True}
        return _attach_callables(out, None)
    q = prepare_quotes(day, F, T, prev=prev, zero_oi_weight=zero_oi_weight)
    out["n_quotes"] = int(len(q["K"]))
    if len(q["K"]) == 0:
        out["flags"] = {"no_data": True, "default_settle": q["default_settle"]}
        return _attach_callables(out, None)
    fit = fit_quotes(q, F, T, method)
    grid, pdf, cdf, neg_mass, tail_out = density_from_fit(fit, F)
    klo = grid[0]
    mean = float(klo + fit.call(np.array([klo]))[0])  # E[S_T] = K + C(K) for K below the support
    cdf_f = lambda x: np.interp(x, grid, cdf, left=0.0, right=1.0)  # noqa: E731
    ext = float(cdf_f(q["K"].min()) + 1 - cdf_f(q["K"].max()))
    flags = {"default_settle": q["default_settle"], "pin_share": q["pin_share"],
             "n_dropped": q["n_dropped"], "marks_kept": q["marks_kept"], "zero_oi": q["zero_oi"], "zero_oi_share": q["zero_oi_share"],
             "extrapolated_share": ext, "low_info": bool(fit.low_info or len(q["K"]) < 3),
             "arb_clipped": neg_mass > 1e-4, "neg_mass": neg_mass, "rmse": fit.rmse, "mean_err": mean - F,
             "parity_gap": (F_fut - F_par) if np.isfinite(F_par) and np.isfinite(F_fut) else np.nan,
             "kmin": float(q["K"].min()), "kmax": float(q["K"].max()), "fit_method": fit.method}
    flags["tail_beyond_grid"] = tail_out
    atm_px = float(q["px"][np.argmin(np.abs(q["K"] - F))])
    flags["atm_px"] = atm_px
    usable = bool(abs(mean - F) < 0.02 and fit.rmse <= max(USABLE_RMSE, 0.05 * atm_px) and neg_mass < 0.02
                  and np.isfinite(cdf).all())
    out.update({"grid": grid, "pdf": pdf, "cdf": cdf, "flags": flags, "usable": usable, "fit": fit, "quotes": q,
                "method": fit.method})
    return _attach_callables(out, cdf_f)


def _attach_callables(out, cdf_f):
    if cdf_f is None:
        nanf = lambda x: np.full(np.shape(x), np.nan) if np.ndim(x) else np.nan  # noqa: E731
        out.update({"grid": np.array([]), "pdf": np.array([]), "cdf": np.array([]), "cdf_f": nanf,
                    "prob_above": nanf, "quantile": nanf})
        return out
    grid, cdf = out["grid"], out["cdf"]
    out["cdf_f"] = cdf_f
    out["prob_above"] = lambda k: 1.0 - cdf_f(k)

    def quantile(p):                                   # generalised inverse: min{x : CDF(x) >= p}
        p = np.atleast_1d(np.asarray(p, float))
        j = np.clip(np.searchsorted(cdf, p, side="left"), 1, len(grid) - 1)
        c0, c1 = cdf[j - 1], cdf[j]
        t = np.where(c1 > c0, (p - c0) / np.where(c1 > c0, c1 - c0, 1), 1.0)
        return grid[j - 1] + np.clip(t, 0, 1) * (grid[j] - grid[j - 1])
    out["quantile"] = quantile
    return out
