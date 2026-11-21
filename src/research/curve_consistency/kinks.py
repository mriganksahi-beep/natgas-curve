"""Part 2: smooth fair curve per day + butterfly benchmark.

Fair curve (per trading day t, contracts at positions 1..18 with > 5 trading days to expiry):
    log F(t, m) = a_t + b_t x + c_t x^2 + lam_t * S[calmonth(m)] + resid      x = months-to-delivery / 12
S (12 numbers, sum zero) is re-estimated each 1 January from the previous N_YEARS of curves only
(walk-forward), by alternating least squares (day quadratics <-> pooled month effects).
Variant A: lam_t = 1.  Variant B: lam_t free (seasonal amplitude breathes with storage).
Kink = resid. Fly-kink = resid(m) - (resid(m-1) + resid(m+1)) / 2.

Benchmark: log butterfly L = log F(m) - (log F(m-1) + log F(m+1)) / 2 minus its own trailing
norm: the mean of L for the same (centre calendar month, months-to-delivery) over the previous
N_YEARS calendar years (same calendar month of observation).
"""
import numpy as np
import pandas as pd

from .data import NPOS

N_YEARS = 3


def _design(x, free_lam, s):
    cols = [np.ones_like(x), x, x ** 2]
    if free_lam:
        cols.append(s)
    return np.column_stack(cols)


def fit_day(x, y, s, free_lam):
    """Return residuals of y (log F) on quadratic in x (+ lam * s, or with s subtracted)."""
    if free_lam:
        X = _design(x, True, s)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ beta, beta
    X = _design(x, False, s)
    beta, *_ = np.linalg.lstsq(X, y - s, rcond=None)
    return y - s - X @ beta, beta


def estimate_shape(sub, iters=25):
    """Pooled month effects given day-specific quadratics (lam=1). sub: rows with di, x, ly, cm."""
    S = np.zeros(12)
    groups = [g for _, g in sub.groupby("di")]
    for _ in range(iters):
        res_all = []
        for g in groups:
            r, _ = fit_day(g.x.values, g.ly.values, S[g.cm.values - 1], False)
            res_all.append(r)
        res = np.concatenate(res_all)
        cm = np.concatenate([g.cm.values for g in groups])
        upd = np.bincount(cm - 1, weights=res, minlength=12) / np.bincount(cm - 1, minlength=12)
        S = S + upd
        S -= S.mean()
        if np.abs(upd).max() < 1e-6:
            break
    return S


def prepare(d):
    f = d[(d.pos <= NPOS) & (d.dte > 5)].copy()
    cur = f.date.dt.year * 12 + f.date.dt.month - 1
    f["tau"] = f.k - cur                      # months to delivery
    f["x"] = f.tau / 12.0
    f["ly"] = np.log(f.settle)
    f["cm"] = f.contract_month
    return f


def fit_all(d):
    """Walk-forward shapes per year and residuals for variants A and B."""
    f = prepare(d)
    f["year"] = f.date.dt.year
    shapes = {}
    for Y in range(2011, 2022):
        lo = pd.Timestamp(f"{Y - N_YEARS}-01-01")
        sub = f[(f.date >= lo) & (f.date < f"{Y}-01-01")]
        sub = sub[sub.di.isin(sub.di.unique()[::5])]          # every 5th day is plenty for 12 numbers
        shapes[Y] = estimate_shape(sub)
    f = f[f.year >= 2011].copy()
    rA = np.full(len(f), np.nan)
    rB = np.full(len(f), np.nan)
    lam = {}
    idx = 0
    f = f.sort_values(["di", "k"]).reset_index(drop=True)
    starts = f.groupby("di").indices
    for di, ii in starts.items():
        g = f.iloc[ii]
        if len(g) < 8:
            continue
        s = shapes[g.year.iat[0]][g.cm.values - 1]
        rA[ii], _ = fit_day(g.x.values, g.ly.values, s, False)
        rB[ii], b = fit_day(g.x.values, g.ly.values, s, True)
        lam[di] = b[3]
    f["kA"] = rA
    f["kB"] = rB
    return f, pd.DataFrame(shapes, index=range(1, 13)), pd.Series(lam)


def add_flies(f):
    """Per (date, centre contract): $ fly, log fly, fitted fly-kinks, benchmark fly-kink."""
    f = f.sort_values(["di", "k"])
    g = f.groupby("di")
    prv, nxt = g.shift(1), g.shift(-1)
    ok = (prv.k == f.k - 1) & (nxt.k == f.k + 1)
    fl = pd.DataFrame({
        "date": f.date, "di": f.di, "k": f.k, "cm": f.cm, "tau": f.tau, "pos": f.pos,
        "F": f.settle,
        "B": f.settle - (prv.settle + nxt.settle) / 2,                         # $/MMBtu
        "L": f.ly - (prv.ly + nxt.ly) / 2,
        "fA": f.kA - (prv.kA + nxt.kA) / 2,
        "fB": f.kB - (prv.kB + nxt.kB) / 2,
        "kA": f.kA, "kB": f.kB,
        "dte_min": np.minimum(np.minimum(prv.dte, nxt.dte), f.dte),
        "vol_min": np.minimum(np.minimum(prv.volume, nxt.volume), f.volume),
        "oi_min": np.minimum(np.minimum(prv.open_interest, nxt.open_interest), f.open_interest),
    })[ok].copy()
    # trailing seasonal norm of L by (centre calendar month, tau), previous N_YEARS years
    fl["year"] = fl.date.dt.year
    ym = fl.groupby(["cm", "tau", "year"]).L.mean().rename("Lm").reset_index()
    parts = []
    for Y in range(2011, 2022):
        prev = ym[(ym.year >= Y - N_YEARS) & (ym.year < Y)]
        nm = prev.groupby(["cm", "tau"]).Lm.mean().rename("Lnorm").reset_index()
        nm["year"] = Y
        parts.append(nm)
    norms = pd.concat(parts)
    fl = fl.merge(norms, on=["cm", "tau", "year"], how="left")
    fl["fbench"] = fl.L - fl.Lnorm
    return fl
