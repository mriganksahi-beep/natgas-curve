"""Detect exchange DEFAULT settlements in G4X March options (orchestrator warning, v5_implied_corr finding).

Per quote: Bachelier implied normal vol (r = 0) from the OTM settle at T = trading days to option expiry/252.
A quote is "pinned" if its normal vol lies in [1.15, 1.21] $/MMBtu/sqrt(yr) (the default level ~1.18).
Per (date, Y):
  pin_share    share of usable OTM quotes (price >= 0.002) that are pinned
  nv_disp      IQR of normal vol across those quotes
  vol_day      total G4X March volume that day (NaN-safe)
Day is flagged DEFAULT if pin_share >= 0.5, or (nv_disp < 0.03 and |median nv - 1.18| < 0.05).
Strike-level: pinned quotes are also dropped from fits on non-default days (variant "drop pinned").
"""
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm

from src.research.v5_implied_storage import data as D, rnd as R

LO, HI = 1.15, 1.21


def bach_price(F, K, T, s, isc):
    v = s * np.sqrt(T)
    d = (F - K) / v
    c = (F - K) * ndtr(d) + v * norm.pdf(d)
    return np.where(isc, c, c - (F - K))


def bach_iv(px, F, K, T, isc, lo=1e-4, hi=20.0, n=60):
    a, b = np.full(len(px), lo), np.full(len(px), hi)
    for _ in range(n):
        m = 0.5 * (a + b)
        up = bach_price(F, K, T, m, isc) < px
        a, b = np.where(up, m, a), np.where(up, b, m)
    out = 0.5 * (a + b)
    out[out > hi * 0.999] = np.nan
    return out


def flag(rnd_daily):
    o = D.load_options()
    vol = o.groupby(["date", "Y"]).volume.sum(min_count=1)
    oi = o.groupby(["date", "Y"]).oi.sum(min_count=1)
    rows = []
    for _, r in rnd_daily.iterrows():
        K, px, isc = r.quotes
        ok = px >= 0.002
        T = max(r["T"], 1 / 252)
        iv = bach_iv(px[ok], r.F, K[ok], T, isc[ok]) if ok.any() else np.array([])
        iv = iv[np.isfinite(iv)]
        pin = ((iv >= LO) & (iv <= HI)).mean() if len(iv) else np.nan
        disp = (np.percentile(iv, 75) - np.percentile(iv, 25)) if len(iv) >= 2 else np.nan
        med = np.median(iv) if len(iv) else np.nan
        dflt = (pin >= 0.5) or (np.isfinite(disp) and disp < 0.03 and abs(med - 1.18) < 0.05)
        rows.append({"date": r.date, "Y": r.Y, "n_iv": len(iv), "pin_share": pin, "nv_disp": disp,
                     "nv_med": med, "default": bool(dflt),
                     "vol_day": vol.get((r.date, r.Y), np.nan), "oi_day": oi.get((r.date, r.Y), np.nan)})
    return pd.DataFrame(rows)


def pinned_mask(r):
    K, px, isc = r.quotes
    T = max(r["T"], 1 / 252)
    iv = np.full(len(K), np.nan)
    ok = px >= 0.002
    if ok.any():
        iv[ok] = bach_iv(px[ok], r.F, K[ok], T, isc[ok])
    return (iv >= LO) & (iv <= HI)
